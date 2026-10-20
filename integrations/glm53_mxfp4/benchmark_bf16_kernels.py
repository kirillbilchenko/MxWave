"""Separate fused BF16 compression costs from the same-weight stock GEMM on SM121."""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import os
import statistics
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from diagnose_bf16_attention import write
from diagnose_bf16_overlap import capture
from prepare_bf16_attention import LIBRARY_SHA, RUNTIME_SHA, rebuild_weight, tensor_digest
from safetensors.torch import load_file

Operation = Callable[[], torch.Tensor]


def digest(path: Path) -> str:
    """Hash artifacts without allocating another full head-sized CPU buffer."""
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def arithmetic(actual: torch.Tensor, expected: torch.Tensor, *, exact: bool) -> dict[str, Any]:
    """Record the existing strict KDA or 0.1% relative-L2 head/single-token gate."""
    a, b = actual.detach().cpu(), expected.detach().cpu()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    bits = a.shape == b.shape and torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    relative = float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30))
    return {"bits_exact": bits, "finite": finite, "relative_l2": relative if finite else None,
            "requires_bits_exact": exact, "passed": finite and (bits if exact else relative <= .001)}


def paired(baseline: Callable[[], Any], candidate: Callable[[], Any], *, trials: int,
           iterations: int) -> dict[str, Any]:
    """Alternate both arm orders, retaining individual GPU-event and synchronized wall times."""
    records = []
    for trial in range(trials):
        order = ("dense", "fused") if trial % 2 == 0 else ("fused", "dense")
        record: dict[str, Any] = {"order": list(order)}
        for name in order:
            operation = baseline if name == "dense" else candidate
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            wall = time.perf_counter_ns()
            start.record()
            for _ in range(iterations):
                operation()
            end.record()
            end.synchronize()
            record[name + "_cuda_us"] = start.elapsed_time(end) * 1000 / iterations
            record[name + "_wall_us"] = (time.perf_counter_ns() - wall) / iterations / 1000
        record["cuda_ratio"] = record["fused_cuda_us"] / record["dense_cuda_us"]
        records.append(record)
    return {"trials": trials, "iterations": iterations, "samples": records,
            "dense_median_us": statistics.median(r["dense_cuda_us"] for r in records),
            "fused_median_us": statistics.median(r["fused_cuda_us"] for r in records),
            "median_paired_ratio": statistics.median(r["cuda_ratio"] for r in records)}


def fixtures(root: Path) -> list[dict[str, Any]]:
    """Reuse immutable rank-zero KDA and head artifacts; never repack or download weights."""
    attention = root / "bf16-attention-v1/artifacts-v2"
    manifest = json.loads((attention / "prepared.json").read_bytes())
    if (manifest["status"] != "packed_gpu_verification_pending"
            or digest(Path(__file__).with_name("prepare_bf16_attention.py")) != manifest["code_sha256"]):
        raise ValueError("KDA preparation or reconstruction code changed")
    result = [{**case, "directory": attention} for case in manifest["cases"]
              if case["name"].startswith("kda-")]
    vocab = root / "vocab-v2"
    heads = json.loads((vocab / "prepared.json").read_bytes())
    if heads["status"] != "packed_gpu_verification_pending":
        raise ValueError("Require completed vocabulary CPU preparation")
    head = next(row for row in heads["cases"] if (row["kind"], row["rank"]) == ("head", 0))
    result.append({**head, "name": "head-r0", "directory": vocab,
        "logical_shape": head["shape"], "native_shape": head["shape"],
        "logical_weight_sha256": head["source_sha256"],
        "padded_weight_sha256": head["source_sha256"], "native_metadata": head["metadata"],
        "source_tensors": [{"name": head["source_key"], "axis": 0,
                            "range": [0, head["shape"][0]]}]})
    if [row["name"] for row in result] != ["kda-input-l0-r0", "kda-output-l0-r0", "head-r0"]:
        raise ValueError("Require exactly the three traced projection shapes")
    return result


def main() -> None:
    """Benchmark unprofiled pairs first; optionally collect separately labeled NCU launches."""
    from mxwave_zipserv import runtime
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.layers.utils import dispatch_unquantized_gemm

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/audit"))
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=12)
    args = parser.parse_args()
    if args.report.exists() or args.trials < 4 or args.iterations < 1:
        raise ValueError("Preserve old evidence and require both arm orders")
    if (digest(Path(os.environ["MXWAVE_ZIPSERV_LIBRARY"])) != LIBRARY_SHA
            or digest(Path(runtime.__file__)) != RUNTIME_SHA):
        raise ValueError("Qualified native library or Python ABI changed")
    source = args.root / "source"
    from kda_policy import CONFIG_SHA, INDEX_SHA

    for name, expected in (("config.json", CONFIG_SHA), ("model.safetensors.index.json", INDEX_SHA)):
        if digest(source / name) != expected:
            raise ValueError("Source metadata changed")
    torch.set_num_threads(2)
    torch.manual_seed(787)
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    if (properties.major, properties.minor) != (12, 1):
        raise ValueError("This experiment is qualified only on SM121")
    torch.cuda.set_per_process_memory_fraction(2.5 * 1024**3 / properties.total_memory)
    index = json.loads((source / "model.safetensors.index.json").read_bytes())["weight_map"]
    stream = torch.cuda.Stream()
    method = UnquantizedLinearMethod()
    report: dict[str, Any] = {"status": "running", "profile": args.profile, "cases": [],
        "library_sha256": LIBRARY_SHA, "runtime_sha256": RUNTIME_SHA,
        "code_sha256": digest(Path(__file__)), "torch": torch.__version__,
        "stock_dispatch": dispatch_unquantized_gemm().__name__,
        "stock_dispatch_source_sha256": digest(Path(inspect.getfile(dispatch_unquantized_gemm))),
        "source_metadata": {"config.json": CONFIG_SHA, "model.safetensors.index.json": INDEX_SHA},
        "device": {"name": properties.name, "sm_count": properties.multi_processor_count,
                   "l2_cache_bytes": getattr(properties, "L2_cache_size", None)},
        "manifests": {str(p.relative_to(args.root)): digest(p) for p in (
            args.root / "bf16-attention-v1/artifacts-v2/prepared.json",
            args.root / "vocab-v2/prepared.json")},
        "limits": ["Layer-zero/rank-zero KDA and rank-zero head, random/basis activations.",
            "Both representations coexist for paired timing; this is not retained model memory.",
            "No full-model reload, TP communication or request throughput measurement.",
            "Split-K above one is diagnostic only; no adapter defaults are changed.",
            "Profiled hardware-counter durations are separate from unprofiled timing medians.",
            "The fused call matches the inner runtime kernel, excluding the outer KDA wrapper."]}
    write(args.report, report)
    started = time.monotonic()
    try:
        for case in fixtures(args.root):
            artifact = case["directory"] / case["artifact"]
            if artifact.name != case["artifact"] or digest(artifact) != case["artifact_sha256"]:
                raise ValueError("Prepared compressed artifact changed")
            cpu = rebuild_weight(source, index, case)
            if tensor_digest(cpu) != case["logical_weight_sha256"]:
                raise ValueError("Independent source reconstruction changed")
            dense = cpu.cuda()
            del cpu
            buffers = load_file(artifact)
            packed = runtime.PackedWeight(tuple(buffers[n].cuda() for n in runtime.BUFFER_NAMES),
                                          *case["native_metadata"])
            del buffers
            restored = runtime.decompress(packed)
            # Chunk the GPU-to-host hash to avoid a second complete head-sized CPU allocation.
            restored_sha = hashlib.sha256()
            for first in range(0, packed.m, 2048):
                restored_sha.update(restored[first:first + 2048].cpu().view(torch.uint8).numpy())
            if restored_sha.hexdigest() != case["padded_weight_sha256"]:
                raise RuntimeError("GPU decode did not restore every source weight byte")
            del restored
            rows, width = case["logical_shape"]
            entry: dict[str, Any] = {"name": case["name"], "logical_shape": [rows, width],
                "native_shape": [packed.m, packed.k], "raw_bytes": case["raw_bytes"],
                "packed_bytes": packed.storage_bytes, "gpu_weight_bits_exact": True,
                "source_weight_sha256": case["logical_weight_sha256"],
                "artifact_sha256": case["artifact_sha256"], "measurements": []}
            report["cases"].append(entry)
            for tokens in ((4,) if args.profile else (1, 4, 8)):
                x = torch.randn((tokens, width), device="cuda", dtype=torch.bfloat16) * .1
                baseline: Operation = partial(method.apply, SimpleNamespace(weight=dense), x)
                reference = baseline()
                splits = (1, 4) if args.profile else (1, 2, 4, 8)
                for split in splits:
                    def fused(value: torch.Tensor = x, storage: Any = packed,
                              count: int = rows, split_k: int = split) -> torch.Tensor:
                        return runtime.linear(value, storage, split_k=split_k)[..., :count].contiguous()

                    strict = case["name"].startswith("kda-") and tokens > 1
                    record: dict[str, Any] = {"tokens": tokens, "split_k": split,
                        "arithmetic": arithmetic(fused(), reference, exact=strict)}
                    basis = torch.zeros_like(x)
                    basis[torch.arange(tokens), torch.arange(tokens)] = 1
                    record["basis"] = arithmetic(
                        fused(basis), method.apply(SimpleNamespace(weight=dense), basis), exact=strict)
                    if split == 1 and not all(record[k]["passed"] for k in ("arithmetic", "basis")):
                        raise RuntimeError("Existing split-K=1 arithmetic qualification regressed")
                    if args.profile:
                        for arm, operation in (("dense", baseline), (f"fused_s{split}", fused)):
                            # The dense arm is profiled once per case, not once per split.
                            if arm == "dense" and split != 1:
                                continue
                            for _ in range(8):
                                operation()
                            torch.cuda.synchronize()
                            label = f"bf16_profile/{case['name']}/M{tokens}/{arm}"
                            torch.cuda.cudart().cudaProfilerStart()
                            torch.cuda.nvtx.range_push(label)
                            output = operation()
                            torch.cuda.nvtx.range_pop()
                            torch.cuda.synchronize()
                            torch.cuda.cudart().cudaProfilerStop()
                            del output
                            record.setdefault("profile_labels", []).append(label)
                    else:
                        dg, expected = capture(baseline, stream)
                        fg, actual = capture(fused, stream)
                        record["graph_arithmetic"] = arithmetic(actual, expected, exact=strict)
                        allocated = torch.cuda.memory_allocated()
                        for _ in range(3):
                            fg.replay()
                        torch.cuda.synchronize()
                        record["replay_allocation_growth_bytes"] = torch.cuda.memory_allocated() - allocated
                        with torch.cuda.stream(stream):
                            streamed = fused()
                        torch.cuda.current_stream().wait_stream(stream)
                        record["nondefault_stream"] = arithmetic(streamed, reference, exact=strict)
                        del streamed
                        record["graph"] = paired(dg.replay, fg.replay, trials=args.trials,
                                                  iterations=args.iterations)
                        record["eager"] = paired(baseline, fused, trials=args.trials,
                                                  iterations=args.iterations)
                        record["candidate_gates_passed"] = all(
                            record[k]["passed"] for k in ("arithmetic", "basis", "graph_arithmetic",
                                                          "nondefault_stream")) and not record[
                                                              "replay_allocation_growth_bytes"]
                        if split == 1 and not record["candidate_gates_passed"]:
                            raise RuntimeError("Existing split-K=1 stream/graph qualification regressed")
                        del dg, fg, expected, actual
                    entry["measurements"].append(record)
                    write(args.report, report)
                    del basis
                del x, reference, baseline, fused
                gc.collect()
                torch.cuda.empty_cache()
            del dense, packed
            gc.collect()
            torch.cuda.empty_cache()
        report.update(status="passed", seconds=time.monotonic() - started,
                      cuda_peak_bytes=torch.cuda.max_memory_allocated())
        write(args.report, report)
        print(json.dumps({k: report[k] for k in ("status", "seconds", "cuda_peak_bytes")}))
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        write(args.report, report)
        raise


if __name__ == "__main__":
    main()
