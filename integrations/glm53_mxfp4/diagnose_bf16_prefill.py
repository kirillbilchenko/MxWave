"""Measure bounded decode-to-scratch plus stock GEMM and preserve component GPU traces."""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from diagnose_bf16_attention import write
from prepare_bf16_attention import LIBRARY_SHA, RUNTIME_SHA, rebuild_weight, tensor_digest
from safetensors.torch import load_file


def profile_paths(operations: dict[str, Callable[[], torch.Tensor]], path: Path) -> dict[str, Any]:
    """Trace each path after timing, with explicit synchronization and bounded repetitions."""
    for operation in operations.values():
        for _ in range(3):
            operation()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False, with_stack=False,
    ) as profiler:
        for _ in range(3):
            for label, operation in operations.items():
                with torch.profiler.record_function(label):
                    operation()
                    torch.cuda.synchronize()
    temporary = path.with_suffix(".json")
    profiler.export_chrome_trace(str(temporary))
    trace = json.loads(temporary.read_bytes())
    kernels: dict[str, dict[str, float | int]] = {}
    for event in trace["traceEvents"]:
        if event.get("cat") != "kernel" or event.get("ph") != "X":
            continue
        item = kernels.setdefault(event["name"], {"calls": 0, "sum_us": 0.0})
        item["calls"] += 1
        item["sum_us"] += event["dur"]
    with temporary.open("rb") as source, gzip.open(path, "wb") as destination:
        while block := source.read(1024**2):
            destination.write(block)
    temporary.unlink()
    return {"file": path.name, "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "kernel_totals": kernels,
            "note": "Three calls per path, sequential scopes; profiled durations are not timing medians"}


def main() -> None:
    """Require exact decoded weights and stock-GEMM outputs before interpreting speed."""
    from benchmark_bf16_head import _graph, _paired
    from mxwave_zipserv import runtime

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("Preserve previous prefill component evidence")
    fixtures = Path("/audit/bf16-attention-v1/artifacts-v2")
    prepared = json.loads((fixtures / "prepared.json").read_bytes())
    if prepared["status"] != "packed_gpu_verification_pending" or len(prepared["cases"]) != 3:
        raise ValueError("Require the three source-bound packed fixtures")
    if (hashlib.sha256(Path(os.environ["MXWAVE_ZIPSERV_LIBRARY"]).read_bytes()).hexdigest()
            != LIBRARY_SHA
            or hashlib.sha256(Path(runtime.__file__).read_bytes()).hexdigest() != RUNTIME_SHA
            or hashlib.sha256(Path(__file__).with_name("prepare_bf16_attention.py").read_bytes()
                              ).hexdigest() != prepared["code_sha256"]):
        raise ValueError("The native library, runtime or source reconstruction changed")
    for name, expected in (("config.json", prepared["source_config_sha256"]),
                           ("model.safetensors.index.json", prepared["source_index_sha256"])):
        if hashlib.sha256((args.source / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Source metadata differs from the verified fixtures")
    index = json.loads((args.source / "model.safetensors.index.json").read_bytes())["weight_map"]
    torch.set_num_threads(2)
    torch.manual_seed(721)
    device = torch.device("cuda", 0)
    torch.cuda.set_per_process_memory_fraction(
        1.5 * 1024**3 / torch.cuda.get_device_properties(device).total_memory, device)
    started = time.monotonic()
    report: dict[str, Any] = {
        "status": "running", "cases": [], "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(device), "library_sha256": LIBRARY_SHA,
        "runtime_sha256": RUNTIME_SHA,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "prepared_sha256": hashlib.sha256((fixtures / "prepared.json").read_bytes()).hexdigest(),
        "gate": "Exact GPU decoded weights and exact stock-GEMM output bytes at every shape",
        "limits": ["Component only, random BF16 activations, one Spark.",
                   "Candidate owns a transient full projection, not a persistent dense model.",
                   "Paired comparison retains a separate dense control; peak is not net saving.",
                   "Profiling is separate from five paired timing trials."]}
    write(args.report, report)
    try:
        for case in prepared["cases"]:
            artifact = fixtures / case["artifact"]
            if hashlib.sha256(artifact.read_bytes()).hexdigest() != case["artifact_sha256"]:
                raise ValueError("The packed fixture changed")
            source = rebuild_weight(args.source, index, case)
            if tensor_digest(source) != case["logical_weight_sha256"]:
                raise ValueError("Source reconstruction changed")
            dense = source.cuda()
            del source
            buffers = load_file(artifact)
            packed = runtime.PackedWeight(
                tuple(buffers[name].cuda() for name in runtime.BUFFER_NAMES),
                *case["native_metadata"])
            del buffers
            restored = runtime.decompress(packed).cpu()
            exact = tensor_digest(restored) == case["padded_weight_sha256"]
            del restored
            rows, width = case["logical_shape"]
            entry: dict[str, Any] = {
                "name": case["name"], "gpu_decode_bits_exact": exact,
                "raw_bytes": case["raw_bytes"], "retained_bytes": packed.storage_bytes,
                "scratch_weight_bytes": packed.m * packed.k * 2, "measurements": [], "traces": []}
            report["cases"].append(entry)
            write(args.report, report)
            if not exact:
                raise RuntimeError("GPU decompression changed source weight bits")
            for tokens in (1, 128, 512, 2048):
                x = torch.randn((tokens, width), device=device, dtype=torch.bfloat16) * .1

                def baseline(value: torch.Tensor = x, weight: torch.Tensor = dense) -> torch.Tensor:
                    return torch.nn.functional.linear(value, weight)

                def candidate(value: torch.Tensor = x, storage: Any = packed,
                              count: int = rows) -> torch.Tensor:
                    return torch.nn.functional.linear(value, runtime.decompress(storage)[:count])

                def fused(value: torch.Tensor = x, storage: Any = packed,
                          count: int = rows) -> torch.Tensor:
                    return runtime.linear(value, storage, split_k=4)[..., :count].contiguous()

                dense_graph, expected = _graph(baseline)
                decoded_graph, actual = _graph(candidate)
                same = bool(torch.equal(expected.view(torch.uint8), actual.view(torch.uint8)))
                entry["measurements"].append({
                    "tokens": tokens, "output_bits_exact": same,
                    "graph_replay": _paired(dense_graph.replay, decoded_graph.replay, 5, 10,
                                             interleaved=True),
                    "eager_cuda_events": _paired(baseline, candidate, 5, 5, interleaved=True)})
                write(args.report, report)
                del dense_graph, decoded_graph, expected, actual
                gc.collect()
                torch.cuda.empty_cache()
                if not same:
                    raise RuntimeError("Decode plus identical stock GEMM changed output bytes")
                if tokens in (1, 2048):
                    trace_path = args.report.parent / f"{case['name']}-m{tokens}.trace.json.gz"
                    entry["traces"].append(profile_paths(
                        {"dense": baseline, "decode_then_dense": candidate, "fused_split4": fused},
                        trace_path))
                    write(args.report, report)
                del x, baseline, candidate, fused
            del dense, packed
            gc.collect()
            torch.cuda.empty_cache()
        report.update(status="passed", passed=True, seconds=time.monotonic() - started,
                      cuda_peak_bytes=torch.cuda.max_memory_allocated())
        write(args.report, report)
        print(json.dumps({key: report[key] for key in ("status", "seconds", "cuda_peak_bytes")}))
    except BaseException as exc:
        report.update(status="failed", passed=False, error=repr(exc))
        write(args.report, report)
        raise


if __name__ == "__main__":
    main()
