"""Screen fused KDA execution at 2–8 tokens against dense and decode-to-scratch controls."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from functools import partial
from pathlib import Path
from typing import Any

import torch
from bf16_prefetch import Projection, decode_into
from diagnose_bf16_attention import write
from diagnose_bf16_kda import compare
from diagnose_bf16_overlap import capture
from prepare_bf16_attention import LIBRARY_SHA, RUNTIME_SHA, rebuild_weight, tensor_digest
from safetensors.torch import load_file


def main() -> None:
    """Retain fixed error/memory gates and avoid modifying the serving model or adapter."""
    from benchmark_bf16_head import _paired
    from mxwave_zipserv import runtime

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--fixtures", type=Path,
                        default=Path("/audit/bf16-attention-v1/artifacts-v2"))
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("Preserve previous small-batch component results")
    prepared_path = args.fixtures / "prepared.json"
    prepared = json.loads(prepared_path.read_bytes())
    cases = [c for c in prepared["cases"] if c["name"].startswith("kda-")]
    if prepared["status"] != "packed_gpu_verification_pending" or len(cases) != 2:
        raise ValueError("Require the existing source-bound KDA input and output fixtures")
    if (hashlib.sha256(Path(os.environ["MXWAVE_ZIPSERV_LIBRARY"]).read_bytes()).hexdigest()
            != LIBRARY_SHA or hashlib.sha256(Path(runtime.__file__).read_bytes()).hexdigest()
            != RUNTIME_SHA or torch.cuda.get_device_capability() != (12, 1)):
        raise ValueError("Require the qualified native library/Python ABI on SM121")
    for name, expected in (("config.json", prepared["source_config_sha256"]),
                           ("model.safetensors.index.json", prepared["source_index_sha256"])):
        if hashlib.sha256((args.source / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Prepared source metadata changed")
    if hashlib.sha256(Path(__file__).with_name("prepare_bf16_attention.py").read_bytes()
                      ).hexdigest() != prepared["code_sha256"]:
        raise ValueError("Prepared source reconstruction code changed")
    index = json.loads((args.source / "model.safetensors.index.json").read_bytes())["weight_map"]
    torch.set_num_threads(2)
    torch.manual_seed(743)
    device = torch.device("cuda", 0)
    torch.cuda.set_per_process_memory_fraction(
        1.5 * 1024**3 / torch.cuda.get_device_properties(device).total_memory, device)
    started = time.monotonic()
    report: dict[str, Any] = {
        "status": "running", "cases": [], "split_k": 1, "tokens": [1, 2, 3, 4, 7, 8],
        "gates": {"decoded_weight_bits_exact": True, "maximum_relative_l2": .001,
                  "fallback_output_bits_exact": True, "replay_allocation_growth_bytes": 0},
        "code_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                        for name in ("diagnose_bf16_small_batch.py", "bf16_prefetch.py",
                                     "prepare_bf16_attention.py", "diagnose_bf16_kda.py")},
        "library_sha256": LIBRARY_SHA, "runtime_sha256": RUNTIME_SHA,
        "prepared_sha256": hashlib.sha256(prepared_path.read_bytes()).hexdigest(),
        "limits": ["Existing layer-0/rank-0 KDA source projections; no adapter policy change.",
                   "Random and basis activations; not captured model activations.",
                   "All three representations coexist only for this component comparison.",
                   "Graph/event component times are not request-level throughput.",
                   "TP collectives, other layers and whole-model quality remain separate gates."]}
    write(args.report, report)
    stream = torch.cuda.Stream()
    try:
        for case in cases:
            artifact = args.fixtures / case["artifact"]
            if hashlib.sha256(artifact.read_bytes()).hexdigest() != case["artifact_sha256"]:
                raise ValueError("A prepared compressed fixture changed")
            cpu = rebuild_weight(args.source, index, case)
            if tensor_digest(cpu) != case["logical_weight_sha256"]:
                raise ValueError("Independent source reconstruction changed")
            dense = cpu.cuda()
            del cpu
            buffers = load_file(artifact)
            packed = runtime.PackedWeight(
                tuple(buffers[name].cuda() for name in runtime.BUFFER_NAMES),
                *case["native_metadata"])
            del buffers
            restored = runtime.decompress(packed).cpu()
            if tensor_digest(restored) != case["padded_weight_sha256"]:
                raise RuntimeError("Compressed fixture does not restore exact BF16 weight bytes")
            del restored
            rows, width = case["logical_shape"]
            scratch = torch.empty((packed.m, packed.k), device=device, dtype=torch.bfloat16)
            scratch.record_stream(stream)
            projection = Projection(case["name"], packed, rows)
            entry: dict[str, Any] = {"name": case["name"], "logical_shape": case["logical_shape"],
                "gpu_decode_bits_exact": True, "raw_bytes": case["raw_bytes"],
                "packed_bytes": packed.storage_bytes, "scratch_bytes": scratch.numel() * 2,
                "measurements": []}
            report["cases"].append(entry)
            for tokens in report["tokens"]:
                x = torch.randn((tokens, width), dtype=torch.bfloat16, device=device) * .1
                baseline = partial(torch.nn.functional.linear, x, dense)

                def fused(value: torch.Tensor = x, storage: Any = packed,
                          count: int = rows) -> torch.Tensor:
                    return runtime.linear(value, storage, split_k=1)[..., :count].contiguous()

                def fallback(value: torch.Tensor = x, component: Projection = projection,
                             buffer: torch.Tensor = scratch, count: int = rows) -> torch.Tensor:
                    decode_into(component, buffer)
                    return torch.nn.functional.linear(value, buffer[:count])

                dense_graph, expected = capture(baseline, stream)
                fused_graph, actual = capture(fused, stream)
                fallback_graph, inflated = capture(fallback, stream)
                record: dict[str, Any] = {"tokens": tokens,
                    "fused_eager": compare(fused(), baseline(), exact=False),
                    "fused_graph": compare(actual, expected, exact=False),
                    "fallback_graph": compare(inflated, expected, exact=True)}
                scratch.fill_(float("nan"))
                before = torch.cuda.memory_allocated()
                for _ in range(3):
                    fused_graph.replay()
                    fallback_graph.replay()
                record["replay_fused"] = compare(actual, expected, exact=False)
                record["replay_fallback"] = compare(inflated, expected, exact=True)
                record["replay_allocation_growth_bytes"] = torch.cuda.memory_allocated() - before
                if record["replay_allocation_growth_bytes"]:
                    raise RuntimeError("Small-batch graph replay grew GPU storage")
                with torch.cuda.stream(stream):
                    streamed = fused()
                torch.cuda.current_stream().wait_stream(stream)
                record["nondefault_stream"] = compare(streamed, expected, exact=False)
                del streamed
                basis = torch.zeros_like(x)
                basis[torch.arange(tokens), torch.arange(tokens)] = 1
                record["basis"] = compare(
                    runtime.linear(basis, packed, split_k=1)[..., :rows].contiguous(),
                    torch.nn.functional.linear(basis, dense), exact=False)
                record["graph_fused_vs_dense"] = _paired(
                    dense_graph.replay, fused_graph.replay, 5, 10, interleaved=True)
                record["graph_fallback_vs_dense"] = _paired(
                    dense_graph.replay, fallback_graph.replay, 5, 10, interleaved=True)
                record["eager_fused_vs_dense"] = _paired(baseline, fused, 5, 5, interleaved=True)
                record["eager_fallback_vs_dense"] = _paired(baseline, fallback, 5, 5,
                                                             interleaved=True)
                entry["measurements"].append(record)
                write(args.report, report)
                del dense_graph, fused_graph, fallback_graph, expected, actual, inflated
                del x, basis, baseline, fused, fallback
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()
            del dense, packed, projection, scratch
            gc.collect()
            torch.cuda.empty_cache()
        report.update(status="passed", passed=True, seconds=time.monotonic() - started,
                      cuda_peak_bytes=torch.cuda.max_memory_allocated())
        write(args.report, report)
        print(json.dumps({k: report[k] for k in ("status", "seconds", "cuda_peak_bytes")}))
    except BaseException as exc:
        report.update(status="failed", passed=False, error=repr(exc))
        write(args.report, report)
        raise


if __name__ == "__main__":
    main()
