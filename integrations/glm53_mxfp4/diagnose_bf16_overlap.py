"""Compare serialized and asynchronous bounded BF16 prefetch without a model reload."""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import os
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import torch
from bf16_prefetch import PrefillPipeline, Projection, decode_into, interval_overlap
from diagnose_bf16_attention import write
from prepare_bf16_attention import LIBRARY_SHA, RUNTIME_SHA, rebuild_weight, tensor_digest
from safetensors.torch import load_file


def require_exact(actual: tuple[torch.Tensor, ...], expected: tuple[torch.Tensor, ...]) -> None:
    """Require every intermediate output to be finite and byte-identical."""
    if len(actual) != len(expected) or any(
        not bool(torch.isfinite(a).all()) or a.shape != b.shape
        or not torch.equal(a.view(torch.uint8), b.view(torch.uint8))
        for a, b in zip(actual, expected, strict=True)
    ):
        raise RuntimeError("Prefetch changed a stock-GEMM intermediate output")


def capture(operation: Callable[[], Any], stream: torch.cuda.Stream) -> tuple[Any, Any]:
    """Reuse one warmup/capture stream to bound per-stream cuBLAS workspace retention."""
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(20):
            operation()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outputs = operation()
    graph.replay()
    return graph, outputs


def trace_overlap(operations: dict[str, Callable[[], Any]], path: Path) -> dict[str, Any]:
    """Preserve real two-stream execution and measure decoder/GEMM interval intersections."""
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
                with torch.profiler.record_function("pipeline/" + label):
                    operation()
                    torch.cuda.synchronize()
    temporary = path.with_suffix(".json")
    profiler.export_chrome_trace(str(temporary))
    events = json.loads(temporary.read_bytes())["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"]
    scopes = []
    for event in events:
        if (event.get("ph") != "X" or event.get("cat") != "user_annotation"
                or not event.get("name", "").startswith("pipeline/")):
            continue
        first, last = event["ts"], event["ts"] + event["dur"]
        selected = [k for k in kernels if first <= k["ts"] and k["ts"] + k["dur"] <= last]
        decodes = [(k["ts"], k["ts"] + k["dur"]) for k in selected
                   if "Decompress_Kernel" in k["name"]]
        gemms = [(k["ts"], k["ts"] + k["dur"]) for k in selected
                 if any(word in k["name"] for word in ("nvjet_", "gemvx::kernel", "gemm", "mma"))]
        decode_us = sum(b - a for a, b in decodes)
        overlap_us = interval_overlap(decodes, gemms)
        scopes.append({"path": event["name"].removeprefix("pipeline/"),
                       "decode_us": decode_us, "gemm_us": sum(b - a for a, b in gemms),
                       "decoder_gemm_overlap_us": overlap_us,
                       "decode_overlap_fraction": overlap_us / decode_us if decode_us else 0,
                       "decoder_streams": sorted({k["args"].get("stream") for k in selected
                                                 if "Decompress_Kernel" in k["name"]}),
                       "gpu_kernel_span_us": (max(k["ts"] + k["dur"] for k in selected)
                                              - min(k["ts"] for k in selected)) if selected else 0,
                       "kernel_count": len(selected)})
    with temporary.open("rb") as source, gzip.open(path, "wb") as destination:
        while block := source.read(1024**2):
            destination.write(block)
    temporary.unlink()
    if len(scopes) != 9 or any(not s["kernel_count"] for s in scopes):
        raise RuntimeError("Profiler did not bind all nine pipeline scopes to CUDA kernels")
    return {"file": path.name, "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "scopes": scopes,
            "note": "Three profiled calls per path; durations are separate from timing medians"}


def main() -> None:
    """Gate exactness and bounded storage before comparing paired eager and graph latency."""
    from benchmark_bf16_head import _paired
    from mxwave_zipserv import runtime

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("Preserve earlier asynchronous component attempts")
    fixtures = Path("/audit/bf16-attention-v1/artifacts-v2")
    prepared = json.loads((fixtures / "prepared.json").read_bytes())
    if prepared["status"] != "packed_gpu_verification_pending" or len(prepared["cases"]) != 3:
        raise ValueError("Require the three immutable real-projection fixtures")
    if (hashlib.sha256(Path(os.environ["MXWAVE_ZIPSERV_LIBRARY"]).read_bytes()).hexdigest()
            != LIBRARY_SHA
            or hashlib.sha256(Path(runtime.__file__).read_bytes()).hexdigest() != RUNTIME_SHA
            or hashlib.sha256(Path(__file__).with_name("prepare_bf16_attention.py").read_bytes()
                              ).hexdigest() != prepared["code_sha256"]):
        raise ValueError("The qualified decoder or source reconstruction changed")
    for name, expected in (("config.json", prepared["source_config_sha256"]),
                           ("model.safetensors.index.json", prepared["source_index_sha256"])):
        if hashlib.sha256((args.source / name).read_bytes()).hexdigest() != expected:
            raise ValueError("The audited model metadata changed")
    torch.set_num_threads(2)
    torch.manual_seed(728)
    device = torch.device("cuda", 0)
    free, total = torch.cuda.mem_get_info(device)
    if free < 2.5 * 1024**3:
        raise MemoryError("Insufficient idle device headroom for the bounded component")
    torch.cuda.set_per_process_memory_fraction(1.5 * 1024**3 / total, device)
    index = json.loads((args.source / "model.safetensors.index.json").read_bytes())["weight_map"]
    report: dict[str, Any] = {
        "status": "running", "cases": [], "source_checks": [], "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(device), "library_sha256": LIBRARY_SHA,
        "runtime_sha256": RUNTIME_SHA,
        "source_code_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()
                              ).hexdigest() for name in ("bf16_prefetch.py",
                                                        "diagnose_bf16_overlap.py")},
        "prepared_sha256": hashlib.sha256((fixtures / "prepared.json").read_bytes()).hexdigest(),
        "enqueue_order": "Current GEMM before next decode",
        "capture_strategy": "One reused warmup/capture stream and one reused caller stream",
        "gates": {"decoded_weight_bits_exact": True, "all_intermediate_output_bits_exact": True,
                  "maximum_pipeline_weight_storage_bytes": 256 * 1024**2,
                  "graph_replay_allocation_growth_bytes": 0},
        "limits": ["Three real matrices with an authored dependent activation bridge; not GLM.",
                   "First 4096 output columns feed the next matrix; real attention is omitted.",
                   "All intermediate outputs are retained deliberately for exactness checks.",
                   "The scratch pool exceeds savings of this three-matrix fixture by itself.",
                   "A full model must amortize the pool across more compressed projections.",
                   "Primed-first timings exclude its separately measured initial decode.",
                   "Both arms coexist for comparison; peak is not a net production allocation.",
                   "Model prefill is eager; graph replay is a separate component measurement."]}
    write(args.report, report)
    started = time.monotonic()
    dense_weights, projections = [], []
    try:
        for case in (prepared["cases"][0], prepared["cases"][2], prepared["cases"][1]):
            artifact = fixtures / case["artifact"]
            if hashlib.sha256(artifact.read_bytes()).hexdigest() != case["artifact_sha256"]:
                raise ValueError("Packed projection artifact changed")
            cpu = rebuild_weight(args.source, index, case)
            if tensor_digest(cpu) != case["logical_weight_sha256"]:
                raise ValueError("Independent source reconstruction changed")
            dense_weights.append(cpu.cuda())
            del cpu
            buffers = load_file(artifact)
            packed = runtime.PackedWeight(
                tuple(buffers[name].cuda() for name in runtime.BUFFER_NAMES),
                *case["native_metadata"])
            del buffers
            projection = Projection(case["name"], packed, case["logical_shape"][0])
            check = torch.empty((packed.m, packed.k), dtype=torch.bfloat16, device=device)
            decode_into(projection, check)
            exact = tensor_digest(check.cpu()) == case["padded_weight_sha256"]
            report["source_checks"].append({"name": case["name"], "bits_exact": exact})
            del check
            if not exact:
                raise RuntimeError("Caller-owned destination changed decoded GPU bytes")
            projections.append(projection)
        weights, specs = tuple(dense_weights), tuple(projections)
        capture_stream, caller = torch.cuda.Stream(), torch.cuda.Stream()
        report["raw_weight_bytes"] = sum(w.numel() * w.element_size() for w in weights)
        report["packed_weight_bytes"] = sum(p.packed.storage_bytes for p in specs)
        for prime in (False, True):
            pipeline = PrefillPipeline(specs, prime_first=prime)
            arena: dict[str, Any] = {
                "first_weight_primed": prime, "first_prime_ms": pipeline.prime_milliseconds,
                "scratch_and_cached_weight_bytes": pipeline.weight_storage_bytes,
                "slot_bytes": [b.numel() * b.element_size() for b in pipeline.buffers],
                "cached_first_bytes": (pipeline.first.numel() * 2
                                       if pipeline.first is not None else 0),
                "measurements": [], "traces": []}
            report["cases"].append(arena)
            for tokens in (32, 128, 512, 2048):
                x = torch.randn((tokens, 4096), device=device, dtype=torch.bfloat16) * .1

                def dense(value: torch.Tensor = x,
                          matrices: tuple[torch.Tensor, ...] = weights) -> tuple[torch.Tensor, ...]:
                    outputs = []
                    for i, weight in enumerate(matrices):
                        output = torch.nn.functional.linear(value, weight)
                        outputs.append(output)
                        if i + 1 < len(matrices):
                            value = output[..., :4096].contiguous()
                    return tuple(outputs)

                serial = partial(pipeline.forward, x, overlap=False)
                asynchronous = partial(pipeline.forward, x, overlap=True)
                dense_graph, expected = capture(dense, capture_stream)
                serial_graph, serial_out = capture(serial, capture_stream)
                async_graph, async_out = capture(asynchronous, capture_stream)
                require_exact(serial_out, expected)
                require_exact(async_out, expected)
                require_exact(serial(), expected)
                require_exact(asynchronous(), expected)
                before = torch.cuda.memory_allocated()
                pointers = [buffer.data_ptr() for buffer in pipeline.buffers]
                for _ in range(3):
                    x.normal_(std=.1)
                    for buffer in pipeline.buffers:
                        buffer.fill_(float("nan"))
                    dense_graph.replay()
                    async_graph.replay()
                    async_graph.replay()
                    require_exact(async_out, expected)
                torch.cuda.synchronize()
                growth = torch.cuda.memory_allocated() - before
                if growth or pointers != [buffer.data_ptr() for buffer in pipeline.buffers]:
                    raise RuntimeError("Repeated graph replay changed the bounded storage")
                caller.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(caller):
                    nondefault_out = asynchronous()
                torch.cuda.current_stream().wait_stream(caller)
                require_exact(nondefault_out, expected)
                del nondefault_out
                row: dict[str, Any] = {
                    "tokens": tokens, "all_intermediate_outputs_bits_exact": True,
                    "poisoned_pool_replay_pairs": 3, "nondefault_caller_stream_exact": True,
                    "replay_allocation_growth_bytes": growth, "graph_replay": {},
                    "eager_cuda_events": {}, "cuda_allocated_at_timing": torch.cuda.memory_allocated()}
                for label, operations in (
                    ("graph_replay", (dense_graph.replay, serial_graph.replay, async_graph.replay)),
                    ("eager_cuda_events", (dense, serial, asynchronous)),
                ):
                    for name, first, second in (("dense_to_serial", 0, 1),
                                                ("serial_to_async", 1, 2),
                                                ("dense_to_async", 0, 2)):
                        row[label][name] = _paired(operations[first], operations[second],
                                                  5, 10 if label == "graph_replay" else 5,
                                                  interleaved=True)
                arena["measurements"].append(row)
                write(args.report, report)
                del dense_graph, serial_graph, async_graph, expected, serial_out, async_out
                gc.collect()
                torch.cuda.empty_cache()
                if tokens in (128, 2048):
                    trace_path = args.report.parent / f"prime{int(prime)}-m{tokens}.trace.json.gz"
                    arena["traces"].append(trace_overlap({
                        "dense": dense,
                        "serial": partial(pipeline.forward, x, overlap=False, annotate=True),
                        "async": partial(pipeline.forward, x, overlap=True, annotate=True)},
                        trace_path))
                    write(args.report, report)
                del x, dense, serial, asynchronous
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()
            del pipeline
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
