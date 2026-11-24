"""Time unchanged FP32 partials against two controls while retaining prior precision failures."""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from benchmark_fp32_split import Operation, load_control, raw
from benchmark_kda_memory import digest, telemetry
from fp32_speed_probe import PAIRS, assess, protocol, summarize
from prepare_bf16_attention import LIBRARY_SHA, RUNTIME_SHA, tensor_digest
from safetensors.torch import load_file


def measure(calls: dict[str, Callable[[], Any]], pressure: torch.Tensor,
            *, evict: bool) -> dict[str, Any]:
    """Retain balanced comparisons and separate repeat controls for all three libraries."""
    p = protocol()
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
              for _ in range(p["samples_per_leg"])]
    pairs = {"original_repeat": ("original", "original"),
             "rebuilt_repeat": ("rebuilt", "rebuilt"),
             "candidate_repeat": ("candidate", "candidate"),
             "rebuilt_vs_original": ("original", "rebuilt"),
             "candidate_vs_original": ("original", "candidate"),
             "candidate_vs_rebuilt": ("rebuilt", "candidate")}
    samples = []
    for trial in range(p["trials"]):
        pair_order = list(PAIRS) if trial % 2 == 0 else list(reversed(PAIRS))
        order = ["a", "b", "b", "a"] if trial % 2 == 0 else ["b", "a", "a", "b"]
        row: dict[str, Any] = {"trial": trial, "pair_order": pair_order, "pairs": {},
                               "before": telemetry()}
        for pair in pair_order:
            legs = []
            for arm in order:
                call = calls[pairs[pair][int(arm == "b")]]
                call()
                for start, end in events:
                    if evict:
                        pressure.add_(1)
                    start.record()
                    call()
                    end.record()
                events[-1][1].synchronize()
                legs.append({"arm": arm, "us": [a.elapsed_time(b) * 1000 for a, b in events]})
            row["pairs"][pair] = legs
        row["after"] = telemetry()
        samples.append(row)
    return {"samples": samples, "summary": summarize(samples)}


def graph_call(op: Operation, x: torch.Tensor) -> Callable[[], torch.Tensor]:
    """Include real runtime input padding and logical output cropping in captured execution."""
    def invoke() -> torch.Tensor:
        if op.x.shape[0] != x.shape[0]:
            op.x.zero_()
            op.x[:x.shape[0]].copy_(x)
        return op().contiguous()
    return invoke


def main() -> None:
    """Reuse hash-bound precision inputs and compare outputs before and after paired timing."""
    from mxwave_zipserv import runtime

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("Preserve the first timing attempt")
    plan = json.loads(args.plan.read_bytes())
    p = protocol()
    if plan["protocol"] != p:
        raise ValueError("Timing protocol changed")
    parent_path = args.plan.with_name(f"parent-rank{args.rank}.json")
    if digest(parent_path) != plan["parent_report_sha256"][str(args.rank)]:
        raise ValueError("Prior precision outputs changed")
    parent = json.loads(parent_path.read_bytes())
    if parent["status"] != "complete" or parent["rank"] != args.rank:
        raise ValueError("Require completed rank-specific precision evidence")
    if (digest(Path(os.environ["MXWAVE_ZIPSERV_LIBRARY"])) != LIBRARY_SHA
            or digest(Path(runtime.__file__)) != RUNTIME_SHA):
        raise ValueError("Qualified control/runtime changed")
    build_path = Path(plan["native_build_path"])
    if digest(build_path) != plan["native_build_sha256"]:
        raise ValueError("Native build changed")
    build = json.loads(build_path.read_bytes())
    libraries = {"original": runtime._library()}
    for arm in build["arms"]:
        path = build_path.parent / arm["library"]
        if digest(path) != arm["library_sha256"]:
            raise ValueError("Native library changed")
        if arm["arm"] == "control":
            libraries["rebuilt"] = load_control(runtime, path)
        else:
            library = ctypes.CDLL(str(path))
            if hasattr(library, "mxzip_mm"):
                raise ValueError("Candidate must not impersonate the original GEMM ABI")
            library.mxzip_mm_fp32.argtypes = [*libraries["original"].mxzip_mm.argtypes,
                                              ctypes.c_uint64]
            library.mxzip_mm_fp32.restype = ctypes.c_int
            libraries["candidate"] = library
    torch.set_num_threads(1)
    props = torch.cuda.get_device_properties(0)
    if (props.major, props.minor) != (12, 1):
        raise ValueError("Require SM121")
    torch.cuda.set_per_process_memory_fraction(p["CUDA_allocation_cap_bytes"] / props.total_memory)
    report: dict[str, Any] = {"status": "running", "rank": args.rank, "protocol": p,
        "source_sha256": digest(Path(__file__)), "plan_sha256": digest(args.plan),
        "native_build_sha256": digest(build_path), "qualified_library_sha256": LIBRARY_SHA,
        "runtime_sha256": RUNTIME_SHA, "parent_report_sha256": digest(parent_path),
        "cases": [], "started_epoch": time.time(), "device": props.name,
        "model_loaded": False, "initial_telemetry": telemetry()}

    def save() -> None:
        data = json.dumps(report, separators=(",", ":"), allow_nan=False) + "\n"
        if len(data.encode()) > p["maximum_report_bytes_per_rank"]:
            raise ValueError("Report exceeds its declared storage bound")
        temporary = args.report.with_suffix(".tmp")
        temporary.write_text(data)
        temporary.replace(args.report)

    def parent_bytes(item: dict[str, Any]) -> bytes:
        relative = Path(item["path"])
        if relative.is_absolute() or len(relative.parts) != 1:
            raise ValueError("Unexpected retained tensor path")
        path = Path(f"/audit/fp32-split-v1/rank{args.rank}") / relative
        data = path.read_bytes()
        if (len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]):
            raise ValueError("Retained precision tensor changed")
        return data

    save()
    try:
        pressure = torch.zeros(p["pressure_bytes"] // 4, dtype=torch.int32, device="cuda")
        stream = torch.cuda.Stream()
        deadline = time.monotonic() + p["initial_warmup_seconds"]
        while time.monotonic() < deadline:
            for _ in range(16):
                pressure.add_(1)
            torch.cuda.synchronize()
        report["after_warmup_telemetry"] = telemetry()
        for case in (c for c in plan["selected_cases"] if c["rank"] == args.rank):
            previous = next(c for c in parent["cases"] if c["name"] == case["name"])
            artifact = Path(case["artifact_path"])
            if digest(artifact) != case["artifact_sha256"]:
                raise ValueError("Packed source artifact changed")
            buffers = load_file(artifact)
            hashes = [hashlib.sha256(buffers[n].numpy().tobytes()).hexdigest()
                      for n in runtime.BUFFER_NAMES]
            packed = runtime.PackedWeight(tuple(buffers[n].cuda() for n in runtime.BUFFER_NAMES),
                                          *case["native_metadata"])
            del buffers
            decoded = torch.empty((packed.m, packed.k), device="cuda", dtype=torch.bfloat16)
            runtime._check(libraries["original"].mxzip_decode(
                torch.cuda.current_stream().cuda_stream,
                *(t.data_ptr() for t in packed.buffers), packed.max_common, packed.max_rare,
                packed.start_exp, decoded.data_ptr(), packed.m, packed.k))
            if tensor_digest(decoded.cpu()) != case["padded_weight_sha256"]:
                raise ValueError("Decoded source bytes changed")
            del decoded
            entry: dict[str, Any] = {"name": case["name"], "decoded_weight_bits_exact": True,
                "artifact_sha256": case["artifact_sha256"], "measurements": []}
            report["cases"].append(entry)
            for tokens in p["tokens"]:
                for split in p["splits"]:
                    prior = next(row for row in previous["checks"] if row["tokens"] == tokens
                                 and row["split"] == split and row["activation"] == p["activation"])
                    artifacts = prior["artifacts"]
                    values = {key: parent_bytes(artifacts[key]) for key in
                              ("activation", "original", "rebuilt", "candidate")}
                    x = torch.frombuffer(bytearray(values["activation"]), dtype=torch.uint8)
                    x = x.view(torch.bfloat16).reshape(artifacts["activation"]["shape"]).clone().cuda()
                    ops = {name: Operation(library, packed, x, case["logical_shape"][0],
                                           split if name == "candidate" else 1,
                                           fp32=name == "candidate")
                           for name, library in libraries.items()}
                    calls = {name: graph_call(op, x) for name, op in ops.items()}
                    before = {name: hashlib.sha256(raw(call())).hexdigest()
                              for name, call in calls.items()}
                    expected = {name: artifacts[name]["sha256"] for name in calls}
                    if before != expected:
                        raise ValueError("Eager timing outputs differ from prior precision words")
                    graphs, outputs = {}, {}
                    for name, call in calls.items():
                        for _ in range(3):
                            call()
                        stream.wait_stream(torch.cuda.current_stream())
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, stream=stream):
                            outputs[name] = call()
                        graph.replay()
                        graphs[name] = graph
                    torch.cuda.synchronize()
                    graph_before = {name: hashlib.sha256(raw(out)).hexdigest()
                                    for name, out in outputs.items()}
                    allocated = torch.cuda.memory_allocated()
                    for _ in range(8):
                        for graph in graphs.values():
                            graph.replay()
                    torch.cuda.synchronize()
                    retained = torch.cuda.memory_allocated() - allocated
                    row = {"tokens": tokens, "split": split, "activation": p["activation"],
                        "outputs_match_parent_before": before == expected,
                        "output_sha256_before": before, "graph_output_sha256": graph_before,
                        "graph_matches_eager": graph_before == expected,
                        "graph_retained_bytes": retained, "timing": []}
                    entry["measurements"].append(row)
                    for cache in p["cache_regimes"]:
                        result = measure({name: graph.replay for name, graph in graphs.items()},
                                         pressure, evict=cache == "pressure")
                        row["timing"].append({"mode": "graph", "cache": cache, **result})
                        save()
                    after = {name: hashlib.sha256(raw(out)).hexdigest()
                             for name, out in outputs.items()}
                    row.update(outputs_match_parent_after=after == expected,
                               output_sha256_after=after,
                               guards_unchanged=all(op.guards() for op in ops.values()))
                    if before != after or graph_before != expected or retained or not row["guards_unchanged"]:
                        raise ValueError("Timing changed precision, guards or graph allocations")
                    del ops, calls, graphs, outputs, graph, call, x, values, result
                    gc.collect()
                    torch.cuda.empty_cache()
                    save()
            entry["packed_buffers_unchanged"] = hashes == [
                hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest() for t in packed.buffers]
            del packed
            gc.collect()
            torch.cuda.empty_cache()
            save()
        report.update(status="complete", CUDA_peak_bytes=torch.cuda.max_memory_allocated(),
                      finished_epoch=time.time())
        report["assessment"] = assess(report)
        print(json.dumps(report["assessment"]))
    except BaseException as exc:
        report.update(status="failed", error=repr(exc), finished_epoch=time.time())
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
