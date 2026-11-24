"""Check the M4 KDA output under explicit cache pressure in a bounded component."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from benchmark_bf16_kernels import arithmetic, digest, fixtures
from diagnose_bf16_attention import write
from diagnose_bf16_overlap import capture
from prepare_bf16_attention import LIBRARY_SHA, RUNTIME_SHA, rebuild_weight, tensor_digest
from safetensors.torch import load_file


def timed_call(operation: Callable[[], Any], evict: Callable[[], Any]) -> float:
    """Queue eviction before the start event so its bandwidth cost is excluded."""
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    evict()
    start.record()
    operation()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) * 1000


def main() -> None:
    """Require unchanged source/decoded bytes and compare both arm/cache treatment orders."""
    from mxwave_zipserv import runtime
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--glm-paused", action="store_true")
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("Preserve prior cache-pressure evidence")
    if (digest(Path(os.environ["MXWAVE_ZIPSERV_LIBRARY"])) != LIBRARY_SHA
            or digest(Path(runtime.__file__)) != RUNTIME_SHA):
        raise ValueError("Require the same qualified native library and runtime")
    from kda_policy import CONFIG_SHA, INDEX_SHA

    for name, expected in (("config.json", CONFIG_SHA), ("model.safetensors.index.json", INDEX_SHA)):
        if digest(args.source / name) != expected:
            raise ValueError("Source metadata changed")
    case = next(c for c in fixtures(args.source.parent) if c["name"] == "kda-output-l0-r0")
    artifact = case["directory"] / case["artifact"]
    if digest(artifact) != case["artifact_sha256"]:
        raise ValueError("Packed output artifact changed")
    torch.set_num_threads(2)
    torch.manual_seed(787)
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    if (properties.major, properties.minor) != (12, 1):
        raise ValueError("Require SM121")
    torch.cuda.set_per_process_memory_fraction(512 * 1024**2 / properties.total_memory)
    index = json.loads((args.source / "model.safetensors.index.json").read_bytes())["weight_map"]
    cpu = rebuild_weight(args.source, index, case)
    if tensor_digest(cpu) != case["logical_weight_sha256"]:
        raise ValueError("Reconstructed source bytes changed")
    dense = cpu.cuda()
    del cpu
    buffers = load_file(artifact)
    packed = runtime.PackedWeight(tuple(buffers[n].cuda() for n in runtime.BUFFER_NAMES),
                                  *case["native_metadata"])
    del buffers
    decoded = runtime.decompress(packed)
    if tensor_digest(decoded.cpu()) != case["padded_weight_sha256"]:
        raise RuntimeError("Decoded GPU weight bytes differ")
    del decoded
    x = torch.randn((4, dense.shape[1]), device="cuda", dtype=torch.bfloat16) * .1
    method = UnquantizedLinearMethod()
    stock = lambda: method.apply(SimpleNamespace(weight=dense), x)
    fused = lambda: runtime.linear(x, packed, split_k=1)
    stream = torch.cuda.Stream()
    dg, expected = capture(stock, stream)
    fg, actual = capture(fused, stream)
    check = arithmetic(actual, expected, exact=True)
    if not check["passed"]:
        raise RuntimeError("Current M4 KDA output arithmetic changed")
    eviction = torch.zeros(128 * 1024**2, device="cuda", dtype=torch.uint8)
    report: dict[str, Any] = {"status": "running", "case": case["name"], "tokens": 4,
        "library_sha256": LIBRARY_SHA, "runtime_sha256": RUNTIME_SHA,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_weight_sha256": case["logical_weight_sha256"],
        "artifact_sha256": case["artifact_sha256"], "gpu_weight_bits_exact": True,
        "arithmetic": check, "eviction_bytes": eviction.numel(),
        "l2_cache_bytes": properties.L2_cache_size, "records": [], "summaries": [],
        "production_paused": args.glm_paused,
        "limits": ["One M4 layer-zero/rank-zero component; no full-model benchmark loaded.",
            "128 MiB read/write buffer is an eviction workload, not a hardware cache flush API.",
            "CUDA graph replay removes most host launch differences; not model throughput.",
            "Per-call CUDA-event intervals exclude eviction work and post-call host synchronization.",
            ("Production GLM temporarily paused under the independent restoration watchdog."
             if args.glm_paused else
             "Production GLM remains resident; guarded idle counters must stay unchanged.")]}
    write(args.report, report)
    modes = ("warm", "evicted", "evicted", "warm", "evicted", "warm", "warm", "evicted")
    for block, first_mode in enumerate(modes):
        for mode in ((first_mode, "evicted" if first_mode == "warm" else "warm")):
            record: dict[str, Any] = {"block": block, "mode": mode,
                "order": ["dense", "fused"] if block % 2 == 0 else ["fused", "dense"]}
            for arm in record["order"]:
                graph = dg if arm == "dense" else fg
                for _ in range(8):
                    graph.replay()
                torch.cuda.synchronize()
                evict = (lambda: eviction.add_(1)) if mode == "evicted" else (lambda: None)
                samples = [timed_call(graph.replay, evict) for _ in range(12)]
                record[arm + "_samples_us"] = samples
                record[arm + "_median_us"] = statistics.median(samples)
            record["ratio"] = record["fused_median_us"] / record["dense_median_us"]
            report["records"].append(record)
            write(args.report, report)
    for mode in ("warm", "evicted"):
        records = [r for r in report["records"] if r["mode"] == mode]
        report["summaries"].append({"mode": mode,
            "dense_median_us": statistics.median(r["dense_median_us"] for r in records),
            "fused_median_us": statistics.median(r["fused_median_us"] for r in records),
            "median_paired_ratio": statistics.median(r["ratio"] for r in records),
            "order_ratios": {first: statistics.median(r["ratio"] for r in records
                if r["order"][0] == first) for first in ("dense", "fused")}})
    report["after_arithmetic"] = arithmetic(fused(), stock(), exact=True)
    if not report["after_arithmetic"]["passed"]:
        raise RuntimeError("Cache-pressure run changed output bytes")
    report.update(status="passed", cuda_peak_bytes=torch.cuda.max_memory_allocated())
    write(args.report, report)
    print(json.dumps({"status": report["status"], "summaries": report["summaries"]}))


if __name__ == "__main__":
    main()
