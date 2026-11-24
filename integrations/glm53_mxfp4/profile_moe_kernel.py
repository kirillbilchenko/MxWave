"""Profile one warmed full-layer GEMM; no serving engine or quantization changes."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from audit_checkpoint import digest, write_json
from benchmark_moe import load_fixture
from moe_runtime import NativeMoe, allocate_buffers, pack_scales
from vllm import _custom_ops as ops


def main() -> None:
    """Expose only one selected grouped GEMM between CUDA profiler start/stop."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--libraries", type=Path, required=True)
    parser.add_argument("--candidate-library", type=Path, required=True)
    parser.add_argument("--codec-root", type=Path, required=True)
    parser.add_argument("--arm", choices=("raw", "candidate"), required=True)
    parser.add_argument("--projection", choices=("gate-up", "down"), required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    device = torch.device("cuda", torch.cuda.current_device())
    allocation_cap = int(2.25 * 1024**3)
    torch.cuda.set_per_process_memory_fraction(
        allocation_cap / torch.cuda.get_device_properties(device).total_memory, device
    )
    manifest_raw = (args.fixture / "manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    if manifest["expert_ids"] != list(range(288)):
        raise ValueError("Hardware profile requires the complete 288-expert fixture")
    # Concentrated routing touches only experts 0..7. Keep exactly those source
    # matrices resident during repeated profiler processes, with all 288 routing
    # entries and the original full-layer launch plan. The device GEMM receives
    # no expert-count argument; inactive expert payloads are never dereferenced.
    active_manifest = dict(manifest)
    active_manifest["expert_ids"] = list(range(8))
    active_manifest["tensors"] = [t for t in manifest["tensors"] if t["expert"] < 8]
    w13, w2, s13, s2 = load_fixture(args.fixture, active_manifest, 2, 0, device)
    weight, scales = (w13, s13) if args.projection == "gate-up" else (w2, s2)
    del w13, w2, s13, s2
    gc.collect()
    torch.cuda.empty_cache()
    raw_library = args.libraries / "libmxwave_moe_raw.so"
    raw_native = NativeMoe(raw_library)
    path = raw_library if args.arm == "raw" else args.candidate_library
    native = raw_native if args.arm == "raw" else NativeMoe(path)
    if native.compressed != (args.arm == "candidate"):
        raise ValueError("Hardware profile arm and scale representation differ")
    codec = args.codec_root / "integrations/marlin_scales/records.py"
    storage = (pack_scales(scales, codec, encoding=native.encoding)
               if native.compressed else scales)
    tokens, topk, experts, block = 128, 8, 288, 8
    ids = torch.arange(topk, dtype=torch.int32, device=device).repeat(tokens, 1)
    generator = torch.Generator().manual_seed(73 + tokens)
    routing_weights = torch.softmax(torch.randn(tokens, topk, generator=generator), -1).to(device)
    capacity = min(tokens * topk + experts * (block - 1), tokens * topk * block)
    routing = (torch.empty(capacity, dtype=torch.int32, device=device),
               torch.empty((capacity + block - 1) // block, dtype=torch.int32, device=device),
               torch.empty(1, dtype=torch.int32, device=device))
    ops.moe_align_block_size(ids, experts, block, *routing, None, None)
    _, groups, n = scales.shape
    k = groups * 32
    gemm_m = tokens if args.projection == "gate-up" else tokens * topk
    gemm_topk = topk if args.projection == "gate-up" else 1
    x = torch.randn((gemm_m, k), dtype=torch.bfloat16, device=device)
    buffers = allocate_buffers(gemm_m, n, gemm_topk, block, capacity, device)
    plan = raw_native.plan(gemm_m, n, k, block, gemm_topk, experts, device.index)
    operation = native.bind(x, weight, storage, buffers, routing, routing_weights,
                            gemm_topk, block, args.projection == "down", plan)
    for _ in range(3):
        operation()
    torch.cuda.synchronize()
    if torch.count_nonzero(buffers.locks).item() or not torch.isfinite(buffers.output).all():
        raise RuntimeError("Warm grouped kernel left locks or non-finite output")
    reference_buffers = allocate_buffers(gemm_m, n, gemm_topk, block, capacity, device)
    reference = raw_native.bind(x, weight, scales, reference_buffers, routing, routing_weights,
                                gemm_topk, block, args.projection == "down", plan)
    reference()
    if not torch.equal(buffers.output.view(torch.uint8), reference_buffers.output.view(torch.uint8)):
        raise RuntimeError("Profiled kernel differs from the identical raw reference")
    torch.cuda.profiler.start()
    operation()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    write_json(args.report, {
        "status": "complete", "purpose": "hardware counters for one warmed grouped GEMM",
        "arm": args.arm, "projection": args.projection, "tokens": tokens,
        "concentrated_routing": True, "moe_block": block, "tp": 2, "rank": 0,
        "resident_experts": 8, "routing_experts": experts, "raw_reference_bits_exact": True,
        "gpu": torch.cuda.get_device_name(device), "plan": plan,
        "script_sha256": digest(Path(__file__).read_bytes()),
        "runtime_sha256": digest(Path(__file__).with_name("moe_runtime.py").read_bytes()),
        "fixture_manifest_sha256": digest(manifest_raw),
        "library_sha256": digest(path.read_bytes()), "codec_sha256": digest(codec.read_bytes()),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "limitations": ["Only the eight active experts are resident; full 288-expert routing metadata and launch plan.",
                        "One synthetic concentrated-routing gate-up/down GEMM, not a model forward.",
                        "Nsight replay counters are diagnostics, not paired latency measurements.",
                        "No new whole-model quality or throughput measurement."],
    })


if __name__ == "__main__":
    main()
