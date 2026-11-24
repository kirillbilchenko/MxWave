"""Collect NCU counters for exact current-codec/palette GEMMs without a model load."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from benchmark_moe import load_fixture, same
from benchmark_sparse_scales import digest, write
from moe_runtime import NativeMoe, allocate_buffers, pack_scales
from vllm import _custom_ops as ops


def main() -> None:
    """Profile only four selected GEMMs after checking their source bytes and arithmetic."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--confirmation", action="store_true",
                        help="Profile the candidate; the first process profiles the control")
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("Preserve previous counter evidence")
    manifest = json.loads((args.build / "manifest.json").read_bytes())
    source = json.loads((args.fixture / "manifest.json").read_bytes())
    prior = json.loads(args.prior.read_bytes())
    if source["expert_ids"] != list(range(288)) or prior["status"] != "passed":
        raise ValueError("Require the complete source layer and CPU proof")
    paths = [args.build / "build" / f"libmxwave_moe_{name}.so" for name in ("aligned", "sparse")]
    if any(digest(p) != manifest["libraries_sha256"][p.name] for p in paths):
        raise ValueError("Native counter inputs changed")
    control, candidate = (NativeMoe(p) for p in paths)
    if control.encoding != "aligned_nibbles" or candidate.encoding != "sparse_palette":
        raise ValueError("Require the current codec and its palette candidate")
    if candidate.shared_stage != manifest.get("sparse_shared_stage", False):
        raise ValueError("Native shared-stage policy differs from the pinned build manifest")
    if candidate.nibble_stage != manifest.get("sparse_nibble_stage", False):
        raise ValueError("Native nibble-stage policy differs from the pinned build manifest")
    if candidate.static_nibble_stage != manifest.get("sparse_static_nibble_stage", False):
        raise ValueError("Native static-stage policy differs from the pinned build manifest")
    torch.set_num_threads(2)
    torch.manual_seed(701)
    device = torch.device("cuda", 0)
    props = torch.cuda.get_device_properties(device)
    if (props.major, props.minor) != (12, 1):
        raise ValueError("Require SM121")
    torch.cuda.set_per_process_memory_fraction(3.5 * 1024**3 / props.total_memory)
    w13, w2, s13, s2 = load_fixture(args.fixture, source, 2, 0, device, current_api=True)
    controls, candidates = [], []
    for family, raw in (("w13", s13), ("w2", s2)):
        cpu = next(c for c in prior["cases"] if c["rank"] == 0 and c["family"] == family)
        if hashlib.sha256(raw.view(torch.uint8).cpu().numpy()).hexdigest() != cpu["raw_sha256"]:
            raise ValueError("GPU source scale permutation changed")
        a = pack_scales(raw, Path("unused"), encoding=control.encoding)
        b = pack_scales(raw, Path("unused"), encoding=candidate.encoding)
        if not same(candidate.decode(b), raw.view(torch.uint8)):
            raise RuntimeError("Profile input changed decoded scale bytes")
        controls.append(a)
        candidates.append(b)
    arm = "candidate" if args.confirmation else "control"
    report: dict[str, Any] = {"status": "running", "purpose": "hardware counters only",
        "arm": arm, "code_sha256": digest(Path(__file__)),
        "candidate_shared_stage": candidate.shared_stage,
        "candidate_nibble_stage": candidate.nibble_stage,
        "candidate_static_nibble_stage": candidate.static_nibble_stage,
        "sm_count": props.multi_processor_count, "l2_cache_bytes": props.L2_cache_size,
        "build_manifest_sha256": digest(args.build / "manifest.json"),
        "fixture_manifest_sha256": digest(args.fixture / "manifest.json"),
        "runtime_sha256": digest(Path(__file__).with_name("moe_runtime.py")),
        "codec_sha256": digest(Path(__file__).with_name("scale_tiles.py")),
        "prior_sha256": digest(args.prior), "cases": [],
        "limits": ["Full 288-expert layer, rank zero, concentrated top-8 routing.",
                   "NCU replay/cache treatments are not paired unprofiled latency.",
                   "No full model, TP collectives or quality qualification."]}
    write(args.report, report)
    try:
        for m in (1, 128):
            topk, experts, block = 8, 288, 8
            ids = torch.arange(topk, dtype=torch.int32, device=device).repeat(m, 1)
            scores = torch.softmax(torch.randn(m, topk, device=device), -1)
            capacity = min(m * topk + experts * (block - 1), m * topk * block)
            routing = (torch.empty(capacity, dtype=torch.int32, device=device),
                torch.empty((capacity + block - 1) // block, dtype=torch.int32, device=device),
                torch.empty(1, dtype=torch.int32, device=device))
            ops.moe_align_block_size(ids, experts, block, *routing, None)
            for family, weight, raw, a, b in zip(("w13", "w2"), (w13, w2), (s13, s2),
                                                controls, candidates, strict=True):
                gemm_m, gemm_topk = (m, topk) if family == "w13" else (m * topk, 1)
                _, groups, n = raw.shape
                k = groups * 32
                x = torch.randn((gemm_m, k), dtype=torch.bfloat16, device=device)
                plan = control.plan(gemm_m, n, k, block, gemm_topk, experts, 0)
                buffers = [allocate_buffers(gemm_m, n, gemm_topk, block, capacity, device)
                           for _ in range(2)]
                operations = [native.bind(x, weight, scales, output, routing, scores,
                    gemm_topk, block, family == "w2", plan)
                    for native, scales, output in zip((control, candidate), (a, b), buffers, strict=True)]
                for operation in operations:
                    for _ in range(3):
                        operation()
                torch.cuda.synchronize()
                if (not same(buffers[0].output, buffers[1].output)
                        or not bool(torch.isfinite(buffers[0].output).all())
                        or any(bool(torch.count_nonzero(v.locks)) for v in buffers)):
                    raise RuntimeError("Profiled GEMM changed current-codec bits or managed locks")
                torch.cuda.profiler.start()
                torch.cuda.nvtx.range_push(f"palette-{arm}-m{m}-{family}")
                operations[int(args.confirmation)]()
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
                torch.cuda.profiler.stop()
                if (not same(buffers[0].output, buffers[1].output)
                        or any(bool(torch.count_nonzero(v.locks)) for v in buffers)):
                    raise RuntimeError("Profiled replay changed current-codec output or locks")
                report["cases"].append({"tokens": m, "family": family, "plan": plan,
                    "shape_m_n_k": [gemm_m, n, k], "current_codec_bits_exact": True,
                    "source_scales_exact": True, "managed_locks_reset": True})
                write(args.report, report)
        report.update(status="passed", cuda_peak_bytes=torch.cuda.max_memory_allocated())
        write(args.report, report)
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        write(args.report, report)
        raise


if __name__ == "__main__":
    main()
