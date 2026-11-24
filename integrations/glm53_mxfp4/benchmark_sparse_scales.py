"""Gate the reusable sparse palette codec against the current aligned-nibble MoE runtime."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
import time
from collections.abc import Callable
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

import torch
from benchmark_moe import check_case, load_fixture, same
from diagnose_bf16_cache import timed_call
from moe_runtime import NativeMoe, pack_scales


def digest(path: Path) -> str:
    """Hash immutable inputs without retaining checkpoint payloads in the report."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: Any) -> None:
    """Save progress atomically."""
    temporary = path.with_suffix(".incomplete")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def pairs(control: Callable[[], Any], candidate: Callable[[], Any], *,
          eviction: torch.Tensor,
          progress: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """Balance both arm/cache orders and exclude eviction work from per-call CUDA events."""
    records = []
    before = torch.cuda.memory_allocated()
    for block, first in enumerate(("warm", "evicted", "evicted", "warm",
                                   "evicted", "warm", "warm", "evicted")):
        for mode in (first, "warm" if first == "evicted" else "evicted"):
            order = ("control", "candidate") if block % 2 == 0 else ("candidate", "control")
            record: dict[str, Any] = {"block": block, "mode": mode, "order": order}
            for arm in order:
                if progress is not None:
                    progress({"block": block, "mode": mode, "arm": arm, "stage": "warmup"})
                operation = control if arm == "control" else candidate
                for _ in range(4):
                    operation()
                torch.cuda.synchronize()
                evict = (lambda: eviction.add_(1)) if mode == "evicted" else (lambda: None)
                samples = []
                for sample in range(12):
                    if progress is not None:
                        progress({"block": block, "mode": mode, "arm": arm,
                                  "stage": "sample", "sample": sample})
                    samples.append(timed_call(operation, evict))
                record[arm + "_samples_us"] = samples
                record[arm + "_median_us"] = statistics.median(samples)
            record["ratio"] = record["candidate_median_us"] / record["control_median_us"]
            records.append(record)
    growth = torch.cuda.memory_allocated() - before
    if growth:
        raise RuntimeError("Sparse paired replay grew CUDA allocations")
    return {"records": records, "allocation_growth_bytes": growth,
        "summaries": [{"mode": mode,
            "median_paired_ratio": statistics.median(r["ratio"] for r in records if r["mode"] == mode),
            "order_ratios": {first: statistics.median(r["ratio"] for r in records
                if r["mode"] == mode and r["order"][0] == first) for first in ("control", "candidate")}}
                       for mode in ("warm", "evicted")]}


def decoder_checks(native: NativeMoe, device: torch.device) -> dict[str, Any]:
    """Cover every byte, all selector words, arbitrary palettes, rare vectors and stream/graphs."""
    codec = Path("unused-by-sparse")
    cases = []
    patterns = torch.arange(256, dtype=torch.uint8, device=device).repeat(256)
    words = torch.arange(65536, dtype=torch.int32, device=device)
    symbols = ((words[:, None] >> (torch.arange(8, device=device) * 2)) & 3).to(torch.uint8)
    inputs = [patterns, symbols.repeat(1, 2).reshape(-1),
              (symbols.repeat(1, 2) + 252).reshape(-1),
              torch.tensor([0, 127, 255, 0], dtype=torch.uint8, device=device).repeat(16384),
              torch.full((65536,), 120, dtype=torch.uint8, device=device)]
    mixed = torch.full((1024**2 + 32,), 120, dtype=torch.uint8, device=device)
    mixed[16:32], mixed[32:48] = 119, 121
    mixed[torch.tensor([0, 1024**2 - 1, 1024**2, 1024**2 + 31], device=device)] = torch.tensor(
        [0, 255, 7, 254], dtype=torch.uint8, device=device)
    inputs.append(mixed)
    if native.nibble_stage:
        # All sixteen aligned buckets, including rare values outside the common
        # palette, exercise exact exception-to-nibble staging without clipping.
        inputs.extend(torch.arange(16, dtype=torch.uint8, device=device).repeat(4096) + base
                      for base in range(0, 256, 16))
        inputs.append(torch.tensor([113, 119, 126, 127], dtype=torch.uint8,
                                   device=device).repeat(16384))
        crossing = torch.tensor([119, 120, 121, 120], dtype=torch.uint8,
                                device=device).repeat(16384)
        crossing[-1] = 128
        inputs.append(crossing)
    for i, flat in enumerate(inputs):
        raw = flat.reshape(1, -1, 16)
        packed = pack_scales(raw.view(torch.float8_e8m0fnu), codec, encoding=native.encoding)
        if native.nibble_stage:
            lower, upper = int(raw.min()), int(raw.max())
            expected_base = (lower & ~15) if lower // 16 == upper // 16 else None
            if not packed.stage_metadata_verified or packed.aligned_nibble_base != expected_base:
                raise RuntimeError("Staging policy omitted source bytes or exception ranges")
            for decode in (native.decode, native.decode_bf16):
                try:
                    decode(replace(packed, stage_metadata_verified=False))
                except ValueError:
                    pass
                else:
                    raise RuntimeError("Nibble staging accepted an unverified descriptor")
        expected = (raw.reshape(-1, 4)[:, [0, 2, 1, 3]].int() << 7).to(torch.int16).contiguous()
        expected = expected.view(torch.uint8).reshape(1, -1, 32)
        if not same(native.decode(packed), raw) or not same(native.decode_bf16(packed), expected):
            raise RuntimeError("Sparse GPU byte/fragment decoder differs from stock bits")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            streamed = native.decode_bf16(packed)
        torch.cuda.current_stream().wait_stream(stream)
        if not same(streamed, expected):
            raise RuntimeError("Sparse decoder changed bits on a nondefault stream")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            replayed = native.decode_bf16(packed)
        replayed.fill_(0xff)
        before = torch.cuda.memory_allocated()
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        if torch.cuda.memory_allocated() != before or not same(replayed, expected):
            raise RuntimeError("Sparse decoder replay changed bytes or grew allocations")
        cases.append({"case": i, "bytes": raw.numel(), "bits_exact": True,
                      "aligned_nibble_base": packed.aligned_nibble_base,
                      "stage_metadata_verified": packed.stage_metadata_verified,
                      "exception_vectors": packed.exceptions.numel() // 16,
                      "nondefault_stream": True, "graph_replay_growth_bytes": 0})
        del graph, replayed, streamed, packed
    return {"status": "passed", "cases": cases}


def main() -> None:
    """Use complete real layer-23 weights with unchanged sorted-routing and stock arithmetic gates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--confirmation", action="store_true")
    parser.add_argument("--diagnostic", action="store_true",
                        help="One M1 case with launch-blocking diagnostics; no performance claim")
    args = parser.parse_args()
    if args.diagnostic and args.confirmation:
        parser.error("Diagnostic launches are separate from performance confirmation")
    if args.report.exists():
        raise FileExistsError("Preserve prior sparse GPU results")
    manifest = json.loads((args.build / "manifest.json").read_bytes())
    source = json.loads((args.fixture / "manifest.json").read_bytes())
    prior = json.loads(args.prior.read_bytes())
    if source["expert_ids"] != list(range(288)) or prior["status"] != "passed":
        raise ValueError("Complete source layer and CPU scale gate required")
    paths = [args.build / "build" / f"libmxwave_moe_{name}.so" for name in ("aligned", "sparse")]
    if any(digest(p) != manifest["libraries_sha256"][p.name] for p in paths):
        raise ValueError("Native libraries differ from their pinned build")
    control, candidate = (NativeMoe(p) for p in paths)
    original_path = Path("/audit/native/libmxwave_moe_aligned.so")
    if digest(original_path) != "53ebb8478607104723786934c96ae7dc973a9d16ff93a2ec8c93a2a653dae658":
        raise ValueError("The original compressed-codec reference changed")
    original = NativeMoe(original_path)
    if control.encoding != "aligned_nibbles" or candidate.encoding != "sparse_palette":
        raise ValueError("Unexpected control/candidate representations")
    if candidate.affine_only != (manifest.get("sparse_palette_policy") == "affine_only"):
        raise ValueError("Native palette policy differs from the pinned build manifest")
    if candidate.shared_stage != manifest.get("sparse_shared_stage", False):
        raise ValueError("Native shared-stage policy differs from the pinned build manifest")
    if candidate.nibble_stage != manifest.get("sparse_nibble_stage", False):
        raise ValueError("Native nibble-stage policy differs from the pinned build manifest")
    if candidate.static_nibble_stage != manifest.get("sparse_static_nibble_stage", False):
        raise ValueError("Native static-stage policy differs from the pinned build manifest")
    fallback = None
    if candidate.nibble_stage:
        fallback_path = args.build / "build/libmxwave_moe_fallback.so"
        if digest(fallback_path) != manifest["libraries_sha256"][fallback_path.name]:
            raise ValueError("Exact U8 fallback reference changed")
        fallback = NativeMoe(fallback_path)
        if fallback.encoding != candidate.encoding or not fallback.shared_stage or fallback.nibble_stage:
            raise ValueError("Require the previous U8 shared-stage consumer as fallback reference")
        if candidate.static_nibble_stage:
            candidate.set_u8_fallback(fallback)
    torch.set_num_threads(2)
    torch.manual_seed(394 if args.confirmation else 391)
    device = torch.device("cuda", 0)
    props = torch.cuda.get_device_properties(device)
    if (props.major, props.minor) != (12, 1):
        raise ValueError("Require the qualified SM121 target")
    torch.cuda.set_per_process_memory_fraction(4.5 * 1024**3 / props.total_memory)
    report: dict[str, Any] = {"status": "running", "confirmation": args.confirmation,
        "diagnostic": args.diagnostic,
        "candidate_affine_only": candidate.affine_only,
        "candidate_shared_stage": candidate.shared_stage,
        "candidate_nibble_stage": candidate.nibble_stage,
        "candidate_static_nibble_stage": candidate.static_nibble_stage,
        "code_sha256": digest(Path(__file__)), "runtime_sha256": digest(Path(__file__).with_name("moe_runtime.py")),
        "codec_sha256": digest(Path(__file__).with_name("scale_tiles.py")),
        "build_manifest_sha256": digest(args.build / "manifest.json"),
        "fixture_manifest_sha256": digest(args.fixture / "manifest.json"),
        "prior_sha256": digest(args.prior), "partitions": [], "fallback_cases": [],
        "original_control_sha256": digest(original_path),
        "limits": ["Complete 288-expert layer; synthetic routing, not a full-model workload.",
            "Two TP shards execute sequentially on one GPU, without collectives.",
            "Control is the current compressed codec, not raw scales.",
            "Installed-stock numerical differences remain reported; stock equivalence is not assumed.",
            "Comparison peak deliberately retains raw, control and candidate buffers."]}
    started = time.monotonic()
    write(args.report, report)

    def diagnostic_progress(value: dict[str, Any]) -> None:
        report["last_launch"] = value
        write(args.report, report)
        print(json.dumps(value), flush=True)

    try:
        report["decoder"] = decoder_checks(candidate, device)
        eviction = torch.zeros(128 * 1024**2, dtype=torch.uint8, device=device)
        for rank in ((0,) if args.diagnostic else (1, 0) if args.confirmation else (0, 1)):
            fixture = load_fixture(args.fixture, source, 2, rank, device, current_api=True)
            controls, candidates, storage = [], [], []
            for family, raw in zip(("w13", "w2"), fixture[2:], strict=True):
                cpu = next(c for c in prior["cases"] if c["rank"] == rank and c["family"] == family)
                if hashlib.sha256(raw.view(torch.uint8).cpu().numpy()).hexdigest() != cpu["raw_sha256"]:
                    raise RuntimeError("Stock GPU Marlin permutation differs from the CPU fixture")
                before = torch.cuda.memory_allocated()
                a = pack_scales(raw, Path("unused"), encoding=control.encoding)
                control_allocation = torch.cuda.memory_allocated() - before
                before = torch.cuda.memory_allocated()
                b = pack_scales(raw, Path("unused"), encoding=candidate.encoding)
                candidate_allocation = torch.cuda.memory_allocated() - before
                if (b.retained_bytes != cpu["retained_bytes"]
                        or not same(candidate.decode(b), raw.view(torch.uint8))):
                    raise RuntimeError("Candidate storage/decode differs from CPU source proof")
                controls.append(a)
                candidates.append(b)
                storage.append({"family": family, "raw_bytes": raw.numel(),
                    "control_retained_bytes": a.retained_bytes, "candidate_retained_bytes": b.retained_bytes,
                    "control_cuda_allocation_bytes": control_allocation,
                    "candidate_cuda_allocation_bytes": candidate_allocation,
                    "candidate_aligned_nibble_base": b.aligned_nibble_base,
                    "source_bytes_exact": True})
            entry: dict[str, Any] = {"rank": rank, "storage": storage, "cases": []}
            report["partitions"].append(entry)
            for m in ((1,) if args.diagnostic else (1, 4, 8, 32, 128, 512, 2048)):
                for skewed in ((False,) if args.diagnostic else (False, True)):
                    entry["cases"].append(check_case(fixture, tuple(candidates), control, candidate,
                        m, skewed, None, control_scales=tuple(controls),
                        timing_function=partial(pairs, eviction=eviction,
                            progress=diagnostic_progress if args.diagnostic else None), current_api=True,
                        reference_native=original))
                    write(args.report, report)
            if fallback is not None and not args.diagnostic:
                # Preserve the real weight codes but force the complete scale
                # ranges across a bucket with one rare exponent in expert zero.
                # These are synthetic safety cases, not new checkpoint evidence.
                crossed = [v.clone() for v in fixture[2:]]
                for value in crossed:
                    value.view(torch.uint8).reshape(-1)[0] = 128
                packed_crossed = tuple(pack_scales(value, Path("unused"),
                    encoding=candidate.encoding) for value in crossed)
                if any(p.aligned_nibble_base is not None for p in packed_crossed):
                    raise RuntimeError("Rare out-of-bucket source did not select exact U8 fallback")
                if candidate.static_nibble_stage and any(
                        candidate.scale_consumer(p) is not fallback for p in packed_crossed):
                    raise RuntimeError("Static dispatch selected the nibble kernel for an invalid range")
                for m in (1, 128):
                    case = check_case((*fixture[:2], *crossed), packed_crossed,
                        fallback, candidate, m, True, None, timings=False,
                        control_scales=packed_crossed, current_api=True, reference_native=fallback)
                    case.update(rank=rank, synthetic_scale_edit=True,
                        arithmetic_reference="Previous exact U8 shared-stage consumer",
                        complete_range_forces_u8_fallback=True,
                        fallback_selected_before_launch=candidate.static_nibble_stage)
                    report["fallback_cases"].append(case)
                    write(args.report, report)
                del crossed, packed_crossed
            del fixture, controls, candidates, a, b, raw
            gc.collect()
            torch.cuda.empty_cache()
        report.update(status="passed", seconds=time.monotonic() - started,
                      cuda_peak_bytes=torch.cuda.max_memory_allocated())
        write(args.report, report)
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        write(args.report, report)
        raise


if __name__ == "__main__":
    main()
