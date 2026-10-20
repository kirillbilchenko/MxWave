"""Review exact current-codec equivalence, incremental memory and repeated component latency."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

SIZES = (1, 4, 8, 32, 128, 512, 2048)


def cpu_audit(report: dict[str, Any]) -> dict[str, Any]:
    """Require exact complete expert-scale coverage without claiming full-model equivalence."""
    expected = {(layer, rank, family) for layer in range(3, 45)
                for rank in (0, 1) for family in ("w13", "w2")}
    cases = report["cases"]
    if (report.get("status") != "passed" or not report.get("palette_only")
            or not report.get("all_main_expert_layers") or len(cases) != 168
            or {(c["layer"], c["rank"], c["family"]) for c in cases} != expected
            or report.get("source_scale_tensors") != 36288):
        raise ValueError("Require all 42 layers, 288 experts, three source scales and both TP shards")
    raw_bytes = baseline_bytes = candidate_bytes = 0
    fallbacks = 0
    for case in cases:
        formats = case["formats"]
        if (len(formats) != 1 or formats[0]["mode"] != "palette16"
                or not formats[0]["cpu_roundtrip_exact"]):
            raise ValueError("Missing exact direct-palette reconstruction")
        value = formats[0]
        if (value["selectors_bytes"] != case["raw_bytes"] // 4
                or value["offsets_bytes"] != 0 or value["exceptions_bytes"] % 16
                or value["retained_bytes"] != value["selectors_bytes"] + value["exceptions_bytes"] + 24):
            raise ValueError("Palette memory accounting changed")
        raw_bytes += case["raw_bytes"]
        baseline_bytes += case["baseline_retained_bytes"]
        candidate_bytes += value["retained_bytes"]
        fallbacks += case["baseline_encoding"] == "raw_e8m0"
    if (raw_bytes, baseline_bytes) != (9512681472, 4794092128):
        raise ValueError("Whole-model expert scale baseline differs from the qualified checkpoint")
    return {"source_scale_tensors": 36288, "layers": 42, "tp_scale_groups": 168,
        "cpu_source_roundtrip_exact": True, "source_scale_bytes": raw_bytes,
        "current_codec_payload_bytes": baseline_bytes, "current_raw_fallback_groups": fallbacks,
        "candidate_payload_bytes": candidate_bytes,
        "additional_payload_saving_bytes": baseline_bytes - candidate_bytes,
        "additional_payload_saving_fraction": 1 - candidate_bytes / baseline_bytes,
        "source_scale_bytes_sha256": report["source_scale_bytes_sha256"],
        "whole_model_cuda_allocation_measured": False,
        "full_model_arithmetic_qualified": False,
        "limits": ["Exact CPU scale bytes; GPU full-model reconstruction is a separate gate.",
                   "Existing quantization error is retained; the reference is the MXFP4 checkpoint.",
                   "Other tensors, model outputs and real-work accuracy are not audited here."]}


def ncu_review(directory: Path) -> dict[str, Any]:
    """Compare source-bound NCU counters without treating replay as unprofiled latency."""
    metrics = ("gpu__time_duration.sum", "smsp__inst_executed.sum",
        "launch__registers_per_thread", "launch__registers_per_thread_allocated",
        "launch__shared_mem_per_block", "launch__occupancy_limit_registers",
        "launch__occupancy_limit_shared_mem", "sm__warps_active.avg.pct_of_peak_sustained_active",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__issue_active.avg.pct_of_peak_sustained_elapsed",
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
        "lts__throughput.avg.pct_of_peak_sustained_elapsed",
        "smsp__warps_eligible.avg.per_cycle_active", "profiler__replayer_passes",
        "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
        "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio")
    arms: dict[str, Any] = {}
    bindings: dict[str, str] = {}
    files = {}
    for stem, arm in (("sparse", "control"), ("sparse-confirm", "candidate")):
        paths = [directory / (stem + suffix) for suffix in (".json", "-ncu.csv")]
        report = json.loads(paths[0].read_bytes())
        if (report.get("status") != "passed" or report.get("purpose") != "hardware counters only"
                or report.get("arm") != arm):
            raise ValueError("Require both qualified hardware-counter processes")
        for key in ("code_sha256", "runtime_sha256", "codec_sha256", "build_manifest_sha256",
                    "fixture_manifest_sha256", "prior_sha256"):
            if not report.get(key) or key in bindings and bindings[key] != report[key]:
                raise ValueError("Counter input bindings differ")
            bindings[key] = report[key]
        table = list(csv.reader(paths[1].read_text().splitlines()))
        header = next(row for row in table if row and row[0] == "ID")
        start = table.index(header)
        units = dict(zip(header, table[start + 1], strict=True))
        rows = [dict(zip(header, row, strict=True)) for row in table[start + 2:]
                if len(row) == len(header) and row[0].isdigit()]
        cases = report["cases"]
        if (len(rows) != 4 or len(cases) != 4 or [(c["tokens"], c["family"]) for c in cases]
                != [(1, "w13"), (1, "w2"), (128, "w13"), (128, "w2")]):
            raise ValueError("Missing concentrated-routing counter cases")
        records = []
        for case, row in zip(cases, rows, strict=True):
            if not all(case[k] for k in ("current_codec_bits_exact", "source_scales_exact",
                                        "managed_locks_reset")):
                raise ValueError("Profiled GEMM failed source/arithmetic gates")
            values = {key: float(row[key].replace(",", "")) for key in metrics}
            if any(not math.isfinite(value) or value < 0 for value in values.values()):
                raise ValueError("Invalid hardware counter")
            records.append({**case, "grid": row["Grid Size"], "block": row["Block Size"],
                "metrics": {k: {"value": v, "unit": units[k]} for k, v in values.items()}})
        arms[arm] = records
        for path in paths:
            files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"arms": arms, "bindings": bindings, "input_sha256": files,
        "limits": ["Eight NCU counter profiles, not new model throughput or a paired timing test.",
                   "The installed export does not measure LPDDR bytes directly.",
                   "Similar occupancy and more instructions identify a lead, not complete causal proof."]}


def latency(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recompute balanced paired ratios from samples, retaining both cache and arm orders."""
    if (len(records) != 16 or {(r["block"], r["mode"]) for r in records}
            != {(b, mode) for b in range(8) for mode in ("warm", "evicted")}):
        raise ValueError("Incomplete paired cache blocks")
    ratios = []
    for record in records:
        order = ["control", "candidate"] if record["block"] % 2 == 0 else ["candidate", "control"]
        if record["order"] != order:
            raise ValueError("Unbalanced timing order")
        medians = {}
        for arm in order:
            values = record[arm + "_samples_us"]
            if (len(values) != 12 or any(not isinstance(v, (int, float))
                    or not math.isfinite(v) or v <= 0 for v in values)):
                raise ValueError("Require twelve finite positive timing samples per arm")
            medians[arm] = statistics.median(values)
        ratios.append({"mode": record["mode"], "first": order[0], **medians,
                       "ratio": medians["candidate"] / medians["control"]})
    return [{"mode": mode,
        "control_median_us": statistics.median(r["control"] for r in ratios if r["mode"] == mode),
        "candidate_median_us": statistics.median(r["candidate"] for r in ratios if r["mode"] == mode),
        "median_paired_ratio": statistics.median(r["ratio"] for r in ratios if r["mode"] == mode),
        "order_ratios": {first: statistics.median(r["ratio"] for r in ratios
            if r["mode"] == mode and r["first"] == first) for first in ("control", "candidate")}}
        for mode in ("warm", "evicted")]


def decoder_audit(report: dict[str, Any]) -> dict[str, Any]:
    """Require the original decoder gates plus full nibble/fallback coverage when enabled."""
    nibble_stage = report.get("candidate_nibble_stage", False)
    cases = report["decoder"]["cases"]
    count = 24 if nibble_stage else 6
    if (report["decoder"]["status"] != "passed" or len(cases) != count
            or {c["case"] for c in cases} != set(range(count))):
        raise ValueError("Incomplete GPU decoder qualification")
    if any(not c["bits_exact"] or not c["nondefault_stream"] or c["graph_replay_growth_bytes"]
           for c in cases):
        raise ValueError("Decoder bytes, stream or graph gate failed")
    result: dict[str, Any] = {"cases": count, "bytes_stream_graph_exact": True}
    if not nibble_stage:
        return result
    if not report.get("candidate_shared_stage") or not all(
            c.get("stage_metadata_verified") for c in cases):
        raise ValueError("Nibble decoder lacks complete source-range proofs")
    bases = [c["aligned_nibble_base"] for c in cases if c["aligned_nibble_base"] is not None]
    if set(bases) != set(range(0, 256, 16)) or len(bases) != 20:
        raise ValueError("Missing aligned bucket or exact U8 decoder fallback")
    fallback = report.get("fallback_cases", [])
    if len(fallback) != 4 or {(c["rank"], c["tokens"]) for c in fallback} != {
            (rank, m) for rank in (0, 1) for m in (1, 128)}:
        raise ValueError("Missing both-rank exact fallback GEMMs")
    for case in fallback:
        if (not all(case[k] for k in ("synthetic_scale_edit",
                "complete_range_forces_u8_fallback", "concentrated_routing",
                "current_codec_control_candidate_eager_stream_graph_bits_exact",
                "managed_locks_reset", "dynamic_eager_stream_graph_matches_stock_for_own_alignment"))
                or case["arithmetic_reference"] != "Previous exact U8 shared-stage consumer"):
            raise ValueError("Synthetic fallback GEMM failed its exactness or scratch gates")
        if (report.get("candidate_static_nibble_stage")
                and not case.get("fallback_selected_before_launch")):
            raise ValueError("Static staging did not select exact fallback before launch")
    result.update(nibble_decoder_cases=len(bases), u8_decoder_cases=count - len(bases),
                  exact_fallback_gemm_cases=len(fallback))
    return result


def review(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep current-codec exactness separate from installed-stock and full-model qualification."""
    if len(reports) != 2 or {r.get("confirmation") for r in reports} != {False, True}:
        raise ValueError("Require a fresh confirmation process")
    for field in ("code_sha256", "runtime_sha256", "codec_sha256", "build_manifest_sha256",
                  "fixture_manifest_sha256", "prior_sha256", "original_control_sha256"):
        if not reports[0].get(field) or len({r.get(field) for r in reports}) != 1:
            raise ValueError(f"Missing or changed input binding: {field}")
    for field in ("candidate_affine_only", "candidate_shared_stage", "candidate_nibble_stage",
                  "candidate_static_nibble_stage"):
        if reports[0].get(field, False) != reports[1].get(field, False):
            raise ValueError("Native consumer policy changed between confirmation processes")
    memory_passed, performance_passed, stock_exact = True, True, True
    sessions = []
    for report in sorted(reports, key=lambda r: r["confirmation"]):
        if report.get("diagnostic"):
            raise ValueError("Launch-blocking diagnostics cannot qualify performance")
        if report["status"] != "passed":
            raise ValueError("Incomplete GPU qualification")
        decoder = decoder_audit(report)
        partitions = report["partitions"]
        if len(partitions) != 2 or {p["rank"] for p in partitions} != {0, 1}:
            raise ValueError("Require both real TP2 partitions")
        session = {"confirmation": report["confirmation"], "partitions": [],
                   "seconds": report["seconds"], "cuda_peak_bytes": report["cuda_peak_bytes"],
                   "decoder_qualification": decoder}
        for partition in partitions:
            storage = partition["storage"]
            if len(storage) != 2 or {s["family"] for s in storage} != {"w13", "w2"}:
                raise ValueError("Missing scale ownership records")
            for item in storage:
                if not item["source_bytes_exact"]:
                    raise ValueError("Source scale bytes differ")
                saving = 1 - item["candidate_cuda_allocation_bytes"] / item["control_cuda_allocation_bytes"]
                memory_passed &= saving >= .20
            cases = partition["cases"]
            if (len(cases) != 14 or {(c["tokens"], c["concentrated_routing"]) for c in cases}
                    != {(size, skewed) for size in SIZES for skewed in (False, True)}):
                raise ValueError("Missing decode/prefill or routing cases")
            rows = []
            for case in cases:
                if (not case["current_codec_control_candidate_eager_stream_graph_bits_exact"]
                        or not case["managed_locks_reset"]
                        or not case["dynamic_eager_stream_graph_matches_stock_for_own_alignment"]
                        or case["arithmetic_reference"] != "original qualified compressed-codec library"):
                    raise ValueError("Current-codec arithmetic or scratch gate failed")
                stock_exact &= all(d["whole_output_bits_exact"]
                                   for d in case["installed_stock_diagnostic"].values())
                row = {k: case[k] for k in ("tokens", "concentrated_routing", "moe_block",
                                            "gate_up_plan", "down_plan", "installed_stock_diagnostic")}
                for key in ("graph_latency", "eager_latency"):
                    required = key == "graph_latency" or (case["tokens"] in (1, 32)
                                                          and not case["concentrated_routing"])
                    if required and key not in case:
                        raise ValueError("Missing paired timing mode")
                    if key not in case:
                        continue
                    timing = case[key]
                    if timing["allocation_growth_bytes"]:
                        raise ValueError("Timing replay grew CUDA allocations")
                    row[key] = latency(timing["records"])
                    for result in row[key]:
                        performance_passed &= max(result["median_paired_ratio"],
                                                  *result["order_ratios"].values()) <= 1.02
                rows.append(row)
            session["partitions"].append({"rank": partition["rank"], "storage": storage, "cases": rows})
        sessions.append(session)
    return {"status": "complete", "current_codec_bits_exact": True,
        "consumer": {"affine_only": reports[0].get("candidate_affine_only", False),
                     "shared_stage": reports[0].get("candidate_shared_stage", False),
                     "nibble_stage": reports[0].get("candidate_nibble_stage", False),
                     "static_nibble_stage": reports[0].get("candidate_static_nibble_stage", False)},
        "incremental_scale_memory_gate_passed": bool(memory_passed),
        "latency_nonregression_gate_passed": bool(performance_passed),
        "incremental_component_gate_passed": bool(memory_passed and performance_passed),
        "installed_stock_bits_exact": bool(stock_exact), "formal_model_quality": "unqualified",
        "declared_gates": {"minimum_incremental_allocated_scale_saving": .20,
                           "maximum_paired_latency_ratio_both_cache_and_arm_orders": 1.02},
        "sessions": sessions,
        "limits": ["Complete expert layer, not a whole-model memory or throughput result.",
            "Current-codec exactness does not establish installed-stock arithmetic equivalence.",
            "Full-model quality and representative serving remain separate qualification steps.",
            "Confirmation is a separate process in the same service-pause window."]}


def main() -> None:
    """Write an input-hash-bound review using no GPU or third-party runtime imports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs=2, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-audit", type=Path)
    parser.add_argument("--ncu-directory", type=Path)
    args = parser.parse_args()
    result = review([json.loads(p.read_bytes()) for p in args.reports])
    if args.cpu_audit is not None:
        result["complete_scale_cpu_audit"] = cpu_audit(json.loads(args.cpu_audit.read_bytes()))
    if args.ncu_directory is not None:
        result["counter_review"] = ncu_review(args.ncu_directory)
    inputs = [*args.reports, *([args.cpu_audit] if args.cpu_audit is not None else [])]
    result["inputs"] = {p.name: {"bytes": p.stat().st_size,
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in inputs}
    result["analysis_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
