"""Reproduce the BF16 component review from timings and NVTX-labeled Nsight Compute CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

CASES = ("kda-input-l0-r0", "kda-output-l0-r0", "head-r0")
METRICS = (
    "gpu__time_duration.sum", "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__waves_per_multiprocessor", "launch__registers_per_thread",
    "launch__registers_per_thread_allocated", "launch__shared_mem_per_block",
    "launch__occupancy_limit_shared_mem", "launch__occupancy_limit_registers",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "smsp__warps_eligible.avg.per_cycle_active",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "profiler__replayer_passes",
)


def review(timing: dict[str, Any], rows: list[list[str]]) -> dict[str, Any]:
    """Reject missing arms and preserve the distinction between component and model results."""
    if (timing.get("status") != "passed" or timing.get("profile")
            or tuple(c["name"] for c in timing["cases"]) != CASES):
        raise ValueError("Require the complete unprofiled three-component comparison")
    cases = []
    rejected = []
    for case in timing["cases"]:
        expected = {(m, s) for m in (1, 4, 8) for s in (1, 2, 4, 8)}
        records = case["measurements"]
        if (not case["gpu_weight_bits_exact"] or len(records) != len(expected)
                or {(r["tokens"], r["split_k"]) for r in records} != expected):
            raise ValueError("Incomplete component timing or changed weight bytes")
        current = []
        for record in records:
            if record["split_k"] != 1:
                rejected.append({"name": case["name"], "tokens": record["tokens"],
                    "split_k": record["split_k"], "passed": record["candidate_gates_passed"],
                    "relative_l2": record["arithmetic"]["relative_l2"],
                    "eager_ratio": record["eager"]["median_paired_ratio"],
                    "graph_ratio": record["graph"]["median_paired_ratio"]})
                continue
            if not record["candidate_gates_passed"]:
                raise ValueError("Current split-K=1 failed an existing gate")
            item: dict[str, Any] = {"tokens": record["tokens"], "split_k": 1,
                "arithmetic": record["arithmetic"], "gates_passed": True}
            for mode in ("eager", "graph"):
                data = record[mode]
                if ({tuple(s["order"]) for s in data["samples"]}
                        != {("dense", "fused"), ("fused", "dense")}):
                    raise ValueError("Require both timing orders")
                item[mode] = {k: data[k] for k in (
                    "dense_median_us", "fused_median_us", "median_paired_ratio")}
                item[mode]["order_median_ratios"] = {first: statistics.median(
                    s["cuda_ratio"] for s in data["samples"] if s["order"][0] == first)
                    for first in ("dense", "fused")}
            current.append(item)
        cases.append({k: case[k] for k in ("name", "logical_shape", "native_shape", "raw_bytes",
            "packed_bytes", "gpu_weight_bits_exact", "source_weight_sha256", "artifact_sha256")}
            | {"current": current})
    header = next(row for row in rows if row and row[0] == "ID")
    start = rows.index(header)
    units = dict(zip(header, rows[start + 1], strict=True))
    kernels = [dict(zip(header, row, strict=True)) for row in rows[start + 2:]
               if len(row) == len(header) and row[0].isdigit()]
    expected_groups = {(name, arm) for name in CASES for arm in ("dense", "fused_s1", "fused_s4")}
    groups: set[tuple[str, str]] = set()
    counters = []
    for kernel in kernels:
        parts = kernel["Kernel Name"].split("/", 4)
        if len(parts) != 5 or parts[0] != "bf16_profile" or parts[2] != "M4":
            raise ValueError("Every profile launch needs an explicit component/arm NVTX label")
        group = (parts[1], parts[3])
        groups.add(group)
        if not any(name in parts[4] for name in ("BF16TripleBitmap_MM_Kernel", "cutlass::Kernel2")):
            continue
        metrics = {}
        for name in METRICS:
            value = kernel.get(name)
            metrics[name] = {"value": (float(value.replace(",", ""))
                                      if value and value != "no data" else None),
                             "unit": units.get(name)}
        counters.append({"id": int(kernel["ID"]), "case": parts[1], "arm": parts[3],
            "kernel": parts[4], "grid": kernel["Grid Size"], "block": kernel["Block Size"],
            "metrics": metrics})
    if groups != expected_groups or len(counters) != 9 or len(kernels) != 20:
        raise ValueError("Missing counter groups, main kernels or expected helper launches")
    extrapolation = []
    for tokens, mode in ((1, "graph"), (4, "eager"), (8, "eager")):
        records = [next(r for r in c["current"] if r["tokens"] == tokens)[mode] for c in cases]
        totals = {arm: (34 * sum(r[arm + "_median_us"] for r in records[:2])
                        + records[2][arm + "_median_us"]) / 1000 for arm in ("dense", "fused")}
        extrapolation.append({"tokens": tokens, "mode": mode, **totals,
                              "fused_minus_dense_ms": totals["fused"] - totals["dense"]})
    return {"status": "complete", "device": timing["device"], "cases": cases,
        "current_gate_passes": 9, "diagnostic_split_cases": rejected,
        "diagnostic_splits_all_rejected": all(not r["passed"] for r in rejected),
        "profile_main_kernels": counters, "profile_total_kernels": len(kernels),
        "dram_byte_counters_present": any(name.startswith("dram__bytes") for name in header),
        "layer0_extrapolation_not_full_model": extrapolation,
        "limits": [*timing["limits"],
            "One GPU/component session; eight alternating-order trials per comparison.",
            "Layer-zero extrapolation repeats its pair 34 times, then adds one head.",
            "Extrapolation is a scale estimate, not measured full-model throughput or causality.",
            "NCU uses cold-cache kernel replay, 17 passes and unmodified GPU clocks.",
            "Compute/cache throughput percentages are not LPDDR bandwidth utilization.",
            "No new model quality measurement or production NVFP4/DFlash qualification."]}


def main() -> None:
    """Write a compact, input-hash-bound review without importing CUDA or loading weights."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = review(json.loads(args.timing.read_bytes()),
                    list(csv.reader(args.csv.read_text().splitlines())))
    result["inputs"] = {p.name: {"bytes": p.stat().st_size,
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in (args.timing, args.csv)}
    result["analysis_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
