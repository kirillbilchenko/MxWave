"""Frozen scope and CPU review for a diagnostic of the remaining prefill cost."""

from __future__ import annotations

import math
from statistics import median
from typing import Any

METRICS = (
    "gpu__time_duration.sum", "lts__t_bytes.sum", "lts__t_sectors_op_read.sum",
    "lts__t_sectors_op_write.sum", "lts__t_sector_hit_rate.pct",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "smsp__inst_executed.sum", "smsp__warps_active.avg.pct_of_peak_sustained_active",
)


def protocol() -> dict[str, Any]:
    """Describe a diagnostic without changing serving or a prior qualification gate."""
    return {
        "revision": "kdamemoryv1", "ranks": [0, 1],
        "fixture_names": ["kda-input-l0-r0", "kda-output-l0-r0"],
        "prefill_rows": [54, 256, 2048], "copy_MiB": [4, 128],
        "copy_blocks": [1024, 4096, 16384],
        "trials": 8, "samples_per_leg": 4, "control_ratio_limit": .02,
        "cache_regimes": ["warm", "pressure"], "execution_modes": ["eager", "graph"],
        "pressure_bytes": 128 * 1024**2, "CUDA_allocation_cap_bytes": 2 * 1024**3,
        "container_memory_bytes": 8 * 1024**3, "component_timeout_seconds": 600,
        "profile_timeout_seconds": 900, "NCU_cache_control": "all",
        "NCU_clock_control": "none", "metrics": list(METRICS),
        "unavailable_requested_metrics": ["dram__bytes_read.sum", "dram__bytes_write.sum"],
        "same_rank_zero_fixtures_on_each_GPU": True,
        "model_quality_or_serving_speed_qualified": False,
    }


def operation_names() -> list[str]:
    """Enumerate all prospective operations, including exact repeat controls."""
    p = protocol()
    result = [f"copy-{size}MiB-b{block}" for size in p["copy_MiB"]
              for block in p["copy_blocks"]]
    for fixture in p["fixture_names"]:
        result.extend([fixture + "/decode", fixture + "/copy-dense"])
        for rows in p["prefill_rows"]:
            result.extend([f"{fixture}/M{rows}/dense", f"{fixture}/M{rows}/decode-dense"])
    return result


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Retain both orders and recompute the unchanged-call repeat control."""
    p = protocol()
    if len(samples) != p["trials"]:
        raise ValueError("Missing timing trials")
    medians, ratios = [], []
    for i, row in enumerate(samples):
        expected = ["a", "b", "b", "a"] if i % 2 == 0 else ["b", "a", "a", "b"]
        if row["trial"] != i or [leg["arm"] for leg in row["legs"]] != expected:
            raise ValueError("Require the prospectively balanced ABBA/BAAB order")
        values: dict[str, list[float]] = {"a": [], "b": []}
        for leg in row["legs"]:
            times = leg["us"]
            if (len(times) != p["samples_per_leg"]
                    or any(not math.isfinite(v) or v <= 0 for v in times)):
                raise ValueError("Invalid or missing event samples")
            values[leg["arm"]].extend(times)
        medians.append(median(values["a"] + values["b"]))
        ratios.append(median(values["b"]) / median(values["a"]))
    orders = {name: median(ratios[i] for i in range(len(ratios)) if i % 2 == parity)
              for parity, name in enumerate(("ABBA", "BAAB"))}
    combined = median(ratios)
    return {"median_us": median(medians), "per_trial_us": medians,
            "median_repeat_ratio": combined, "repeat_ratio_by_order": orders,
            "repeat_control_passed": all(abs(v - 1) <= p["control_ratio_limit"]
                                         for v in [combined, *orders.values()]),
            "events": len(samples) * 4 * p["samples_per_leg"]}


def assess(report: dict[str, Any]) -> dict[str, Any]:
    """Verify coverage and exactness, preserving unstable controls as diagnostic failures."""
    p = protocol()
    if report["protocol"] != p or report["status"] != "complete":
        raise ValueError("Incomplete diagnostic or changed prospective protocol")
    if [r["name"] for r in report["operations"]] != operation_names():
        raise ValueError("Missing or reordered operation")
    summaries, unstable = [], []
    for item in report["operations"]:
        if not item["exact_output_and_guards"]:
            raise ValueError("Output bytes or destination guards changed")
        if report["profile"]:
            if item["profile_label"] != "kda_memory/" + item["name"]:
                raise ValueError("Profile label does not bind the operation")
            continue
        expected = {(mode, cache) for mode in p["execution_modes"] for cache in p["cache_regimes"]}
        if {(m["mode"], m["cache"]) for m in item["measurements"]} != expected:
            raise ValueError("Missing cache condition or execution mode")
        for measured in item["measurements"]:
            result = summarize(measured["samples"])
            if result != measured["summary"]:
                raise ValueError("Stored event summary does not reproduce")
            row = {"name": item["name"], "mode": measured["mode"],
                   "cache": measured["cache"], **result,
                   "logical_payload_GB_per_second": item["logical_payload_bytes"]
                   / result["median_us"] / 1000,
                   "useful_GFLOP_per_second": item["useful_FLOPs"]
                   / result["median_us"] / 1000}
            summaries.append(row)
            if not result["repeat_control_passed"]:
                unstable.append({k: row[k] for k in ("name", "mode", "cache")})
    if report["CUDA_peak_bytes"] > p["CUDA_allocation_cap_bytes"]:
        raise ValueError("The component exceeded its CUDA bound")
    return {"diagnostic_complete": True, "operations": len(report["operations"]),
            "summaries": summaries, "unstable_repeat_controls": unstable,
            "all_repeat_controls_passed": not unstable,
            "model_quality_or_serving_speed_qualified": False,
            "logical_payload_is_measured_DRAM_traffic": False}
