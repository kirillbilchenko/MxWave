"""Prospective paired timing of FP32 partials without overriding failed precision gates."""

from __future__ import annotations

import math
from statistics import median
from typing import Any

PAIRS = ("original_repeat", "rebuilt_repeat", "candidate_repeat",
         "rebuilt_vs_original", "candidate_vs_original", "candidate_vs_rebuilt")


def protocol() -> dict[str, Any]:
    """Freeze both baseline comparisons, repeat controls and resource limits before timing."""
    return {"revision": "fp32speedv1", "ranks": [0, 1], "layers": [0, 22, 44],
        "kinds": ["input", "output"], "tokens": [1, 4, 8, 16], "splits": [2, 4, 8],
        "activation": "787", "execution_modes": ["graph"],
        "cache_regimes": ["warm", "pressure"], "pressure_bytes": 128 * 1024**2,
        "trials": 16, "samples_per_leg": 8, "pairs": list(PAIRS),
        "repeat_ratio_limit": .02, "rebuilt_original_ratio_limit": .02,
        "candidate_ratio_limit": .95, "initial_warmup_seconds": 3,
        "CUDA_allocation_cap_bytes": 2 * 1024**3, "container_memory_bytes": 8 * 1024**3,
        "component_timeout_seconds": 900, "minimum_head_free_bytes": 2 * 1024**3,
        "maximum_report_bytes_per_rank": 128 * 1024**2,
        "padding_and_output_crop_in_graph": True,
        "parent_exact_word_screen_passed": False, "model_loaded": False,
        "model_quality_or_serving_speed_qualified": False, "automatic_promotion_allowed": False}


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Recompute every timing decision from balanced raw events, including all controls."""
    p = protocol()
    if len(samples) != p["trials"]:
        raise ValueError("Missing timing trials")
    records: dict[str, list[dict[str, float]]] = {name: [] for name in PAIRS}
    for trial, row in enumerate(samples):
        pairs = list(PAIRS) if trial % 2 == 0 else list(reversed(PAIRS))
        order = ["a", "b", "b", "a"] if trial % 2 == 0 else ["b", "a", "a", "b"]
        if row["trial"] != trial or row["pair_order"] != pairs or set(row["pairs"]) != set(PAIRS):
            raise ValueError("Timing pair coverage or order changed")
        for name in pairs:
            legs = row["pairs"][name]
            if [leg["arm"] for leg in legs] != order:
                raise ValueError("Require balanced ABBA/BAAB legs")
            values: dict[str, list[float]] = {"a": [], "b": []}
            for leg in legs:
                times = leg["us"]
                if len(times) != p["samples_per_leg"] or any(
                    not math.isfinite(t) or t <= 0 for t in times
                ):
                    raise ValueError("Missing or invalid CUDA events")
                values[leg["arm"]].extend(times)
            a, b = median(values["a"]), median(values["b"])
            records[name].append({"reference_us": a, "candidate_us": b,
                                  "ratio": b / a, "extra_us": b - a})
    summaries = {}
    for name, rows in records.items():
        combined = {key: median(row[key] for row in rows) for key in rows[0]}
        orders = {order: {key: median(rows[i][key] for i in range(parity, len(rows), 2))
                          for key in rows[0]}
                  for parity, order in enumerate(("ABBA", "BAAB"))}
        summaries[name] = {"combined": combined, "orders": orders, "per_trial": rows}

    def ratios(name: str) -> list[float]:
        entry = summaries[name]
        return [entry["combined"]["ratio"], *(v["ratio"] for v in entry["orders"].values())]

    stable = all(abs(ratio - 1) <= p["repeat_ratio_limit"] for name in PAIRS[:3]
                 for ratio in ratios(name))
    equivalent = all(abs(ratio - 1) <= p["rebuilt_original_ratio_limit"]
                     for ratio in ratios("rebuilt_vs_original"))
    faster = all(ratio <= p["candidate_ratio_limit"]
                 for name in PAIRS[4:] for ratio in ratios(name))
    return {"pairs": summaries, "repeat_controls_passed": stable,
        "rebuilt_original_equivalence_passed": equivalent, "candidate_speed_passed": faster,
        "timing_screen_passed": stable and equivalent and faster,
        "events": p["trials"] * len(PAIRS) * 4 * p["samples_per_leg"],
        "parent_exact_word_screen_passed": False, "model_quality_qualified": False}


def assess(report: dict[str, Any]) -> dict[str, Any]:
    """Require full coverage and unchanged outputs while preserving every failed condition."""
    p = protocol()
    rank = report["rank"]
    if report["status"] != "complete" or report["protocol"] != p or rank not in p["ranks"]:
        raise ValueError("Incomplete or altered timing protocol")
    names = [f"kda-{kind}-l{layer}-r{rank}" for layer in p["layers"] for kind in p["kinds"]]
    if [case["name"] for case in report["cases"]] != names:
        raise ValueError("Missing or reordered timing fixtures")
    failures, summaries = [], []
    events = 0
    for case in report["cases"]:
        if not case["decoded_weight_bits_exact"] or not case["packed_buffers_unchanged"]:
            raise ValueError("Weight storage changed")
        wanted = [(n, split) for n in p["tokens"] for split in p["splits"]]
        if [(x["tokens"], x["split"]) for x in case["measurements"]] != wanted:
            raise ValueError("Missing declared row/split combinations")
        for row in case["measurements"]:
            for key in ("outputs_match_parent_before", "outputs_match_parent_after",
                        "guards_unchanged", "graph_matches_eager"):
                if not row[key]:
                    raise ValueError(f"Execution invariant failed: {key}")
            if row["graph_retained_bytes"] != 0:
                raise ValueError("Graph replay retained new allocation")
            if [(x["mode"], x["cache"]) for x in row["timing"]] != [
                ("graph", cache) for cache in p["cache_regimes"]
            ]:
                raise ValueError("Missing cache regimes")
            for timing in row["timing"]:
                summary = summarize(timing["samples"])
                if summary != timing["summary"]:
                    raise ValueError("Stored timing decision differs from raw events")
                key = {"name": case["name"], "tokens": row["tokens"], "split": row["split"],
                       "mode": "graph", "cache": timing["cache"]}
                summaries.append({**key, **summary})
                events += summary["events"]
                if not summary["timing_screen_passed"]:
                    failures.append({**key, **{k: v for k, v in summary.items() if k != "pairs"}})
    return {"diagnostic_complete": True, "rank": rank, "conditions": len(summaries),
        "events": events, "failed_conditions": failures,
        "timing_screen_passed_all_conditions": not failures,
        "parent_exact_word_screen_passed": False,
        "model_quality_or_serving_speed_qualified": False, "automatic_promotion_allowed": False}
