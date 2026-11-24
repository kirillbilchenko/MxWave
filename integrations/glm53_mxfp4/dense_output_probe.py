"""Prospective screening for a dense-output storage tradeoff, without model promotion."""

from __future__ import annotations

import math
from statistics import median
from typing import Any

PAIRS = ("parent_repeat", "dense_vs_parent", "dense_repeat")


def protocol() -> dict[str, Any]:
    """Freeze representative weights, paired timing, memory bounds and screening gates."""
    return {
        "revision": "denseoutputv1", "ranks": [0, 1],
        "selection": "min_median_max_packed_bytes_per_rank_then_prefix",
        "fixtures_per_rank": 3, "output_projections_per_rank": 34,
        "tokens": [1, 8, 16, 54, 256, 2048], "fused_tokens": [1, 8, 16],
        "trials": 16, "samples_per_leg": 8, "pairs": list(PAIRS),
        "execution_modes": ["eager", "graph"], "cache_regimes": ["warm", "pressure"],
        "pressure_bytes": 128 * 1024**2, "warmup_seconds": 3,
        "control_ratio_limit": .02, "prefill_candidate_ratio_limit": .95,
        "decode_extra_us_per_projection_limit": 50.,
        "relative_l2_limit": .01, "activation_seed": 1909,
        "CUDA_allocation_cap_bytes": 2 * 1024**3,
        "container_memory_bytes": 8 * 1024**3, "component_timeout_seconds": 900,
        "unchanged_model_speed_and_quality_gates": True,
        "model_quality_or_serving_speed_qualified": False,
    }


def select_cases(parent: dict[str, Any]) -> list[dict[str, Any]]:
    """Select each rank's compression extremes from a completed, unchanged model load."""
    if parent["status"] != "complete" or parent["arm"] != "candidate":
        raise ValueError("Require the completed compressed parent")
    result = []
    for rank in (0, 1):
        before = next(x for x in parent["source_before"] if x["tp_rank"] == rank)
        after = next(x for x in parent["source_after"] if x["tp_rank"] == rank)
        layers = before["allocation"]["kda"]["layers"]
        if layers != after["allocation"]["kda"]["layers"] or len(layers) != 68:
            raise ValueError("Require all unchanged KDA storage records")
        outputs = sorted((x for x in layers if x["kind"] == "output"),
                         key=lambda x: (x["packed_bytes"], x["prefix"]))
        if len(outputs) != 34 or len({x["prefix"] for x in outputs}) != 34:
            raise ValueError("Require 34 unique output projections per rank")
        for position, label in ((0, "minimum"), (17, "median"), (33, "maximum")):
            entry = outputs[position]
            prefix = entry["prefix"]
            layer = int(prefix.split(".layers.")[1].split(".")[0])
            if (prefix != f"language_model.model.layers.{layer}.self_attn.o_proj"
                    or entry["rank"] != rank or entry["logical_shape"] != [4096, 4096]
                    or entry["native_shape"] != [4096, 4096]
                    or entry["raw_bytes"] != 33554432
                    or not 0 < entry["packed_bytes"] < entry["raw_bytes"]
                    or not entry["gpu_decode_bits_exact"]):
                raise ValueError("Selected output differs from the qualified geometry/storage")
            result.append({
                "name": f"kda-output-l{layer}-r{rank}", "rank": rank, "prefix": prefix,
                "selection_position": position, "selection_label": label,
                "logical_shape": [4096, 4096], "native_shape": [4096, 4096],
                "logical_weight_sha256": entry["loaded_weight_sha256"],
                "raw_bytes": entry["raw_bytes"], "expected_packed_bytes": entry["packed_bytes"],
                "source_tensors": [{
                    "name": f"model.language_model.layers.{layer}.self_attn.o_proj.weight",
                    "axis": 1, "range": [rank * 4096, (rank + 1) * 4096],
                }],
            })
    return result


def summarize(samples: list[dict[str, Any]], tokens: int) -> dict[str, Any]:
    """Keep every paired event and require repeat stability in both timing directions."""
    p = protocol()
    if len(samples) != p["trials"] or tokens not in p["tokens"]:
        raise ValueError("Missing timing trials or undeclared shape")
    records: dict[str, list[dict[str, float]]] = {name: [] for name in PAIRS}
    for trial, row in enumerate(samples):
        pairs = list(PAIRS) if trial % 2 == 0 else list(reversed(PAIRS))
        order = ["a", "b", "b", "a"] if trial % 2 == 0 else ["b", "a", "a", "b"]
        if (row["trial"] != trial or row["pair_order"] != pairs
                or set(row["pairs"]) != set(PAIRS)):
            raise ValueError("Paired timing order or coverage changed")
        for name in pairs:
            legs = row["pairs"][name]
            if [x["arm"] for x in legs] != order:
                raise ValueError("Require balanced ABBA/BAAB legs")
            values: dict[str, list[float]] = {"a": [], "b": []}
            for leg in legs:
                times = leg["us"]
                if (len(times) != p["samples_per_leg"]
                        or any(not math.isfinite(v) or v <= 0 for v in times)):
                    raise ValueError("Invalid or missing CUDA event sample")
                values[leg["arm"]].extend(times)
            first, second = median(values["a"]), median(values["b"])
            records[name].append({"reference_us": first, "candidate_us": second,
                                  "ratio": second / first, "extra_us": second - first})
    summarized = {}
    for name, rows in records.items():
        combined = {key: median(row[key] for row in rows) for key in rows[0]}
        orders = {order: {key: median(rows[i][key] for i in range(parity, len(rows), 2))
                          for key in rows[0]}
                  for parity, order in enumerate(("ABBA", "BAAB"))}
        summarized[name] = {"combined": combined, "orders": orders, "per_trial": rows}
    stable = all(abs(summary["ratio"] - 1) <= p["control_ratio_limit"]
                 for name in ("parent_repeat", "dense_repeat")
                 for summary in [summarized[name]["combined"],
                                 *summarized[name]["orders"].values()])
    candidate = summarized["dense_vs_parent"]
    candidate_summaries = [candidate["combined"], *candidate["orders"].values()]
    cost_passes = all(
        row["extra_us"] <= p["decode_extra_us_per_projection_limit"]
        if tokens in p["fused_tokens"] else row["ratio"] <= p["prefill_candidate_ratio_limit"]
        for row in candidate_summaries
    )
    return {"pairs": summarized, "repeat_controls_passed": stable,
            "candidate_cost_screen_passed": cost_passes,
            "component_timing_screen_passed": stable and cost_passes,
            "events": p["trials"] * len(PAIRS) * 4 * p["samples_per_leg"],
            "decode_budget_is_model_speed_qualification": False}


def assess(report: dict[str, Any]) -> dict[str, Any]:
    """Require complete sampled coverage while keeping model qualification separate."""
    p = protocol()
    if report["status"] != "complete" or report["protocol"] != p:
        raise ValueError("Incomplete report or changed prospective protocol")
    rank = report["rank"]
    if rank not in (0, 1) or len(report["cases"]) != 3:
        raise ValueError("Require all three selected fixtures on a declared rank")
    if [x["selection_position"] for x in report["cases"]] != [0, 17, 33]:
        raise ValueError("Selected compression extremes changed")
    summaries = []
    numerical_passed = True
    for case in report["cases"]:
        if (case["rank"] != rank or not case["weight_bits_exact"]
                or not case["destination_guards_unchanged"]
                or case["packed_bytes"] != case["expected_packed_bytes"]
                or not case["packed_buffers_unchanged"]
                or case["dense_weight_sha256_after"] != case["logical_weight_sha256"]
                or [x["tokens"] for x in case["measurements"]] != p["tokens"]):
            raise ValueError("Fixture source, exact reconstruction or shape coverage failed")
        for row in case["measurements"]:
            if (not row["dense_source_bits_exact"] or not row["parent_eager_graph_bits_exact"]
                    or not row["dense_eager_graph_bits_exact"] or not row["repeats_bits_exact"]
                    or row["graph_replay_retained_allocation_bytes"] != 0
                    or (row["tokens"] not in p["fused_tokens"]
                        and not row["parent_source_bits_exact"])):
                raise ValueError("An unchanged execution path or graph lifetime check failed")
            arithmetic = row["arithmetic"]
            arithmetic_passed = bool(arithmetic["finite"] and arithmetic["relative_l2"] is not None
                                     and arithmetic["relative_l2"] < p["relative_l2_limit"])
            if arithmetic["passed"] != arithmetic_passed:
                raise ValueError("Arithmetic gate does not reproduce")
            numerical_passed &= arithmetic_passed
            if {(x["mode"], x["cache"]) for x in row["timing"]} != {
                (mode, cache) for mode in p["execution_modes"] for cache in p["cache_regimes"]
            } or len(row["timing"]) != 4:
                raise ValueError("Missing or duplicated cache/mode measurement")
            for measured in row["timing"]:
                summary = summarize(measured["samples"], row["tokens"])
                if summary != measured["summary"]:
                    raise ValueError("Stored timing summary does not reproduce")
                summaries.append({"name": case["name"], "tokens": row["tokens"],
                                  "mode": measured["mode"], "cache": measured["cache"],
                                  **summary})
    if report["CUDA_peak_bytes"] > p["CUDA_allocation_cap_bytes"]:
        raise ValueError("Component exceeded its CUDA memory bound")
    failures = [{key: row[key] for key in ("name", "tokens", "mode", "cache")}
                for row in summaries if not row["component_timing_screen_passed"]]
    return {"diagnostic_complete": True, "rank": rank, "fixtures": 3,
            "numerical_component_screen_passed": numerical_passed,
            "component_screen_passed": numerical_passed and not failures,
            "failed_conditions": failures, "summaries": summaries,
            "events": sum(row["events"] for row in summaries),
            "model_quality_or_serving_speed_qualified": False,
            "model_CPU_saving_demonstrated": False, "automatic_promotion_allowed": False}
