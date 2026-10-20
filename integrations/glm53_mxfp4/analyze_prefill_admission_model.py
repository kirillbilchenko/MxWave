"""Recompute the frozen admission comparison, source scope and unchanged decode gates."""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

from analyze_paired_speed import timing_metrics
from benchmark_full_model import write
from decode_quality import compare


def forward_histogram(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> dict[int, int]:
    """Require both ranks' eager call deltas and distinguish graph replays from Python calls."""
    if {r["rank"] for r in before} != {0, 1} or {r["rank"] for r in after} != {0, 1}:
        raise ValueError("Require both distinct TP ranks")
    values = []
    for first, last in zip(before, after, strict=True):
        if first["rank"] != last["rank"]:
            raise ValueError("Counter rank identity changed")
        a = first["stable_dispatch"]["calls_by_tokens"]
        b = last["stable_dispatch"]["calls_by_tokens"]
        delta = {int(k): b.get(k, 0) - a.get(k, 0) for k in a.keys() | b.keys()
                 if b.get(k, 0) != a.get(k, 0)}
        if any(k < 2 or n < 1 or n % 42 for k, n in delta.items()):
            raise ValueError("Incomplete eager base-model forward count")
        values.append({k: n // 42 for k, n in sorted(delta.items())})
    if len(values) != 2 or values[0] != values[1]:
        raise ValueError("Both rank eager counters must agree")
    return values[0]


def weight_checks(profile: dict[str, Any]) -> bool:
    """Require each profiled eager forward to visit all source-identified KDA weights once."""
    armed = {r["rank"]: r for r in profile.get("weight_observation_armed", [])}
    observed = {r["rank"]: r for r in profile.get("weight_observation", [])}
    if profile["status"] != "complete" or set(armed) != {0, 1} or set(observed) != {0, 1}:
        return False
    for rank in (0, 1):
        expected = armed[rank]["expected_weight_prefixes"]
        value = observed[rank]
        if (len(expected) != 68 or len(set(expected)) != 68
                or not value["saved_methods_restored"] or value["during_unprofiled_timing"]
                or not value["events"]):
            return False
        for event in value["events"]:
            if (event["duplicate_weight_prefixes"] or len(event["calls"]) != 68
                    or sorted(c["prefix"] for c in event["calls"]) != expected):
                return False
    return True


def numerical_passed(value: dict[str, Any]) -> bool:
    """Apply unchanged KL/NLL floors to every identically conditioned actual decode step."""
    return bool(value["decode_steps"] == value["expected_decode_steps"] == 48
        and value["all_decode_conditioning_matches"] and value["all_generated_ids_match"]
        and value["mean_decode_kl"] is not None and 0 <= value["mean_decode_kl"] <= 1e-5
        and value["mean_decode_nll_difference"] is not None
        and abs(value["mean_decode_nll_difference"]) <= 1e-4)


def analyze(root: Path) -> dict[str, Any]:
    """Keep all failures and timing rows; no serving improvement follows from batching alone."""
    path = root / "prefill-admission-model-review.json"
    if path.exists():
        raise FileExistsError("Preserve the first admission analysis")
    plan = json.loads((root / "prefill-admission-plan.json").read_bytes())
    report_path = root / "prefill-admission-measurement.json"
    report = json.loads(report_path.read_bytes())
    if (report["status"] != "complete" or len(report["trials"]) != 108
            or len(report["warmups"]) != 30 or len(report["quality"]) != 6):
        raise ValueError("Incomplete admission timing grid")
    expected_keys = {(name, trial, window) for name, _, _, _, repeats in plan["cases"]
                     for trial in range(repeats) for window in plan["windows_ms"]}
    rows = []
    seen = set()
    for trial in report["trials"]:
        key = (trial["case"], trial["trial"], trial["window_ms"])
        if key not in expected_keys or key in seen:
            raise ValueError("Duplicated or unexpected admission trial")
        seen.add(key)
        metrics = timing_metrics(trial["measurement"], trial["concurrency"])
        histogram = forward_histogram(trial["before"], trial["after"])
        count = sum(n for m, n in histogram.items() if m > 8)
        admission = trial["admission"]
        if admission["window_ms"] != trial["window_ms"]:
            raise ValueError("Timed input window differs from its recorded arm")
        if any(e["deadline_extended"] for e in admission["events"]):
            raise ValueError("Input collection extended the frozen maximum window")
        rows.append({"case": key[0], "trial": key[1], "window_ms": key[2], **metrics,
            "eager_forward_histogram": histogram, "large_eager_forwards": count,
            "inferred_weight_reconstructions_per_rank": 68 * count,
            "observed_input_collection_seconds": sum(e["elapsed_seconds"]
                                                       for e in admission["events"]),
            "input_window_events": admission["events"]})
    if seen != expected_keys:
        raise ValueError("Missing admission trials")
    summaries = []
    for name, _, _, _, repeats in plan["cases"]:
        for window in plan["windows_ms"]:
            selected = sorted((r for r in rows if r["case"] == name and r["window_ms"] == window),
                              key=lambda r: r["trial"])
            if len(selected) != repeats:
                raise ValueError("Missing case/arm timing samples")
            summaries.append({"case": name, "window_ms": window, "trials": repeats,
                "median_tps": statistics.median(r["aggregate_tokens_per_second"] for r in selected),
                "median_ttft_seconds": statistics.median(r["median_ttft_seconds"] for r in selected),
                "median_large_eager_forwards": statistics.median(r["large_eager_forwards"]
                                                                  for r in selected),
                "min_tps": min(r["aggregate_tokens_per_second"] for r in selected),
                "max_tps": max(r["aggregate_tokens_per_second"] for r in selected)})
    lookup = {(r["case"], r["window_ms"]): r for r in summaries}
    ids = []
    for window in (3, 10):
        matched, total = 0, 0
        for name, _, _, _, repeats in plan["cases"]:
            for trial in range(repeats):
                arms = [next(t for t in report["trials"] if
                    (t["case"], t["trial"], t["window_ms"]) == (name, trial, value))
                    for value in (0, window)]
                for a, b in zip(arms[0]["measurement"]["requests"],
                                arms[1]["measurement"]["requests"], strict=True):
                    total += 1
                    matched += a["token_ids"] == b["token_ids"]
        ids.append({"window_ms": window, "matching_sequences": matched,
                    "paired_sequences": total, "all_match": matched == total})
    decisions = []
    for window in (3, 10):
        ratios = {name: lookup[name, window]["median_tps"] / lookup[name, 0]["median_tps"]
                  for name, *_ in plan["cases"]}
        c1_penalty = lookup["c1-short", window]["median_ttft_seconds"] - lookup[
            "c1-short", 0]["median_ttft_seconds"]
        fewer = (lookup["c4-short", window]["median_large_eager_forwards"] <
                 lookup["c4-short", 0]["median_large_eager_forwards"])
        speed = (ratios["c4-short"] >= plan["primary_min_tps_ratio"]
            and all(ratios[name] >= plan["control_min_tps_ratio"] for name in
                    ("c1-short", "c8-short", "c4-long", "c4-stagger"))
            and c1_penalty <= plan["max_c1_ttft_penalty_seconds"])
        decisions.append({"window_ms": window, "median_tps_ratios_vs_zero": ratios,
            "c1_ttft_penalty_seconds": c1_penalty,
            "fewer_primary_large_forwards": fewer, "observed_speed_gate_passed": speed,
            "scheduling_mechanism_and_speed_gate_passed": speed and fewer})
    numerical = []
    for first, second in (("window0-a", "window0-b"), ("window3-a", "window3-b"),
                          ("window10-a", "window10-b"),
                          ("window0-a", "window3-a"), ("window0-b", "window3-b"),
                          ("window0-a", "window10-a"), ("window0-b", "window10-b")):
        a, b = [json.loads((root / (n + "-decode-quality.json")).read_bytes())
                for n in (first, second)]
        value = compare(a, b, root)
        numerical.append({"reference": first, "candidate": second,
                          "passed": numerical_passed(value), **value})
    profiles_ok = len(report["profiles"]) == 3 and all(weight_checks(p) for p in report["profiles"])
    source_ok = all(len(report[key]) == 2 and all(r["status"] == "passed" for r in report[key])
                    for key in ("source_before", "source_after", "loaded_checks"))
    result = {"status": "complete", "source_checks_passed": source_ok,
        "timed_trial_groups": len(rows), "all_trials_retained": True,
        "summaries": summaries, "decisions": decisions, "trials": rows,
        "timed_output_ID_agreement": ids,
        "profile_weight_identity_checks_passed": profiles_ok,
        "numerical_comparisons": numerical, "numerical_gate_passed": all(r["passed"] for r in numerical),
        "quality_qualified": False, "previous_failed_gates_preserved": True,
        "plan_sha256": hashlib.sha256((root / "prefill-admission-plan.json").read_bytes()).hexdigest(),
        "raw_measurement_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limits": ["One resident model with interleaved window order; no full stock-model reload.",
            "Timing includes client launch staggering and completion drain; no profile time is used.",
            "Timed rebuild counts follow the frozen >8-row path; per-weight identities are untimed.",
            "All three windows receive C4 full-vocabulary repeats; other concurrency quality is untested.",
            "Earlier MLA/repeat and practical-format failures remain; no release qualification."]}
    write(path, result)
    return result
