"""Recompute C8/C16 compression comparisons under identical graphs and admission policies."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from analyze_paired_speed import timing_metrics
from benchmark_full_model import write
from decode_quality import compare
from high_concurrency_protocol import CASES, WINDOWS, validate_plan, validate_snapshot


def numerical_passed(value: dict[str, Any]) -> bool:
    """Keep complete actual-decode conditioning, unchanged KL and unchanged NLL requirements."""
    return bool(value["decode_steps"] == value["expected_decode_steps"] == 48
        and value["all_decode_conditioning_matches"] and value["all_generated_ids_match"]
        and value["mean_decode_kl"] is not None and 0 <= value["mean_decode_kl"] <= 1e-5
        and value["mean_decode_nll_difference"] is not None
        and abs(value["mean_decode_nll_difference"]) <= 1e-4)


def quality_pairs() -> list[tuple[str, str]]:
    """Declare every repeat, cross-target and cross-window comparison before reading outcomes."""
    pairs = []
    for count in (8, 16):
        for window in WINDOWS:
            for arm in ("control", "candidate"):
                pairs.append((f"{arm}-c{count}-w{window}-a", f"{arm}-c{count}-w{window}-b"))
            for repeat in ("a", "b"):
                pairs.append((f"control-c{count}-w{window}-{repeat}",
                              f"candidate-c{count}-w{window}-{repeat}"))
        for arm in ("control", "candidate"):
            for repeat in ("a", "b"):
                pairs.append((f"{arm}-c{count}-w0-{repeat}", f"{arm}-c{count}-w3-{repeat}"))
    return pairs


def review(root: Path) -> dict[str, Any]:
    """Retain all raw trials, independent repeats and policy-matched compression differences."""
    plan_path = root / "high-concurrency-plan.json"
    plan = json.loads(plan_path.read_bytes())
    validate_plan(plan)
    audit = json.loads((root / "source-audit.json").read_bytes())
    arms, rows, summaries = {}, [], []
    source_groups = 0
    for label in ("control", "candidate"):
        arm = json.loads((root / (label + "-high-concurrency.json")).read_bytes())
        if (arm["status"] != "complete" or arm["quality_qualified"]
                or arm["plan_sha256"] != hashlib.sha256(plan_path.read_bytes()).hexdigest()
                or len(arm["warmups"]) != 24 or len(arm["quality"]) != 8):
            raise ValueError("Incomplete high-concurrency arm")
        arms[label] = arm
        for phase in ("source_before", "source_after"):
            if len(arm[phase]) != 2 or {r["tp_rank"] for r in arm[phase]} != {0, 1}:
                raise ValueError("Incomplete source rank coverage")
            for row in arm[phase]:
                expected = Counter((tuple(c["shape"]), c["raw_sha256"]) for c in audit["cases"]
                                   if c["rank"] == row["tp_rank"])
                actual = Counter((tuple(c["shape"]), c["sha256"])
                                 for c in row["source_scale_hashes"])
                if (row["status"] != "passed" or row["candidate"] != (label == "candidate")
                        or expected != actual):
                    raise ValueError("Actual source scales differ from the named arm")
                source_groups += sum(actual.values())
        if len(arm["loaded_checks"]) != 2 or any(
                row["status"] != "passed" for row in arm["loaded_checks"]):
            raise ValueError("Loaded source byte checks failed")
        expected_keys = {(name, trial, window) for name, _, _ in CASES
                         for trial in range(6) for window in WINDOWS}
        seen = set()
        for trial in arm["trials"]:
            key = trial["case"], trial["trial"], trial["window_ms"]
            if key not in expected_keys or key in seen:
                raise ValueError("Missing, repeated or unexpected timing trial")
            seen.add(key)
            expected_case = next(c for c in CASES if c[0] == trial["case"])
            if (trial["concurrency"], trial["prompt_tokens"]) != expected_case[1:]:
                raise ValueError("Case labels and actual workload differ")
            validate_snapshot(trial["before"])
            validate_snapshot(trial["after"])
            queue = trial["admission"]
            if queue["window_ms"] != key[2] or not queue["only_idle_fresh_text_prefills"]:
                raise ValueError("Actual admission policy differs from named timing policy")
            if any(e["deadline_extended"] or e["window_ms"] != key[2] for e in queue["events"]):
                raise ValueError("Admission exceeded its fixed collection window")
            rows.append({"arm": label, "case": key[0], "trial": key[1], "window_ms": key[2],
                **timing_metrics(trial["measurement"], trial["concurrency"]),
                "input_window_events": queue["events"]})
        if seen != expected_keys:
            raise ValueError("Incomplete high-concurrency timing grid")
        for name, _, _ in CASES:
            for window in WINDOWS:
                chosen = [r for r in rows if (r["arm"], r["case"], r["window_ms"])
                          == (label, name, window)]
                summaries.append({"arm": label, "case": name, "window_ms": window, "trials": 6,
                    "median_tps": statistics.median(r["aggregate_tokens_per_second"] for r in chosen),
                    "min_tps": min(r["aggregate_tokens_per_second"] for r in chosen),
                    "max_tps": max(r["aggregate_tokens_per_second"] for r in chosen),
                    "median_ttft_seconds": statistics.median(r["median_ttft_seconds"] for r in chosen),
                    "median_request_seconds": statistics.median(r["median_request_seconds"] for r in chosen),
                    "median_request_decode_tps": statistics.median(
                        r["median_request_decode_tokens_per_second"] for r in chosen)})
    lookup = {(r["arm"], r["case"], r["window_ms"]): r for r in summaries}
    comparisons, decisions, sequence_agreement = [], [], []
    for window in WINDOWS:
        ratios, ttft_ratios = {}, {}
        for name, _, _ in CASES:
            a, b = [lookup[arm, name, window] for arm in ("control", "candidate")]
            ratios[name] = b["median_tps"] / a["median_tps"]
            ttft_ratios[name] = b["median_ttft_seconds"] / a["median_ttft_seconds"]
            comparisons.append({"case": name, "window_ms": window,
                "compression_tps_ratio": ratios[name],
                "compression_tps_change_percent": 100 * (ratios[name] - 1),
                "compression_ttft_ratio": ttft_ratios[name]})
        primary = ("c8-short", "c16-short", "c8-long", "c16-long")
        speed = (all(ratios[name] >= plan["primary_min_compression_tps_ratio"] for name in primary)
            and all(ratios[name] >= plan["control_min_compression_tps_ratio"]
                    for name in ("c1-short", "c4-short"))
            and all(value <= plan["max_compression_ttft_ratio"] for value in ttft_ratios.values()))
        decisions.append({"window_ms": window, "observed_speed_gate_passed": speed,
            "compression_tps_ratios": ratios, "compression_ttft_ratios": ttft_ratios})
    for left in arms["control"]["trials"]:
        right = next(t for t in arms["candidate"]["trials"] if
            (t["case"], t["trial"], t["window_ms"]) ==
            (left["case"], left["trial"], left["window_ms"]))
        matches = [a["token_ids"] == b["token_ids"] for a, b in zip(
            left["measurement"]["requests"], right["measurement"]["requests"], strict=True)]
        sequence_agreement.append({"case": left["case"], "trial": left["trial"],
            "window_ms": left["window_ms"], "matching_sequences": sum(matches),
            "sequences": len(matches)})
    numerical = []
    for first, second in quality_pairs():
        a, b = [json.loads((root / (name + "-decode-quality.json")).read_bytes())
                for name in (first, second)]
        value = compare(a, b, root)
        numerical.append({"reference": first, "candidate": second,
                          "passed": numerical_passed(value), **value})
    allocations = {label: sum(row["allocation"]["cuda_allocated_bytes"]
                              for row in arm["source_after"]) for label, arm in arms.items()}
    return {"status": "complete", "quality_qualified": False,
        "previous_failed_gates_preserved": True, "throughput_issue_declared_fixed": False,
        "source_scale_groups_verified": source_groups, "timed_groups": len(rows),
        "timed_requests": sum(len(t["measurement"]["requests"])
                              for arm in arms.values() for t in arm["trials"]),
        "summaries": summaries, "comparisons": comparisons, "decisions": decisions,
        "trials": rows, "timed_output_ID_agreement": sequence_agreement,
        "numerical_comparisons": numerical, "numerical_gate_passed": all(r["passed"] for r in numerical),
        "profiles_complete": all(len(arm["profiles"]) == 4 and all(
            p["status"] == "complete" for p in arm["profiles"]) for arm in arms.values()),
        "allocated_cuda_bytes_after_inference": allocations,
        "combined_cuda_saving_bytes": allocations["control"] - allocations["candidate"],
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "limits": plan["limits"]}


def main() -> None:
    """Keep strict gates unchanged and write a review after both model arms finish."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = review(args.root)
    result["analysis_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write(args.root / "quality-model-review.json", result)


if __name__ == "__main__":
    main()
