"""Recompute strict matched-cohort quality and descriptive fresh-pair API measurements."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import statistics
from pathlib import Path
from typing import Any

from analyze_paired_speed import timing_metrics
from benchmark_full_model import write
from decode_quality import compare
from measure_c8_trace import passed
from measure_matched_c8 import model_signature


def quality_pairs() -> list[tuple[str, str]]:
    """Freeze every self-repeat, cross-arm and profile comparison without result selection."""
    pairs = []
    for count in (1, 8):
        for arm in ("control", "candidate"):
            pairs.extend((f"{arm}-c{count}-r0", f"{arm}-c{count}-r{i}") for i in (1, 2))
        pairs.extend(
            (f"control-c{count}-r{i}", f"candidate-c{count}-r{j}")
            for i, j in itertools.product(range(3), repeat=2)
        )
    for arm in ("control", "candidate"):
        pairs.extend((f"{arm}-c8-r{i}", f"{arm}-c8-profile") for i in range(3))
    pairs.append(("control-c8-profile", "candidate-c8-profile"))
    assert len(pairs) == len(set(pairs)) == 33
    return pairs


def review(root: Path) -> dict[str, Any]:
    """Require measured cohort equality and preserve all failed numerical/speed decisions."""
    plan_path = root / "matched-c8-plan.json"
    plan = json.loads(plan_path.read_bytes())
    assert plan["numerical_limits"] == {"mean_decode_KL": 1e-5, "absolute_mean_NLL": 1e-4}
    assert plan["quality_pairs"] == [list(p) for p in quality_pairs()]
    arms = {}
    groups = {}
    summaries = []
    rows = []
    source_groups = 0
    for arm in ("control", "candidate"):
        data = json.loads((root / f"{arm}-matched-c8.json").read_bytes())
        assert data["status"] == "complete" and len(data["quality"]) == 7
        assert len(data["timing"]) == 18 and len(data["warmups"]) == 6
        assert len(data["profiles"]) == 1
        assert data["plan_sha256"] == hashlib.sha256(plan_path.read_bytes()).hexdigest()
        assert not data["controlled_admission_used_for_speed"]
        assert all(p["native_mhc_unchanged"] for p in data["native_mhc_policy"])
        assert all(p["native_mhc_unchanged"] for p in data["native_mhc_policy_after"])
        for before, after in zip(data["source_before"], data["source_after"], strict=True):
            assert before["status"] == after["status"] == "passed"
            assert before["candidate"] == after["candidate"] == (arm == "candidate")
            assert before["source_scale_hashes"] == after["source_scale_hashes"]
            assert len(before["source_scale_hashes"]) == 84
            source_groups += 2 * len(before["source_scale_hashes"])
        assert len(data["source_before"]) == len(data["source_after"]) == 2
        for g in data["quality"]:
            assert g["status"] == "complete" and model_signature(g, 0) == model_signature(g, 1)
            assert g["label"] not in groups
            assert g["controlled"] == (g["concurrency"] == 8)
            groups[g["label"]] = g
        arms[arm] = data
        seen = set()
        for t in data["timing"]:
            key = t["case"], t["repeat"]
            assert key not in seen and 0 <= t["repeat"] < 6
            seen.add(key)
            assert [t["case"], t["concurrency"], t["prompt_tokens"]] in plan["timing_cases"]
            assert t["admission"]["window_ms"] == 3
            assert len(t["measurement"]["requests"]) == t["concurrency"]
            assert all(len(r["token_ids"]) == 64 for r in t["measurement"]["requests"])
            assert len(t["telemetry_before"]) == len(t["telemetry_after"]) == 2
            rows.append({"arm": arm, "case": t["case"], "repeat": t["repeat"],
                         **timing_metrics(t["measurement"], t["concurrency"])})
        assert len(seen) == 18
        for name, _, _ in plan["timing_cases"]:
            chosen = [r for r in rows if (r["arm"], r["case"]) == (arm, name)]
            assert len(chosen) == 6
            summaries.append({
                "arm": arm, "case": name, "trials": 6,
                "median_tps": statistics.median(r["aggregate_tokens_per_second"] for r in chosen),
                "median_ttft_seconds": statistics.median(r["median_ttft_seconds"] for r in chosen),
                "minimum_tps": min(r["aggregate_tokens_per_second"] for r in chosen),
                "maximum_tps": max(r["aggregate_tokens_per_second"] for r in chosen),
            })
    numerical = []
    for left, right in quality_pairs():
        a, b = [json.loads((root / f"{name}-decode-quality.json").read_bytes())
                for name in (left, right)]
        result = compare(a, b, root)
        same = model_signature(groups[left]) == model_signature(groups[right])
        exact = {
            kind: sum(x["sha256"] == y["sha256"]
                      for c, d in zip(a["contexts"], b["contexts"], strict=True)
                      for x, y in zip(c["steps"], d["steps"], strict=True)
                      if x["kind"] == kind)
            for kind in ("prefill", "decode")
        }
        numerical.append({
            "reference": left, "candidate": right, **result,
            "actual_model_grouping_equal": same, "numerical_gate_passed": passed(result),
            "passed": same and passed(result), "bitwise_equal_full_vocabulary_rows": exact,
        })
    lookup = {(r["arm"], r["case"]): r for r in summaries}
    speed = []
    for name, _, _ in plan["timing_cases"]:
        a, b = [lookup[arm, name] for arm in ("control", "candidate")]
        speed.append({
            "case": name, "compression_tps_ratio": b["median_tps"] / a["median_tps"],
            "compression_ttft_ratio": b["median_ttft_seconds"] / a["median_ttft_seconds"],
        })
    allocations = {
        arm: sum(r["cuda_allocated_bytes"] for r in data["snapshot_before_profiles"])
        for arm, data in arms.items()
    }
    return {
        "status": "complete", "quality_qualified": False,
        "compression_numerical_gate_passed": all(r["passed"] for r in numerical),
        "all_quality_groupings_matched": all(r["actual_model_grouping_equal"] for r in numerical),
        "observed_speed_gate_passed": all(
            r["compression_tps_ratio"] >= (1.05 if r["case"].startswith("c8") else 1.0)
            and r["compression_ttft_ratio"] <= 1.05 for r in speed
        ),
        "automatic_promotion_allowed": False, "MTP_deferred_by_user": True,
        "previous_failures_preserved": True, "general_serving_batch_invariance_qualified": False,
        "source_scale_groups_verified": source_groups, "numerical_comparisons": numerical,
        "summaries": summaries, "speed_comparisons": speed, "timed_groups": len(rows),
        "timed_requests": sum(len(t["measurement"]["requests"])
                              for d in arms.values() for t in d["timing"]),
        "allocated_cuda_bytes_before_profiles": allocations,
        "combined_cuda_saving_bytes": allocations["control"] - allocations["candidate"],
        "CPU_memory_observations": {
            arm: {key: data[key] for key in ("telemetry_before", "telemetry_after")}
            for arm, data in arms.items()
        },
        "limits": [
            "Quality applies to the retained contexts and actual matched grouping.",
            "Two fresh model loads have fixed stock-then-compressed order and a time gap.",
            "Ordinary API timing excludes controlled pause/resume, full logprobs and profiling.",
            "Process/container memory and CUDA allocations overlap on Spark; do not add them.",
            "C16 input/output integration and broader quality are outside this run.",
        ],
    }


def main() -> None:
    """Write a reproducible review without treating finite gates as a release approval."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    result = review(args.root)
    write(args.root / "quality-model-review.json", result)
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("numerical_comparisons", "CPU_memory_observations")}))


if __name__ == "__main__":
    main()
