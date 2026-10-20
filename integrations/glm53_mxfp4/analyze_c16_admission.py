"""Recompute focused 3/10 ms C16 admission and keep compression advantage separate."""

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
from c16_admission_protocol import (
    ARMS,
    CASES,
    WINDOWS,
    policy_decisions,
    quality_labels,
    quality_pairs,
    timing_schedule,
    validate_plan,
)
from decode_quality import compare
from high_concurrency_protocol import validate_snapshot
from measure_c8_trace import passed
from measure_c16_admission import model_signature
from startup_preparation import validate_startup_report


def review(root: Path) -> dict[str, Any]:
    """Require measured cohort equality and preserve all failed numerical/speed decisions."""
    plan_path = root / "c16-admission-plan.json"
    plan = json.loads(plan_path.read_bytes())
    validate_plan(plan)
    arms = {}
    groups = {}
    summaries = []
    rows = []
    source_groups = 0
    for arm in ("control", "candidate"):
        data = json.loads((root / f"{arm}-c16-admission.json").read_bytes())
        assert data["status"] == "complete" and len(data["quality"]) == 10
        assert len(data["timing"]) == 36 and len(data["warmups"]) == 8
        assert len(data["profiles"]) == 2
        assert [p["window_ms"] for p in data["profiles"]] == list(WINDOWS)
        assert all(p["admission"]["window_ms"] == p["window_ms"] for p in data["profiles"])
        assert all(
            p["concurrency"] == 16
            and p["status"] == "complete"
            and p["after_all_unprofiled_timing"]
            for p in data["profiles"]
        )
        assert data["plan_sha256"] == hashlib.sha256(plan_path.read_bytes()).hexdigest()
        assert not data["controlled_admission_used_for_speed"]
        assert all(p["native_mhc_unchanged"] for p in data["native_mhc_policy"])
        assert all(p["native_mhc_unchanged"] for p in data["native_mhc_policy_after"])
        startup = data["dispatch_startup"]
        assert len(startup) == 2 and {r["rank"] for r in startup} == {0, 1}
        baseline = {r["rank"]: r["cache_after_trim_probe"] for r in startup}
        for row in startup:
            assert row["status"] == "complete"
            assert validate_startup_report(row) == row["checks"]
            assert not row["model_quality_or_serving_speed_qualified"]
        cache_groups = [
            data["startup_cache_after_quality"],
            data["startup_cache_after_profiles"],
            *[t["dispatch_cache_after"] for t in data["timing"]],
        ]
        for cached in cache_groups:
            assert len(cached) == 2 and {r["rank"] for r in cached} == {0, 1}
            assert all(r["cache_unchanged"] and r["cache"] == baseline[r["rank"]] for r in cached)
        for stage in ("snapshot_before", "snapshot_before_profiles", "snapshot_after"):
            validate_snapshot(data[stage])
            for sample in data[stage]:
                assert not sample.get("fp32_input_policy", {}).get("enabled", False)
                policy = sample["focused_row16_policy"]
                assert policy["compressed"] == (arm == "candidate")
                assert policy["global_fused_limit"] == 8
                assert policy["resident_BF16_weight_cache_bytes"] == 0
                assert policy["input_exact_rows"] == ([16] if arm == "candidate" else [])
                assert policy["output_exact_rows"] == ([16] if arm == "candidate" else [])
                assert policy["component_evidence_sha256"] == (
                    "76b00c670bf2747448afe35ca9e7f011747b5c3d17ac0b6fca9caa930936b34c"
                )
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
            assert g["controlled"] == (g["concurrency"] in (8, 16))
            assert g["window_ms"] in WINDOWS
            assert g["admission"]["window_ms"] == g["window_ms"]
            assert g["label"] in quality_labels(arm, g["concurrency"])
            groups[g["label"]] = g
        arms[arm] = data
        assert [
            {k: t[k] for k in timing_schedule()[0]} for t in data["timing"]
        ] == timing_schedule()
        for t in data["timing"]:
            assert t["admission"]["window_ms"] == t["window_ms"]
            assert len(t["measurement"]["requests"]) == t["concurrency"]
            assert len(t["telemetry_before"]) == len(t["telemetry_after"]) == 2
            for request in t["measurement"]["requests"]:
                assert len(request["token_ids"]) == 64
                assert request["usage"]["prompt_tokens"] == 256
                assert request["usage"]["completion_tokens"] == 64
                assert (
                    0 <= request["request_start_offset_seconds"] < t["measurement"]["wall_seconds"]
                )
            rows.append(
                {
                    "arm": arm,
                    "case": t["case"],
                    "repeat": t["repeat"],
                    "window_ms": t["window_ms"],
                    **timing_metrics(t["measurement"], t["concurrency"]),
                }
            )
        for name, _, _, repeats in CASES:
            for window in WINDOWS:
                chosen = [
                    r for r in rows if (r["arm"], r["case"], r["window_ms"]) == (arm, name, window)
                ]
                assert len(chosen) == repeats
                summaries.append(
                    {
                        "arm": arm,
                        "case": name,
                        "window_ms": window,
                        "trials": repeats,
                        "median_tps": statistics.median(
                            r["aggregate_tokens_per_second"] for r in chosen
                        ),
                        "median_ttft_seconds": statistics.median(
                            r["median_ttft_seconds"] for r in chosen
                        ),
                        "minimum_tps": min(r["aggregate_tokens_per_second"] for r in chosen),
                        "maximum_tps": max(r["aggregate_tokens_per_second"] for r in chosen),
                    }
                )
    assert set(groups) == {
        label for arm in ARMS for count in (1, 16) for label in quality_labels(arm, count)
    }
    numerical = []
    for left, right in quality_pairs():
        a, b = [
            json.loads((root / f"{name}-decode-quality.json").read_bytes())
            for name in (left, right)
        ]
        result = compare(a, b, root)
        same = model_signature(groups[left]) == model_signature(groups[right])
        exact = {
            kind: sum(
                x["sha256"] == y["sha256"]
                for c, d in zip(a["contexts"], b["contexts"], strict=True)
                for x, y in zip(c["steps"], d["steps"], strict=True)
                if x["kind"] == kind
            )
            for kind in ("prefill", "decode")
        }
        numerical.append(
            {
                "reference": left,
                "candidate": right,
                **result,
                "actual_model_grouping_equal": same,
                "numerical_gate_passed": passed(result),
                "passed": same and passed(result),
                "bitwise_equal_full_vocabulary_rows": exact,
            }
        )
    timed = {
        (a, t["case"], t["repeat"], t["window_ms"]): t
        for a, data in arms.items()
        for t in data["timing"]
    }
    timed_pairs = []
    for case, count, _, repeats in CASES:
        for repeat in range(repeats):
            for left, right in itertools.combinations(itertools.product(ARMS, WINDOWS), 2):
                a, b = (
                    timed[arm, case, repeat, window]["measurement"]["requests"]
                    for arm, window in (left, right)
                )
                assert len(a) == len(b) == count
                for index, (first, second) in enumerate(zip(a, b, strict=True)):
                    timed_pairs.append(
                        {
                            "case": case,
                            "repeat": repeat,
                            "request": index,
                            "left_arm_window": list(left),
                            "right_arm_window": list(right),
                            "token_ids_match": first["token_ids"] == second["token_ids"],
                        }
                    )
    assert len(timed_pairs) == 1188
    all_timed_ids_equal = all(x["token_ids_match"] for x in timed_pairs)
    numerical_passed = all(r["passed"] for r in numerical)
    decisions = policy_decisions(
        summaries, numerical_passed=numerical_passed, timed_ids_match=all_timed_ids_equal
    )
    allocations = {
        arm: sum(r["cuda_allocated_bytes"] for r in data["snapshot_before_profiles"])
        for arm, data in arms.items()
    }
    return {
        "status": "complete",
        "quality_qualified": False,
        "compression_numerical_gate_passed": numerical_passed,
        **decisions,
        "full_workload_speed_qualified": False,
        "parent_model_archive_sha256": plan["parent_model_archive_sha256"],
        "all_quality_groupings_matched": all(r["actual_model_grouping_equal"] for r in numerical),
        "timed_token_sequence_pairs": len(timed_pairs),
        "all_timed_generated_ids_match": all_timed_ids_equal,
        "timed_token_mismatches": [x for x in timed_pairs if not x["token_ids_match"]],
        "C16_matched_numerical_gate_passed": all(
            x["passed"] for x in numerical if "-c16-" in x["reference"]
        ),
        "automatic_promotion_allowed": False,
        "MTP_not_part_of_this_experiment": True,
        "previous_failures_preserved": True,
        "general_serving_batch_invariance_qualified": False,
        "source_scale_groups_verified": source_groups,
        "numerical_comparisons": numerical,
        "summaries": summaries,
        "timed_groups": len(rows),
        "timed_requests": sum(
            len(t["measurement"]["requests"]) for d in arms.values() for t in d["timing"]
        ),
        "allocated_cuda_bytes_before_profiles": allocations,
        "combined_cuda_saving_bytes": allocations["control"] - allocations["candidate"],
        "dispatch_startup_preparation_passed": True,
        "dispatch_cache_unchanged_through_model_work": True,
        "startup_resource_observations": {
            arm: data["dispatch_startup"] for arm, data in arms.items()
        },
        "CPU_memory_observations": {
            arm: {key: data[key] for key in ("telemetry_before", "telemetry_after")}
            for arm, data in arms.items()
        },
        "limits": [
            "Quality applies to the retained contexts and actual matched grouping.",
            "Two fresh model loads have fixed stock-then-compressed order and a time gap.",
            "Ordinary API timing excludes controlled pause/resume, full logprobs and profiling.",
            "Process/container memory and CUDA allocations overlap on Spark; do not add them.",
            "Only short-C1/C16 policy speed is measured here; C8/long-prompt results are not requalified.",
            "Quality covers finite matched C1/C16 contexts at both windows; broader quality remains unmeasured.",
        ],
    }


def main() -> None:
    """Write a reproducible review without treating finite gates as a release approval."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    result = review(args.root)
    write(args.root / "quality-model-review.json", result)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("numerical_comparisons", "CPU_memory_observations")
            }
        )
    )


if __name__ == "__main__":
    main()
