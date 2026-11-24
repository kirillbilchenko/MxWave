"""Prospective, bounded C16 admission comparison on a separately qualified model parent."""

from __future__ import annotations

import itertools
import re
from typing import Any

ARMS = ("control", "candidate")
WINDOWS = (3, 10)
CASES = (("c16-short", 16, 256, 12), ("c1-short", 1, 256, 6))
NUMERICAL_LIMITS = {"mean_decode_KL": 1e-5, "absolute_mean_NLL": 1e-4}
POLICY_LIMITS = {
    "candidate_C16_TTFT_ratio_max": 0.95,
    "control_C16_TTFT_ratio_max": 1.05,
    "all_case_TPS_ratio_min": 0.98,
    "C1_TTFT_increase_seconds_max": 0.015,
}


def timing_schedule() -> list[dict[str, Any]]:
    """Balance policy order within every workload; preserve all twelve primary repeats."""
    rows = []
    for repeat in range(12):
        for name, count, length, repeats in CASES if repeat % 2 == 0 else CASES[::-1]:
            if repeat >= repeats:
                continue
            for window in WINDOWS if repeat % 2 == 0 else WINDOWS[::-1]:
                rows.append(
                    {
                        "case": name,
                        "concurrency": count,
                        "prompt_tokens": length,
                        "repeat": repeat,
                        "window_ms": window,
                    }
                )
    return rows


def quality_labels(arm: str, count: int) -> list[str]:
    """Name every repeat and separately profiled quality cohort in the declared scope."""
    if arm not in ARMS or count not in (1, 16):
        raise ValueError("Undeclared admission quality arm or concurrency")
    labels = [f"{arm}-c{count}-w{w}-r{r}" for w in WINDOWS for r in range(2)]
    if count == 16:
        labels.extend(f"{arm}-c16-w{w}-profile" for w in WINDOWS)
    return labels


def quality_pairs() -> list[tuple[str, str]]:
    """Compare every same-concurrency arm/window/repeat/profile combination, without selection."""
    return [
        pair
        for count in (1, 16)
        for pair in itertools.combinations(
            [label for arm in ARMS for label in quality_labels(arm, count)], 2
        )
    ]


def plan(
    source_protocol_sha256: str, parent_archive_sha256: str, *, input_fp32: bool
) -> dict[str, Any]:
    """Bind prospective policy gates and unchanged capacity to one verified model parent."""
    result = {
        "source_protocol_sha256": source_protocol_sha256,
        "parent_model_archive_sha256": parent_archive_sha256,
        "candidate_input_fp32": input_fp32,
        "windows_ms": list(WINDOWS),
        "cases": [list(c) for c in CASES],
        "timing_schedule": timing_schedule(),
        "generated_tokens": 64,
        "warmups_per_case_window": 2,
        "warmup_tokens": 16,
        "quality_repeats_per_window": 2,
        "quality_concurrencies": [1, 16],
        "quality_pairs": [list(p) for p in quality_pairs()],
        "numerical_limits": dict(NUMERICAL_LIMITS),
        "policy_limits": dict(POLICY_LIMITS),
        "compression_speed_limits": {"TPS_ratio_min": 1.05, "TTFT_ratio_max": 1.05},
        "capture_sizes": [1, 2, 4, 8, 16],
        "max_num_seqs": 16,
        "max_batched_tokens": 2048,
        "KV_bytes_per_rank": 4294967296,
        "MTP_deferred": True,
        "automatic_promotion_allowed": False,
    }
    validate_plan(result)
    return result


def validate_plan(value: dict[str, Any]) -> None:
    """Reject omitted cases, weakened gates, altered order or an unbound parent."""
    for key in ("source_protocol_sha256", "parent_model_archive_sha256"):
        if not isinstance(value[key], str) or re.fullmatch(r"[0-9a-f]{64}", value[key]) is None:
            raise ValueError("Require full immutable protocol and model-parent hashes")
    if value["candidate_input_fp32"] is not False:
        raise ValueError("Use the qualified warmed exact16 parent, without input FP32")
    expected = {
        "windows_ms": list(WINDOWS),
        "cases": [list(c) for c in CASES],
        "timing_schedule": timing_schedule(),
        "generated_tokens": 64,
        "warmups_per_case_window": 2,
        "warmup_tokens": 16,
        "quality_repeats_per_window": 2,
        "quality_concurrencies": [1, 16],
        "quality_pairs": [list(p) for p in quality_pairs()],
        "numerical_limits": NUMERICAL_LIMITS,
        "policy_limits": POLICY_LIMITS,
        "compression_speed_limits": {"TPS_ratio_min": 1.05, "TTFT_ratio_max": 1.05},
        "capture_sizes": [1, 2, 4, 8, 16],
        "max_num_seqs": 16,
        "max_batched_tokens": 2048,
        "KV_bytes_per_rank": 4294967296,
        "MTP_deferred": True,
        "automatic_promotion_allowed": False,
    }
    for key, required in expected.items():
        if value.get(key) != required:
            raise ValueError(f"Frozen C16 admission protocol changed: {key}")


def policy_decisions(
    summaries: list[dict[str, Any]], *, numerical_passed: bool, timed_ids_match: bool
) -> dict[str, Any]:
    """Keep policy benefit separate from the unchanged compression advantage requirements."""
    expected = {(a, case[0], w) for a in ARMS for case in CASES for w in WINDOWS}
    lookup = {(r["arm"], r["case"], r["window_ms"]): r for r in summaries}
    if set(lookup) != expected or len(summaries) != len(expected):
        raise ValueError("Missing or duplicate policy summary")
    policies, compression = [], []
    for arm in ARMS:
        for case, _, _, _ in CASES:
            a, b = (lookup[arm, case, w] for w in WINDOWS)
            policies.append(
                {
                    "arm": arm,
                    "case": case,
                    "TPS_ratio": b["median_tps"] / a["median_tps"],
                    "TTFT_ratio": b["median_ttft_seconds"] / a["median_ttft_seconds"],
                    "TTFT_increase_seconds": b["median_ttft_seconds"] - a["median_ttft_seconds"],
                }
            )
    for window in WINDOWS:
        for case, _, _, _ in CASES:
            a, b = (lookup[arm, case, window] for arm in ARMS)
            compression.append(
                {
                    "window_ms": window,
                    "case": case,
                    "TPS_ratio": b["median_tps"] / a["median_tps"],
                    "TTFT_ratio": b["median_ttft_seconds"] / a["median_ttft_seconds"],
                "TTFT_increase_seconds": b["median_ttft_seconds"] - a["median_ttft_seconds"],
                }
            )
    checks = {
        "quality": numerical_passed,
        "timed_outputs": timed_ids_match,
        "throughput_nonregression": all(p["TPS_ratio"] >= 0.98 for p in policies),
        "C1_delay_guard": all(
            p["TTFT_increase_seconds"] <= 0.015 for p in policies if p["case"] == "c1-short"
        ),
        "candidate_C16_TTFT_improves": next(
            p["TTFT_ratio"] for p in policies if (p["arm"], p["case"]) == ("candidate", "c16-short")
        )
        <= 0.95,
        "stock_C16_TTFT_nonregression": next(
            p["TTFT_ratio"] for p in policies if (p["arm"], p["case"]) == ("control", "c16-short")
        )
        <= 1.05,
    }
    return {
        "policy_comparisons": policies,
        "policy_checks": checks,
        "policy_gate_passed": all(checks.values()),
        "compression_comparisons": compression,
        "compression_speed_gate_by_window": {
            str(w): numerical_passed
            and timed_ids_match
            and all(
                p["TPS_ratio"] >= 1.05 and p["TTFT_ratio"] <= 1.05
                for p in compression
                if p["window_ms"] == w
            )
            for w in WINDOWS
        },
    }
