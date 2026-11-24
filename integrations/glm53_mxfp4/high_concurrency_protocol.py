"""Frozen matched C8/C16 compression experiment, with smaller concurrency controls."""

from __future__ import annotations

from typing import Any

CAPTURES = [1, 2, 4, 8, 16]
CASES = (("c1-short", 1, 256), ("c4-short", 4, 256), ("c8-short", 8, 256),
         ("c16-short", 16, 256), ("c8-long", 8, 2048), ("c16-long", 16, 2048))
WINDOWS = (0, 3)


def validate_plan(plan: dict[str, Any]) -> None:
    """Reject changed workload, precision, capacity, source or numerical-gate scope."""
    fixed = {"version": 1, "cases": [list(c) for c in CASES], "windows_ms": list(WINDOWS),
        "trials": 6, "generated_tokens": 64, "warmups": 2, "warmup_tokens": 16,
        "capture_sizes": CAPTURES, "max_num_seqs": 16, "max_batched_tokens": 2048,
        "KV_bytes_per_rank": 4294967296, "max_model_len": 8192, "model_loads": 2,
        "arms_in_order": ["control", "candidate"], "speculative_decoding": False,
        "KDA_fused_max_tokens": 8, "quality_concurrencies": [8, 16],
        "quality_repeats": 2, "mean_decode_kl_floor": 1e-5,
        "abs_mean_decode_nll_floor": 1e-4, "primary_min_compression_tps_ratio": 1.05,
        "control_min_compression_tps_ratio": .98, "max_compression_ttft_ratio": 1.05,
        "quality_qualified": False, "previous_failed_gates_preserved": True}
    if any(plan.get(key) != value for key, value in fixed.items()):
        raise ValueError("Frozen C8/C16 compression protocol changed")


def validate_snapshot(rows: list[dict[str, Any]]) -> None:
    """Require both actual workers, exact C16 replay and unchanged non-speculative limits."""
    if len(rows) != 2 or {r["rank"] for r in rows} != {0, 1}:
        raise ValueError("Require both high-concurrency worker ranks")
    for row in rows:
        if (row["physical_capture_sizes"] != CAPTURES or not row["graphs_enabled"]
                or row["max_num_seqs"] != 16 or row["max_batched_tokens"] != 2048
                or row["KV_bytes_per_rank"] != 4294967296
                or row["max_model_len"] != 8192 or not row["speculative_config_absent"]):
            raise ValueError("Actual graph, sequence, KV or drafter scope changed")
        for size in CAPTURES:
            descriptor = row["descriptors"][str(size)]
            if not descriptor["mode"].endswith("FULL") or descriptor["num_tokens"] != size:
                raise ValueError("Require exact graph dispatch, including sixteen tokens")
