"""Bracket each routing candidate with unchanged work to control short-term timing drift."""

from __future__ import annotations

import statistics
from typing import Any

import torch

TARGETS = ("stock", "raw_dense", "fused_split1")


def measure(graphs: dict[str, Any], pressure: torch.Tensor, *, evict: bool) -> dict[str, Any]:
    """Retain eight ABBA/BAAB blocks per target, with four event samples in every leg."""
    samples = []
    for trial in range(8):
        forward = trial % 2 == 0
        targets = TARGETS if forward else tuple(reversed(TARGETS))
        blocks = {}
        for target in targets:
            order = ("qualified", target, target, "qualified") if forward else (
                target, "qualified", "qualified", target)
            legs = []
            for arm in order:
                values = []
                for _ in range(4):
                    if evict:
                        pressure.add_(1)
                    first = torch.cuda.Event(enable_timing=True)
                    last = torch.cuda.Event(enable_timing=True)
                    first.record()
                    graphs[arm].replay()
                    last.record()
                    last.synchronize()
                    values.append(first.elapsed_time(last) * 1000)
                legs.append({"arm": arm, "times_us": values})
            reference = [v for leg in legs if leg["arm"] == "qualified" for v in leg["times_us"]]
            candidate = [v for leg in legs if leg["arm"] == target for v in leg["times_us"]]
            blocks[target] = {"order": list(order), "legs": legs,
                "reference_median_us": statistics.median(reference),
                "candidate_median_us": statistics.median(candidate),
                "ratio": statistics.median(candidate) / statistics.median(reference)}
        samples.append({"trial": trial, "direction": "ABBA" if forward else "BAAB",
                        "target_order": list(targets), "blocks": blocks})
    result = {"protocol": "bracketed_ABBA_BAAB", "trials": 8, "iterations_per_leg": 4,
        "pressure_bytes": pressure.numel() * pressure.element_size() if evict else 0,
        "samples": samples,
        "reference_median_us": {target: statistics.median(sample["blocks"][target][
            "reference_median_us"] for sample in samples) for target in TARGETS},
        "median_us": {target: statistics.median(sample["blocks"][target]["candidate_median_us"]
            for sample in samples) for target in TARGETS},
        "median_paired_ratio": {target: statistics.median(sample["blocks"][target]["ratio"]
            for sample in samples) for target in TARGETS},
        "order_ratios": {order: {target: statistics.median(sample["blocks"][target]["ratio"]
            for sample in samples if sample["direction"] == order) for target in TARGETS}
            for order in ("ABBA", "BAAB")}}
    # The displayed global reference is descriptive; each decision uses its
    # own before/after baseline from the matching candidate block above.
    result["median_us"]["qualified"] = statistics.median(
        statistics.median(block["reference_median_us"] for block in sample["blocks"].values())
        for sample in samples)
    return result
