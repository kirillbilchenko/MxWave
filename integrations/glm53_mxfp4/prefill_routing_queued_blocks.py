"""Queue complete timing legs before synchronization to reduce host gaps between samples."""

from __future__ import annotations

import statistics
from typing import Any

import torch

TARGETS = ("stock", "raw_dense", "fused_split1")
ITERATIONS = 8


def leg(operation: Any, pressure: torch.Tensor, *, evict: bool) -> dict[str, Any]:
    """Prime once, enqueue every event pair, then synchronize after the final recorded event."""
    if evict:
        pressure.add_(1)
    operation.replay()
    pairs = []
    for _ in range(ITERATIONS):
        if evict:
            pressure.add_(1)
        first = torch.cuda.Event(enable_timing=True)
        last = torch.cuda.Event(enable_timing=True)
        first.record()
        operation.replay()
        last.record()
        pairs.append((first, last))
    pairs[-1][1].synchronize()
    return {"times_us": [first.elapsed_time(last) * 1000 for first, last in pairs],
            "untimed_primer_calls": 1, "synchronizations_after_enqueue": 1,
            "all_event_pairs_queued_before_synchronization": True}


def measure(graphs: dict[str, Any], pressure: torch.Tensor, *, evict: bool) -> dict[str, Any]:
    """Keep eight balanced bracketed blocks, every timed sample and unchanged target controls."""
    if graphs["stock"] is not graphs["qualified"]:
        raise ValueError("Queued control must replay the identical reference callable or graph")
    samples = []
    for trial in range(8):
        forward = trial % 2 == 0
        targets = TARGETS if forward else tuple(reversed(TARGETS))
        blocks = {}
        for target in targets:
            order = ("qualified", target, target, "qualified") if forward else (
                target, "qualified", "qualified", target)
            legs = [{"arm": arm, **leg(graphs[arm], pressure, evict=evict)} for arm in order]
            reference = [v for item in legs if item["arm"] == "qualified" for v in item["times_us"]]
            candidate = [v for item in legs if item["arm"] == target for v in item["times_us"]]
            blocks[target] = {"order": list(order), "legs": legs,
                "reference_median_us": statistics.median(reference),
                "candidate_median_us": statistics.median(candidate),
                "ratio": statistics.median(candidate) / statistics.median(reference)}
        samples.append({"trial": trial, "direction": "ABBA" if forward else "BAAB",
                        "target_order": list(targets), "blocks": blocks})
    result = {"protocol": "queued_bracketed_ABBA_BAAB", "trials": 8,
        "iterations_per_leg": ITERATIONS, "untimed_primer_calls_per_leg": 1,
        "pressure_before_each_timed_event_pair": evict,
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
    result["median_us"]["qualified"] = statistics.median(
        statistics.median(block["reference_median_us"] for block in sample["blocks"].values())
        for sample in samples)
    return result
