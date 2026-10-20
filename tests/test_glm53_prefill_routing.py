"""Reject routing recommendations that hide order regressions or omit planned shapes."""

from __future__ import annotations

import copy
import statistics
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))
from verify_prefill_routing import ARMS, checked_blocks, verify


def _fixture(reverse_fused_us: float) -> tuple[dict[str, Any], dict[str, Any]]:
    samples = []
    for trial in range(8):
        forward = trial % 2 == 0
        times = {"qualified": [100.] * 4, "stock": [100.] * 4, "raw_dense": [90.] * 4,
                 "fused_split1": [80. if forward else reverse_fused_us] * 4}
        samples.append({"trial": trial, "direction": "forward" if forward else "reverse",
            "order": list(ARMS if forward else reversed(ARMS)), "times_us": times,
            "ratios": {arm: values[0] / 100 for arm, values in times.items()}})
    measured = {"trials": 8, "iterations": 4, "samples": samples, "pressure_bytes": 0,
        "median_us": {arm: statistics.median(row["times_us"][arm][0] for row in samples)
                      for arm in ARMS},
        "median_paired_ratio": {arm: statistics.median(row["ratios"][arm] for row in samples)
                                for arm in ARMS},
        "order_ratios": {order: {arm: statistics.median(row["ratios"][arm] for row in samples
            if row["direction"] == order) for arm in ARMS} for order in ("forward", "reverse")}}
    pressure = copy.deepcopy(measured)
    pressure["pressure_bytes"] = 128 * 1024**2
    row = {"tokens": 16, "native_padded_tokens": 16,
        "arithmetic": {"finite": True, "relative_l2": .005, "passed": True},
        "stock_outputs_bits_exact": True, "fused_eager_graph_bits_exact": True,
        "component_eligible": True,
        "timing": {execution: {"warm": copy.deepcopy(measured), "pressure": copy.deepcopy(pressure)}
                   for execution in ("eager", "graph")}}
    cases = [{"name": name, "weight_bits_exact": True, "destination_guards_unchanged": True,
              "measurements": [copy.deepcopy(row)]} for name in ("input", "output")]
    plan = {"tokens": [16], "cases": ["input", "output"],
            "timing_cuda_allocator_cap_bytes": 1024**3}
    report = {"status": "complete", "tokens": [16], "arms": list(ARMS), "cases": cases,
        "peak_cuda_bytes": 1024, "full_model_loaded": False, "qualification_attempted": False,
        "serving_adapter_changed": False, "serving_routing_qualified": False,
        "component_eligible_tested_rows": [16], "correctness_passed": True}
    return report, plan


def test_routing_verifier_rejects_a_win_hidden_by_combined_order() -> None:
    report, plan = _fixture(105.)
    with pytest.raises(AssertionError):
        verify(report, plan)


def test_routing_verifier_requires_the_complete_planned_sweep() -> None:
    report, plan = _fixture(80.)
    plan["tokens"].append(2048)
    report["tokens"].append(2048)
    with pytest.raises(AssertionError):
        verify(report, plan)


def test_component_lead_does_not_qualify_serving_routing() -> None:
    report, plan = _fixture(80.)
    result = verify(report, plan)
    assert result["shared_eligible_tested_rows"] == [16]
    assert result["event_samples"] == 1024
    assert result["serving_routing_qualified"] is False


def _blocks(reverse_fused_us: float = 80.) -> dict[str, Any]:
    samples = []
    targets = ARMS[1:]
    for index in range(8):
        forward = index % 2 == 0
        blocks = {}
        for target in targets:
            candidate = {"stock": 100., "raw_dense": 90.,
                         "fused_split1": 80. if forward else reverse_fused_us}[target]
            order = ["qualified", target, target, "qualified"] if forward else [
                target, "qualified", "qualified", target]
            blocks[target] = {"order": order, "reference_median_us": 100.,
                "candidate_median_us": candidate, "ratio": candidate / 100,
                "legs": [{"arm": arm, "times_us": [100. if arm == "qualified" else candidate] * 4}
                         for arm in order]}
        samples.append({"trial": index, "direction": "ABBA" if forward else "BAAB",
            "target_order": list(targets if forward else reversed(targets)), "blocks": blocks})
    return {"protocol": "bracketed_ABBA_BAAB", "trials": 8, "iterations_per_leg": 4,
        "pressure_bytes": 0, "samples": samples,
        "reference_median_us": dict.fromkeys(targets, 100.),
        "median_us": {"qualified": 100., "stock": 100., "raw_dense": 90.,
                      "fused_split1": (80. + reverse_fused_us) / 2},
        "median_paired_ratio": {"stock": 1., "raw_dense": .9,
                                "fused_split1": (.8 + reverse_fused_us / 100) / 2},
        "order_ratios": {"ABBA": {"stock": 1., "raw_dense": .9, "fused_split1": .8},
                         "BAAB": {"stock": 1., "raw_dense": .9,
                                  "fused_split1": reverse_fused_us / 100}}}


def test_bracketed_control_recomputes_both_sides_and_keeps_bad_order() -> None:
    assert checked_blocks(_blocks(105.))["fused_split1"] == [.925, .8, 1.05]
    damaged = _blocks()
    damaged["samples"][0]["blocks"]["fused_split1"]["legs"].pop()
    with pytest.raises(AssertionError):
        checked_blocks(damaged)


def test_bracketed_complete_sweep_counts_every_control_and_candidate_leg() -> None:
    report, plan = _fixture(80.)
    report.update(paired_blocks=True, same_reference_graph_control=True)
    plan["paired_blocks"] = True
    for case in report["cases"]:
        row = case["measurements"][0]
        for groups in row["timing"].values():
            for cache in groups:
                groups[cache] = _blocks()
                groups[cache]["pressure_bytes"] = 128 * 1024**2 if cache == "pressure" else 0
    result = verify(report, plan)
    assert result["event_samples"] == 3072
    assert result["shared_eligible_tested_rows"] == [16]


def test_queued_verifier_keeps_bad_order_and_requires_enqueue_and_primer_evidence() -> None:
    row = _blocks(105.)
    row.update(protocol="queued_bracketed_ABBA_BAAB", iterations_per_leg=8,
               untimed_primer_calls_per_leg=1, pressure_before_each_timed_event_pair=False)
    for sample in row["samples"]:
        for block in sample["blocks"].values():
            for leg in block["legs"]:
                leg["times_us"] *= 2
                leg.update(untimed_primer_calls=1, synchronizations_after_enqueue=1,
                           all_event_pairs_queued_before_synchronization=True)
    assert checked_blocks(row, queued=True)["fused_split1"] == [.925, .8, 1.05]
    row["samples"][0]["blocks"]["stock"]["legs"][0][
        "all_event_pairs_queued_before_synchronization"] = False
    with pytest.raises(AssertionError):
        checked_blocks(row, queued=True)
