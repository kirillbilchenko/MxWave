"""Reject unstable comparisons and bind the storage tradeoff to actual parent weights."""

from __future__ import annotations

import importlib
from copy import deepcopy
from pathlib import Path

import pytest


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))
    return importlib.import_module("dense_output_probe")


def events(candidate=80.):
    """Supply stable independent controls and a separately measured faster candidate."""
    result = []
    for trial in range(16):
        pairs = ["parent_repeat", "dense_vs_parent", "dense_repeat"]
        order = ["a", "b", "b", "a"] if trial % 2 == 0 else ["b", "a", "a", "b"]
        result.append({"trial": trial, "pair_order": pairs if trial % 2 == 0 else pairs[::-1],
                       "pairs": {name: [{"arm": arm, "us": [
                           candidate if name == "dense_repeat" or (
                               name == "dense_vs_parent" and arm == "b") else 100.
                       ] * 8} for arm in order] for name in pairs}})
    return result


def test_fast_candidate_cannot_hide_an_unstable_dense_control(probe):
    rows = events()
    for row in rows:
        for leg in row["pairs"]["dense_repeat"]:
            if leg["arm"] == "b":
                leg["us"] = [84.] * 8
    result = probe.summarize(rows, 256)
    assert result["candidate_cost_screen_passed"]
    assert not result["repeat_controls_passed"]
    assert not result["component_timing_screen_passed"]
    assert result["events"] == 1536


def test_opposing_drift_does_not_cancel_the_repeat_gate(probe):
    rows = events()
    for row in rows:
        for leg in row["pairs"]["parent_repeat"]:
            if leg["arm"] == "b":
                leg["us"] = [96. if row["trial"] % 2 else 104.] * 8
    result = probe.summarize(rows, 256)
    assert result["pairs"]["parent_repeat"]["combined"]["ratio"] == 1.
    assert not result["repeat_controls_passed"]


def test_decode_cost_budget_and_prefill_gain_are_distinct(probe):
    assert probe.summarize(events(149.), 1)["component_timing_screen_passed"]
    assert not probe.summarize(events(151.), 1)["component_timing_screen_passed"]
    assert not probe.summarize(events(99.), 256)["component_timing_screen_passed"]
    assert probe.summarize(events(94.), 256)["component_timing_screen_passed"]
    assert not probe.summarize(events(94.), 256)["decode_budget_is_model_speed_qualification"]


@pytest.mark.parametrize("change", ["trial", "sample", "order", "nan"])
def test_incomplete_or_changed_events_are_rejected(probe, change):
    rows = events()
    if change == "trial":
        rows.pop()
    elif change == "sample":
        rows[0]["pairs"]["dense_vs_parent"][0]["us"].pop()
    elif change == "order":
        rows[0]["pair_order"].reverse()
    else:
        rows[0]["pairs"]["dense_vs_parent"][0]["us"][0] = float("nan")
    with pytest.raises(ValueError):
        probe.summarize(rows, 16)


def test_fixture_selection_spans_each_actual_rank_and_rejects_changed_parent(probe):
    before = []
    for rank in (0, 1):
        outputs = [{"prefix": f"language_model.model.layers.{layer}.self_attn.o_proj",
                    "rank": rank, "kind": "output", "logical_shape": [4096, 4096],
                    "native_shape": [4096, 4096], "raw_bytes": 33554432,
                    "packed_bytes": 23000000 + layer * 10,
                    "loaded_weight_sha256": f"{rank}{layer:063d}",
                    "gpu_decode_bits_exact": True} for layer in reversed(range(34))]
        before.append({"tp_rank": rank, "allocation": {"kda": {
            "layers": outputs + [{"kind": "input"}] * 34}}})
    parent = {"status": "complete", "arm": "candidate", "source_before": before,
              "source_after": deepcopy(before)}
    cases = probe.select_cases(parent)
    assert [x["name"] for x in cases] == [
        f"kda-output-l{layer}-r{rank}" for rank in (0, 1) for layer in (0, 17, 33)]
    assert cases[0]["source_tensors"][0]["range"] == [0, 4096]
    assert cases[-1]["source_tensors"][0]["range"] == [4096, 8192]
    parent["source_after"][0]["allocation"]["kda"]["layers"][0]["packed_bytes"] += 1
    with pytest.raises(ValueError, match="unchanged"):
        probe.select_cases(parent)


def test_component_has_no_profiler_privileges_and_a_finite_timeout(probe):
    controller = importlib.import_module("run_dense_output")
    for rank in (0, 1):
        command = controller.command(rank)
        assert "--cap-add" not in command and "--user" in command
        assert command[command.index("--network") + 1] == "none"
        assert command[command.index("--memory") + 1] == "8g"
        assert command[command.index("--rank") + 1] == str(rank)
        assert command[command.index("--kill-after=10") + 1] == "900"
    with pytest.raises(ValueError):
        controller.command(0, profile=True)
