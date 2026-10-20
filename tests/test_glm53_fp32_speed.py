"""Timing controls cannot hide precision failures, drift or missing events."""

from __future__ import annotations

import copy
import importlib
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def source_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Load pure timing contracts without initializing CUDA."""
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))


def samples() -> list[dict]:
    """Represent a genuinely faster candidate with stable unchanged controls."""
    probe = importlib.import_module("fp32_speed_probe")
    p = probe.protocol()
    rows = []
    for trial in range(p["trials"]):
        pairs = list(probe.PAIRS) if trial % 2 == 0 else list(reversed(probe.PAIRS))
        order = ["a", "b", "b", "a"] if trial % 2 == 0 else ["b", "a", "a", "b"]
        rows.append({"trial": trial, "pair_order": pairs, "pairs": {
            name: [{"arm": arm, "us": [80. if arm == "b" and name.startswith("candidate_vs")
                                      else 100.] * p["samples_per_leg"]} for arm in order]
            for name in pairs}})
    return rows


@pytest.mark.parametrize("control", ["original_repeat", "rebuilt_repeat", "candidate_repeat",
                                     "rebuilt_vs_original"])
def test_each_unstable_control_blocks_even_a_fast_candidate(control: str) -> None:
    probe = importlib.import_module("fp32_speed_probe")
    data = samples()
    assert probe.summarize(data)["timing_screen_passed"]
    for trial in data:
        for leg in trial["pairs"][control]:
            if leg["arm"] == "b":
                leg["us"] = [105.] * 8
    result = probe.summarize(data)
    assert result["candidate_speed_passed"] and not result["timing_screen_passed"]
    assert not result["parent_exact_word_screen_passed"] and not result["model_quality_qualified"]


def test_opposite_order_drift_cannot_cancel() -> None:
    probe = importlib.import_module("fp32_speed_probe")
    data = samples()
    for row in data:
        for leg in row["pairs"]["candidate_repeat"]:
            if leg["arm"] == "b":
                leg["us"] = [105. if row["trial"] % 2 else 95.] * 8
    result = probe.summarize(data)
    assert result["pairs"]["candidate_repeat"]["combined"]["ratio"] == 1.
    assert not result["repeat_controls_passed"]


@pytest.mark.parametrize("value", [0., -1., float("nan"), float("inf")])
def test_invalid_event_rejected(value: float) -> None:
    probe = importlib.import_module("fp32_speed_probe")
    data = samples()
    data[0]["pairs"]["original_repeat"][0]["us"][0] = value
    with pytest.raises(ValueError, match="invalid CUDA events"):
        probe.summarize(data)


def test_missing_events_and_changed_comparison_order_rejected() -> None:
    probe = importlib.import_module("fp32_speed_probe")
    data = samples()
    data[0]["pairs"]["original_repeat"][0]["us"].pop()
    with pytest.raises(ValueError):
        probe.summarize(data)
    data = samples()
    data[1]["pair_order"].reverse()
    with pytest.raises(ValueError):
        probe.summarize(data)


def complete_report() -> dict:
    """Provide all declared cases with raw events for coverage and lifetime checks."""
    probe = importlib.import_module("fp32_speed_probe")
    p = probe.protocol()
    data = samples()
    summary = probe.summarize(data)
    measurement = {"outputs_match_parent_before": True, "outputs_match_parent_after": True,
        "guards_unchanged": True, "graph_matches_eager": True, "graph_retained_bytes": 0,
        "timing": [{"mode": "graph", "cache": cache, "samples": data, "summary": summary}
                   for cache in p["cache_regimes"]]}
    return {"status": "complete", "protocol": p, "rank": 0, "cases": [
        {"name": f"kda-{kind}-l{layer}-r0", "decoded_weight_bits_exact": True,
         "packed_buffers_unchanged": True, "measurements": [
             {"tokens": n, "split": split, **copy.deepcopy(measurement)}
             for n in p["tokens"] for split in p["splits"]]}
        for layer in p["layers"] for kind in p["kinds"]]}


def test_full_coverage_and_graph_lifetime_required_without_quality_promotion() -> None:
    probe = importlib.import_module("fp32_speed_probe")
    report = complete_report()
    result = probe.assess(report)
    assert result["conditions"] == 144 and result["events"] == 442368
    assert not result["model_quality_or_serving_speed_qualified"]
    assert not result["parent_exact_word_screen_passed"]
    report["cases"][0]["measurements"][0]["graph_retained_bytes"] = 4
    with pytest.raises(ValueError, match="retained"):
        probe.assess(report)
    report["cases"].pop()
    with pytest.raises(ValueError, match="Missing"):
        probe.assess(report)


def test_timing_controller_bounded_without_profiler_privileges() -> None:
    runner = importlib.import_module("run_fp32_speed")
    command = runner.command(1)
    assert "SYS_ADMIN" not in command and command[command.index("--network") + 1] == "none"
    assert "/src/benchmark_fp32_speed.py" in command and "/result/timing.json" in command
    assert "900" in command and "8g" in command
    for rank, profile in ((-1, False), (0, True)):
        with pytest.raises(ValueError):
            runner.command(rank, profile=profile)
