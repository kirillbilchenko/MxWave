"""Keep diagnostic coverage, repeat failures and profiler permissions explicit."""

from __future__ import annotations

import importlib
from copy import deepcopy
from pathlib import Path

import pytest


@pytest.fixture
def probe(monkeypatch):
    """Load only CPU protocol/controller helpers."""
    root = Path(__file__).parents[1] / "integrations/glm53_mxfp4"
    monkeypatch.syspath_prepend(str(root))
    return importlib.import_module("kda_memory_probe")


@pytest.fixture
def samples():
    """Use repeat arms with the same event latency in balanced bracketed order."""
    return [{"trial": i, "legs": [{"arm": a, "us": [100., 100., 100., 100.]}
                                    for a in (["a", "b", "b", "a"] if i % 2 == 0
                                              else ["b", "a", "a", "b"])]} for i in range(8)]


def test_repeat_instability_remains_a_failure(probe, samples):
    assert probe.summarize(samples)["repeat_control_passed"]
    for trial in samples:
        for leg in trial["legs"]:
            if leg["arm"] == "b":
                leg["us"] = [104.] * 4
    result = probe.summarize(samples)
    assert not result["repeat_control_passed"]
    assert result["median_repeat_ratio"] == 1.04
    assert result["events"] == 128


@pytest.mark.parametrize("change", ["missing_trial", "missing_sample", "order", "nan"])
def test_invalid_event_evidence_is_rejected(probe, samples, change):
    if change == "missing_trial":
        samples.pop()
    elif change == "missing_sample":
        samples[0]["legs"][0]["us"].pop()
    elif change == "order":
        samples[0]["legs"][0]["arm"] = "b"
    else:
        samples[0]["legs"][0]["us"][0] = float("nan")
    with pytest.raises(ValueError):
        probe.summarize(samples)


def test_full_profile_coverage_and_no_model_qualification(probe):
    report = {"protocol": probe.protocol(), "profile": True, "status": "complete",
              "CUDA_peak_bytes": 1024, "operations": [
                  {"name": name, "exact_output_and_guards": True,
                   "profile_label": "kda_memory/" + name} for name in probe.operation_names()]}
    result = probe.assess(report)
    assert result["diagnostic_complete"] and result["operations"] == 22
    assert not result["model_quality_or_serving_speed_qualified"]
    assert not result["logical_payload_is_measured_DRAM_traffic"]
    for kind in ("missing", "guard", "label", "memory", "protocol"):
        bad = deepcopy(report)
        if kind == "missing":
            bad["operations"].pop()
        elif kind == "guard":
            bad["operations"][0]["exact_output_and_guards"] = False
        elif kind == "label":
            bad["operations"][0]["profile_label"] = "other"
        elif kind == "memory":
            bad["CUDA_peak_bytes"] = 3 * 1024**3
        else:
            bad["protocol"]["prefill_rows"] = [256]
        with pytest.raises(ValueError):
            probe.assess(bad)


def test_only_profile_container_has_counter_capability(probe):
    controller = importlib.import_module("run_kda_memory")
    for rank in (0, 1):
        timing = controller.command(rank, profile=False)
        profile = controller.command(rank, profile=True)
        for command in (timing, profile):
            assert command[command.index("--network") + 1] == "none"
            assert command[command.index("--memory") + 1] == "8g"
            assert command[command.index("--entrypoint") + 1] == "/usr/bin/timeout"
            assert command[command.index("--name") + 1].endswith(f"-r{rank}")
        assert "--cap-add" not in timing and "/ncu/ncu" not in timing
        assert "--user" in timing and "--user" not in profile
        assert profile[profile.index("--cap-add") + 1] == "SYS_ADMIN"
        assert profile[profile.index("--clock-control") + 1] == "none"
        assert profile[profile.index("--cache-control") + 1] == "all"
    with pytest.raises(ValueError):
        controller.command(2, profile=False)
