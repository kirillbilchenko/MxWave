"""The final admission comparison must preserve quality and distinguish policy effects."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def source_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Import the experiment protocol without initializing CUDA."""
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))


def test_policy_order_is_balanced_and_every_primary_trial_is_retained() -> None:
    module = importlib.import_module("c16_admission_protocol")
    schedule = module.timing_schedule()
    assert len(schedule) == 36
    assert sum(row["concurrency"] for row in schedule) == 396
    for case, repeats in (("c16-short", 12), ("c1-short", 6)):
        orders = []
        for repeat in range(repeats):
            rows = [row for row in schedule if (row["case"], row["repeat"]) == (case, repeat)]
            orders.append(tuple(row["window_ms"] for row in rows))
        assert orders.count((3, 10)) == orders.count((10, 3)) == repeats // 2
    assert len(module.quality_pairs()) == len(set(module.quality_pairs())) == 94


@pytest.mark.parametrize("fault", ["fp32", "quality", "omitted_window", "omitted_trial"])
def test_protocol_rejects_unqualified_kernel_or_weakened_comparison(fault: str) -> None:
    module = importlib.import_module("c16_admission_protocol")
    plan = module.plan("a" * 64, "b" * 64, input_fp32=False)
    if fault == "fp32":
        plan["candidate_input_fp32"] = True
    elif fault == "quality":
        plan["numerical_limits"]["mean_decode_KL"] = 2e-5
    elif fault == "omitted_window":
        plan["windows_ms"] = [10]
    else:
        plan["timing_schedule"].pop()
    with pytest.raises(ValueError):
        module.validate_plan(plan)


def summaries() -> list[dict[str, Any]]:
    """Represent useful admission latency gains with compressed throughput at parity."""
    return [
        {
            "arm": arm, "case": case, "window_ms": window, "median_tps": 80.0,
            "median_ttft_seconds": (
                (2.0 if window == 3 else 1.8) if case == "c16-short"
                else (0.4 if window == 3 else 0.407)
            ),
        }
        for arm in ("control", "candidate")
        for case in ("c1-short", "c16-short")
        for window in (3, 10)
    ]


def test_policy_can_pass_at_compression_parity_without_rewriting_old_speed_gate() -> None:
    module = importlib.import_module("c16_admission_protocol")
    result = module.policy_decisions(summaries(), numerical_passed=True, timed_ids_match=True)
    assert result["policy_gate_passed"]
    assert result["compression_speed_gate_by_window"] == {"3": False, "10": False}


@pytest.mark.parametrize("fault", ["quality", "tokens", "C1_delay", "TPS", "C16_no_gain"])
def test_latency_policy_cannot_hide_a_failed_guard(fault: str) -> None:
    module = importlib.import_module("c16_admission_protocol")
    values = summaries()
    for row in values:
        if row["arm"] == "candidate" and row["window_ms"] == 10:
            if fault == "C1_delay" and row["case"] == "c1-short":
                row["median_ttft_seconds"] = 0.42
            if fault == "TPS":
                row["median_tps"] = 75.0
            if fault == "C16_no_gain" and row["case"] == "c16-short":
                row["median_ttft_seconds"] = 2.0
    result = module.policy_decisions(
        values, numerical_passed=fault != "quality", timed_ids_match=fault != "tokens"
    )
    assert not result["policy_gate_passed"]


def test_launcher_preserves_the_qualified_parent_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module("run_c16_admission")
    parent: tuple[list[str], dict[str, Any]] = (["exact", "parent", "command"], {"same": True})
    monkeypatch.setattr(module.warmed, "launch", lambda *args: parent)
    for arm in ("control", "candidate"):
        assert module.launch(0, "test", arm, {}, 4, "manifest") is parent
    with pytest.raises(ValueError):
        module.launch(0, "test", "undeclared", {}, 4, "manifest")
