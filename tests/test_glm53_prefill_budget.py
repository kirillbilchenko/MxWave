"""Bound scheduling changes, retain memory/quality gates and cover added integer shapes."""

from __future__ import annotations

import importlib
import queue
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def source_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Import experiment code without starting a model or CUDA context."""
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))


def engine() -> Any:
    """Represent the native scheduler cap and independently fixed worker allocation."""
    return SimpleNamespace(has_work=lambda: False, input_queue=queue.Queue(), batch_queue=None,
        scheduler=SimpleNamespace(max_num_scheduled_tokens=4096),
        vllm_config=SimpleNamespace(speculative_config=None,
            scheduler_config=SimpleNamespace(max_num_batched_tokens=4096, max_num_seqs=16)))


def test_idle_budget_switch_preserves_capacity() -> None:
    module = importlib.import_module("prefill_budget")
    target = engine()
    for budget in (2048, 4096, 2048):
        result = module.configure(target, budget)
        assert result["budget_tokens"] == budget
        assert result["allocation_capacity_tokens"] == 4096


@pytest.mark.parametrize("value", [True, 2048.0, 4097, 2080, 0, -1])
def test_undeclared_budget_rejected(value: Any) -> None:
    with pytest.raises(ValueError):
        importlib.import_module("prefill_budget").configure(engine(), value)


@pytest.mark.parametrize("busy", ["running", "queued", "inflight"])
def test_budget_change_cannot_interrupt_work(busy: str) -> None:
    target = engine()
    if busy == "running":
        target.has_work = lambda: True
    elif busy == "queued":
        target.input_queue.put("request")
    else:
        target.batch_queue = ["request"]
    with pytest.raises(RuntimeError):
        importlib.import_module("prefill_budget").configure(target, 2048)
    assert target.scheduler.max_num_scheduled_tokens == 4096


def test_warmup_covers_old_shapes_and_five_new_capacities() -> None:
    old = importlib.import_module("stable_dispatch_warmup").specializations()
    module = importlib.import_module("budget_dispatch_warmup")
    new = module.specializations()
    keys = lambda rows: {(r["block"], r["capacity"]) for r in rows}
    assert len(new) == 39 and keys(old) <= keys(new)
    assert keys(new) - keys(old) == {(b, 4096) for b in (8, 16, 32, 64)} | {(48, 8192)}
    for block in (8, 16, 32, 48, 64):
        for tokens in range(2, 4097):
            padded = ((tokens + block - 1) // block) * block
            cases = [r for r in new if r["block"] == block
                     and r["first_tokens"] <= tokens <= r["last_tokens"]]
            assert len(cases) == 1 and cases[0]["capacity"] >= padded
    with pytest.raises(ValueError):
        module.specializations(4097)


def test_dispatch_extension_changes_only_the_shape_bound() -> None:
    root = Path(__file__).parents[1] / "integrations/glm53_mxfp4"
    old, new = ((root / name).read_text() for name in ("stable_dispatch.py", "budget_dispatch.py"))
    extract = lambda text: text[text.index("def canonicalize("):text.index("\ndef install()")]
    assert extract(old).replace("2048", "4096") == extract(new)


def test_protocol_retains_both_budgets_and_original_quality_limits() -> None:
    module = importlib.import_module("prefill_budget_protocol")
    plan = module.plan("a" * 64, "b" * 64, input_fp32=False)
    assert len(plan["timing_schedule"]) == 36
    assert len(plan["quality_pairs"]) == 94
    assert plan["admission_window_ms"] == 10
    assert plan["memory_limits"]["minimum_TP2_saving_bytes"] > 9 * 1024**3
    for case, repeats in (("c16-short", 12), ("c1-short", 6)):
        for budget in (2048, 4096):
            assert sum(t["case"] == case and t["budget_tokens"] == budget
                       for t in plan["timing_schedule"]) == repeats
    plan["numerical_limits"]["mean_decode_KL"] = 2e-5
    with pytest.raises(ValueError):
        module.validate_plan(plan)


def test_tail_can_fit_when_decode_tokens_have_budget() -> None:
    # Eight 256-token prompts remain while eight admitted requests each decode one token.
    prompt_tokens, decode_tokens = 8 * 256, 8
    assert prompt_tokens + decode_tokens > 2048
    assert prompt_tokens + decode_tokens <= 4096


def test_latency_tail_and_coalesced_streams_are_not_hidden_by_median() -> None:
    module = importlib.import_module("prefill_budget_protocol")
    requests = [{"arrivals": [{"seconds": first, "tokens": 1},
                               {"seconds": first + 0.1, "tokens": 1}]}
                for first in [0.1] * 18 + [1.0, 2.0]]
    result = module.latency_metrics(requests)
    assert result["mean_ttft_seconds"] == pytest.approx(0.24)
    assert result["p95_ttft_seconds"] == 1.0
    assert result["p95_itl_seconds"] == pytest.approx(0.1)
    requests[-1]["arrivals"][-1]["tokens"] = 2
    mixed = module.latency_metrics(requests)
    assert mixed["requests"] == 20 and mixed["single_token_streams"] == 19
    assert mixed["single_token_intervals"] == 19


def test_mean_regression_rejects_an_improved_median() -> None:
    module = importlib.import_module("prefill_budget_protocol")
    summaries = [{"arm": arm, "case": case[0], "budget_tokens": budget,
                  "median_tps": 100, "median_ttft_seconds": 1 if budget == 2048 else 0.7,
                  "mean_ttft_seconds": 1 if budget == 2048 else 1.1,
                  "p95_ttft_seconds": 2, "p95_itl_seconds": 0.1,
                  "requests": 12, "single_token_streams": 12}
                 for arm in module.ARMS for case in module.CASES for budget in module.BUDGETS]
    result = module.policy_decisions(summaries, numerical_passed=True, timed_ids_match=True)
    assert not result["policy_gate_passed"]
    assert not result["policy_checks"]["candidate_C16_TTFT_improves"]
