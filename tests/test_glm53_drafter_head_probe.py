"""Safety and independent numerical contracts for the live shadow-head diagnostic."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch


def _load(name: str) -> ModuleType:
    source = Path(__file__).parents[1] / "integrations/glm53_mxfp4" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, source)
    assert spec is not None and spec.loader is not None
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


analysis = _load("drafter_head_analysis")
probe = _load("drafter_head_probe")


def test_shadow_head_restores_after_exception() -> None:
    head = SimpleNamespace(weight=None, quant_method=object())
    original_method = head.quant_method
    with (
        pytest.raises(RuntimeError, match="projection failed"),
        probe.temporary_head(head, "dense", "alternate"),
    ):
        assert head.weight == "dense"
        raise RuntimeError("projection failed")
    assert head.weight is None and head.quant_method is original_method


def test_bf16_payloads_preserve_sign_and_subnormals() -> None:
    values = analysis.bf16_values(
        np.array([0x0000, 0x8000, 0x3F80, 0xC000, 0x0001], dtype=np.uint16)
    )
    assert values.tolist() == [0.0, -0.0, 1.0, -2.0, 2.0**-133]
    assert np.signbit(values[1])


def test_greedy_path_uses_previous_choice_and_stable_tie() -> None:
    ids = np.arange(7 * 16).reshape(7, 16)
    scores = np.full((7, 16, 16), -10.0)
    previous = 0
    expected = []
    for step in range(7):
        selected = step + 2
        scores[step, previous, selected:selected + 2] = 1.0
        previous = selected
        expected.append(int(ids[step, selected]))
    assert analysis.greedy_path(ids, scores) == expected


def test_capture_taps_distinguish_graphs_with_same_number_of_draft_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = False
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: active)
    draft = SimpleNamespace(compute_candidates=lambda hidden: (hidden[:, :2], hidden[:, :2]))
    spec = SimpleNamespace(model=draft, _sample_path=lambda ids, scores, count: None)

    def generate(count: int, padded_tokens: int) -> torch.Tensor:
        hidden = torch.full((7, 4), float(padded_tokens))
        ids, values = draft.compute_candidates(hidden)
        spec._sample_path(ids, values, count)
        return hidden

    spec._generate_draft = generate
    worker = SimpleNamespace(model_runner=SimpleNamespace(speculator=spec))
    probe.arm_capture(worker)
    spec._generate_draft(1, 8)
    assert not spec._mxwave_head_taps["graphs"]
    active = True
    captured = spec._generate_draft(1, 8)
    spec._generate_draft(1, 16)
    graphs = spec._mxwave_head_taps["graphs"]
    assert sorted(graphs) == [(1, 8), (1, 16)]
    assert graphs[(1, 8)]["hidden"] is captured
    captured.fill_(3)
    assert graphs[(1, 8)]["hidden"].eq(3).all()
    assert graphs[(1, 16)]["hidden"].eq(16).all()
    assert spec._mxwave_head_taps["capturing"] is None


def test_round_coverage_preserves_post_cap_lookahead_and_matches_acceptance() -> None:
    result = analysis.verify_round_coverage([11, 14, 17, 18], 10, 6, 2, 4, [2, 2, 0, 0, 0, 0, 0])
    assert result["passed"]
    assert result["post_cap_replay_starts"] == [17, 18]
    assert not result["literal_replay_count_equals_verified_rounds"]


@pytest.mark.parametrize("starts", [[11, 17, 18], [11, 14, 14, 17], [12, 14, 17, 18]])
def test_round_coverage_rejects_missing_duplicate_or_shifted_observations(
    starts: list[int],
) -> None:
    result = analysis.verify_round_coverage(starts, 10, 6, 2, 4, [2, 2, 0, 0, 0, 0, 0])
    assert not result["passed"]
