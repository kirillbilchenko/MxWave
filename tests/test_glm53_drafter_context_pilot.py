"""Training block boundaries and offline selection must preserve the frozen contract."""

from __future__ import annotations

from integrations.glm53_mxfp4.train_drafter_context_pilot import (
    anchors_for,
    grouped_metrics,
    select_checkpoint,
)


def test_blocks_include_eos_as_label_but_never_post_eos_text() -> None:
    tokens = [1, 2, 3] + list(range(20, 28)) + [99] + [888] * 10
    selected = anchors_for(tokens, 3, [99], seed_key="test", validation=True)
    assert selected == [3, 4]
    assert tokens[selected[-1] + 7] == 99
    assert anchors_for([1, 2, 20, 99, 888], 2, [99], seed_key="test", validation=False) == []


def test_training_anchor_selection_is_reproducible_and_bounded() -> None:
    tokens = list(range(100))
    chosen = anchors_for(tokens, 10, [], seed_key="42:train:one", validation=False)
    assert chosen == anchors_for(tokens, 10, [], seed_key="42:train:one", validation=False)
    assert len(set(chosen)) == 4 and min(chosen) >= 10 and max(chosen) + 7 < len(tokens)


def test_domain_aggregation_weights_prompts_equally() -> None:
    result = grouped_metrics([
        {"domain": "code", "mean_prefix": 1.0, "unary_ce": 2.0, "coverage": 0.5},
        {"domain": "code", "mean_prefix": 3.0, "unary_ce": 4.0, "coverage": 1.0},
    ])
    assert result["code"] == {
        "prompts": 2, "mean_prefix": 2.0, "unary_ce": 3.0, "coverage": 0.75,
    }


def test_checkpoint_selection_rejects_regression_and_prefers_earlier_ties() -> None:
    baseline = {d: {"mean_prefix": 3.0, "unary_ce": 2.0}
                for d in ("code", "chat", "explanation")}

    def candidate(step: int, code: float, chat: float) -> dict:
        return {
            "step": step, "relative_l2": 0.001,
            "domains": {
                "code": {"mean_prefix": code, "unary_ce": 1.5},
                "chat": {"mean_prefix": chat, "unary_ce": 2.0},
                "explanation": {"mean_prefix": 3.0, "unary_ce": 2.0},
            },
        }

    good, tied, bad = candidate(8, 4.0, 3.0), candidate(16, 4.0, 3.0), candidate(24, 5.0, 2.9)
    assert select_checkpoint(baseline, [bad, tied, good]) == good
    assert select_checkpoint(baseline, [candidate(8, 3.0, 3.0)]) is None
    assert select_checkpoint({"code": baseline["code"]}, [good]) is None
