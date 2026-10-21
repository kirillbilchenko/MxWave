"""Keep aggregate proposal acceptance distinct from prefix survival."""

from __future__ import annotations

import pytest

from integrations.glm53_mxfp4.measure_production_acceptance import acceptance, metrics


def sample() -> dict[str, float]:
    """Represent ten seven-token rounds with progressively fewer surviving prefixes."""
    values = {"vllm:spec_decode_num_drafts_total{engine=\"0\"}": 10.,
              "vllm:spec_decode_num_draft_tokens_total{engine=\"0\"}": 70.,
              "vllm:spec_decode_num_accepted_tokens_total{engine=\"0\"}": 28.}
    for i, count in enumerate((7, 6, 5, 4, 3, 2, 1)):
        values[f'vllm:spec_decode_num_accepted_tokens_per_pos_total{{position="{i}"}}'] = count
    return values


def test_aggregate_and_first_position_have_different_denominators() -> None:
    result = acceptance(sample())
    assert result["acceptance_rate"] == .4
    assert result["accepted_per_round"] == 2.8
    assert result["prefix_survival"] == [.7, .6, .5, .4, .3, .2, .1]


@pytest.mark.parametrize("wrong", ["rounds", "positions", "accepted"])
def test_counter_mismatch_rejected(wrong: str) -> None:
    values = sample()
    key = {"rounds": 'vllm:spec_decode_num_drafts_total{engine="0"}',
           "positions": 'vllm:spec_decode_num_accepted_tokens_per_pos_total{position="0"}',
           "accepted": 'vllm:spec_decode_num_accepted_tokens_total{engine="0"}'}[wrong]
    values[key] += 1
    with pytest.raises(ValueError):
        acceptance(values)


def test_metrics_excludes_creation_timestamps_and_preserves_activity() -> None:
    raw = '\n'.join([f"{k} {v}" for k, v in sample().items()] + [
        'vllm:spec_decode_num_drafts_created{engine="0"} 1000',
        'vllm:num_requests_running{engine="0"} 0',
        'vllm:request_success_total{finished_reason="length"} 4',
    ])
    result = metrics(raw)
    assert len(result) == 12
    assert result['vllm:request_success_total{finished_reason="length"}'] == 4
