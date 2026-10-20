"""Keep real speculative metric suffixes, positions and resets visible in the MTP pilot."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))
from analyze_mtp_pilot import numerical_passed
from mtp_metrics import counters, delta


def test_native_counter_suffix_and_position_labels_are_preserved() -> None:
    value = ('# HELP vllm:spec_decode_num_drafts\n'
        'vllm:spec_decode_num_drafts_total{model_name="pilot"} 3\n'
        'vllm:spec_decode_num_accepted_tokens_per_pos{position="0"} 2\n'
        'vllm:spec_decode_num_accepted_tokens_per_pos{position="1"} 1\n')
    assert counters(value) == {'vllm:spec_decode_num_drafts{model_name="pilot"}': 3.,
        'vllm:spec_decode_num_accepted_tokens_per_pos{position="0"}': 2.,
        'vllm:spec_decode_num_accepted_tokens_per_pos{position="1"}': 1.}
    assert delta('', 'vllm:spec_decode_num_drafts 5\n') == {'vllm:spec_decode_num_drafts': 5.}


@pytest.mark.parametrize("before,after", [
    ('vllm:spec_decode_num_drafts 5', 'vllm:spec_decode_num_drafts 4'),
    ('vllm:spec_decode_num_drafts 5', ''),
    ('', 'vllm:spec_decode_num_drafts 1\nvllm:spec_decode_num_drafts_total 2'),
])
def test_reset_disappearance_or_duplicate_cannot_become_an_acceptance_gain(
    before: str, after: str,
) -> None:
    with pytest.raises(ValueError):
        delta(before, after)


def test_mtp_numerical_gate_cannot_pass_missing_conditioning_or_loosened_floors() -> None:
    value = {"expected_decode_steps": 48, "decode_steps": 48,
        "all_decode_conditioning_matches": True, "all_generated_ids_match": True,
        "mean_decode_kl": 1e-5, "mean_decode_nll_difference": -1e-4}
    assert numerical_passed(value)
    for key, wrong in (("mean_decode_kl", 1.001e-5),
                       ("mean_decode_nll_difference", -1.001e-4),
                       ("all_decode_conditioning_matches", False),
                       ("all_generated_ids_match", False), ("decode_steps", 47)):
        assert not numerical_passed({**value, key: wrong})
