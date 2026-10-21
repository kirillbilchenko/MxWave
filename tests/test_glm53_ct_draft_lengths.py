"""Keep target settings fixed and reject mislabeled native draft-depth measurements."""

from __future__ import annotations

import json

import pytest

from integrations.glm53_mxfp4.measure_ct_draft_lengths import summarize_counters
from integrations.glm53_mxfp4.prepare_ct_draft_lengths import configure


def command() -> list[str]:
    """Provide a small launch preserving target and serving settings."""
    return ["serve", "/model", "--speculative-config", json.dumps({
        "method": "dflash", "model": "/draft", "num_speculative_tokens": 7,
        "num_speculative_tokens_per_batch_size": [[1, 1, 7], [2, 2, 5], [3, 16, 3]],
        "rejection_sample_method": "standard",
    }), "--max-model-len", "8192", "--kv-cache-memory-bytes", "6442450944"]


@pytest.mark.parametrize("depth", [0, 3, 5, 7])
def test_only_speculation_changes(depth: int) -> None:
    original = command()
    result = configure(original, depth)
    assert original == command()
    if depth:
        config = json.loads(result[3])
        assert config["num_speculative_tokens"] == depth
        assert config["num_speculative_tokens_per_batch_size"][0] == [1, 1, depth]
        assert config["rejection_sample_method"] == "standard"
        del result[2:4]
    assert result == original[:2] + original[4:]


def test_off_means_no_speculative_work() -> None:
    assert summarize_counters({}, 0)["acceptance_rate"] is None
    with pytest.raises(ValueError):
        summarize_counters({"vllm:spec_decode_num_drafts": 1}, 0)


def test_three_drafts_use_three_as_denominator() -> None:
    values = {"vllm:spec_decode_num_drafts": 10., "vllm:spec_decode_num_draft_tokens": 30.,
              "vllm:spec_decode_num_accepted_tokens": 18.}
    for i, n in enumerate((9., 6., 3.)):
        values[f'vllm:spec_decode_num_accepted_tokens_per_pos{{position="{i}"}}'] = n
    result = summarize_counters(values, 3)
    assert result["acceptance_rate"] == .6
    assert result["prefix_survival"] == [.9, .6, .3]
    with pytest.raises(ValueError):
        summarize_counters(values, 7)
    values['vllm:spec_decode_num_accepted_tokens_per_pos{position="2"}'] = 4.
    with pytest.raises(ValueError):
        summarize_counters(values, 3)


def test_refuse_other_drafter_and_depth() -> None:
    with pytest.raises(ValueError):
        configure(command(), 4)
    value = command()
    value[3] = json.dumps({"method": "mtp", "model": "/draft"})
    with pytest.raises(ValueError):
        configure(value, 3)
