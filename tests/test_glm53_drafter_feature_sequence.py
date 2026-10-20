"""Accepted CT features must exclude rejected tokens and post-cap work."""

from __future__ import annotations

import copy

import pytest
import torch

from integrations.glm53_mxfp4.drafter_feature_sequence import LAYERS, AcceptedHistory


def frame(index: int) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    """Make a prompt plus one partially accepted verification fixture."""
    ids = [10, 11, 12] if index == 0 else [20, 21, 99, 99, 99, 99, 99, 99]
    start, valid, sampled = (0, 3, [20]) if index == 0 else (3, 2, [21, 22])
    data = {
        "target_input_ids": torch.tensor(ids),
        "target_positions": torch.arange(start, start + len(ids)),
        "target_num_sampled": torch.tensor([len(sampled)]),
        "target_sampled_tokens": torch.tensor([sampled]),
        "target_num_rejected": torch.tensor([0 if index == 0 else 6]),
        "draft_query_positions": torch.arange(start + valid, start + valid + 8),
        "draft_query_ids": torch.tensor([sampled[-1]] + [154856] * 7),
        "anchor_before_propose": torch.tensor([[sampled[-1]]]),
    }
    data.update({
        f"aux_layer_{layer}": torch.arange(len(ids)).to(torch.bfloat16)[:, None]
        .expand(-1, 4096).clone()
        for layer in LAYERS
    })
    return data, {"index": index, "target_num_draft_tokens": 0 if index == 0 else 7}


def test_accepted_features_exclude_rejection_and_post_cap() -> None:
    history = AcceptedHistory()
    history.append(*frame(0))
    history.append(*frame(1))
    report = history.validate_response([10, 11, 12], [20, 21])
    assert history.tokens.tolist() == [10, 11, 12, 20, 21]
    assert history.context().shape == (5, 20480)
    assert report["feature_rows"] == 4
    assert report["rejected_rows_excluded"] == 6
    assert report["post_cap_sampled_tokens_excluded"] == 1
    assert report["post_cap_feature_rows_excluded"] == 1


@pytest.mark.parametrize("fault", ["position", "accepted_token", "anchor", "rejected_count"])
def test_corrupt_alignment_is_rejected(fault: str) -> None:
    history = AcceptedHistory()
    history.append(*frame(0))
    data, meta = copy.deepcopy(frame(1))
    key, index = {
        "position": ("target_positions", (0,)),
        "accepted_token": ("target_input_ids", (1,)),
        "anchor": ("anchor_before_propose", (0, 0)),
        "rejected_count": ("target_num_rejected", (0,)),
    }[fault]
    data[key][index] += 1
    with pytest.raises(ValueError):
        history.append(data, meta)


def test_missing_frame_and_response_disagreement_fail_closed() -> None:
    history = AcceptedHistory()
    data, meta = frame(0)
    with pytest.raises(ValueError, match="unordered"):
        history.append(data, {**meta, "index": 1})
    history.append(data, meta)
    with pytest.raises(ValueError, match="API response"):
        history.validate_response([10, 11, 12], [999])
