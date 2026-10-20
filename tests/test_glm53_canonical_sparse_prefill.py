"""Canonical selection must preserve padding, membership, native calls and returns."""

from __future__ import annotations

import pytest
import torch

from integrations.glm53_mxfp4.canonical_sparse_prefill import canonical_topk


def test_canonical_selection_preserves_multiset_padding_and_native_contract() -> None:
    scores = torch.zeros((4, 600), dtype=torch.float32)
    starts = torch.zeros(4, dtype=torch.int32)
    ends = torch.tensor([0, 1, 16, 600], dtype=torch.int32)
    selected = torch.full((4, 512), -1, dtype=torch.int32)
    selected[1, 94] = 0
    selected[2, :16] = torch.arange(16).flip(0)
    selected[3] = torch.arange(88, 600).roll(127)
    output = torch.empty_like(selected)
    sentinel = object()
    calls = []

    def native(*args: object) -> object:
        calls.append(args)
        output.copy_(selected)
        return sentinel

    result = canonical_topk(native, scores, starts, ends, output, 4, 600, 1, 512)
    assert result is sentinel and len(calls) == 1
    assert calls[0][:4] == (scores, starts, ends, output)
    assert torch.equal(output, selected.sort(dim=-1).values)
    assert torch.equal((output >= 0).sum(-1), torch.tensor([0, 1, 16, 512]))
    assert not scores.any()


def test_unqualified_layout_fails_before_native_mutation() -> None:
    scores = torch.zeros((4, 600), dtype=torch.float32)

    def native(*args: object) -> None:
        raise AssertionError("Must reject the layout before calling native")

    with pytest.raises(ValueError, match="Unqualified"):
        canonical_topk(native, scores, torch.zeros(4, dtype=torch.int32),
                       torch.ones(4, dtype=torch.int32), torch.empty(4, 256, dtype=torch.int32),
                       4, 600, 1, 256)
