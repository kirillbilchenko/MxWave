"""The decode correction must preserve native contracts and limit its shape scope."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from integrations.glm53_mxfp4.canonical_sparse_decode import canonical_decode


@pytest.mark.parametrize("rows,next_n,sorted_result", [(8, 1, True), (16, 1, False), (8, 8, False)])
def test_shape_scope_and_native_return(rows: int, next_n: int, sorted_result: bool) -> None:
    scores = torch.zeros(rows, 600, dtype=torch.float32)
    lengths = torch.full((rows,), 600, dtype=torch.int32)
    output = torch.empty(rows, 512, dtype=torch.int32)
    selected = torch.arange(512, dtype=torch.int32).roll(81).expand(rows, -1).clone()
    selected[0, :400] = -1
    calls = []
    sentinel = object()

    def native(*args: Any) -> object:
        calls.append(args)
        output.copy_(selected)
        return sentinel

    assert canonical_decode(native, scores, next_n, lengths, output, rows, 600, 1, 512) is sentinel
    assert len(calls) == 1
    assert torch.equal(output, selected.sort(-1).values if sorted_result else selected)
    assert torch.equal((output == -1).sum(-1), (selected == -1).sum(-1))


def test_selected_shape_rejects_invalid_layout_before_native_call() -> None:
    def native(*args: Any) -> None:
        raise AssertionError("Must reject before invoking native")

    with pytest.raises(ValueError, match="Unqualified"):
        canonical_decode(native, torch.zeros(8, 600), 1, torch.ones(8, dtype=torch.int64),
                         torch.empty(8, 512, dtype=torch.int32), 8, 600, 1, 512)
