"""Reject lossy routing reconstruction and changes to the retained causal prefix."""

from __future__ import annotations

import numpy as np
import pytest

from integrations.glm53_mxfp4.expert_wide_fixture import expand, recover_topk


def test_routing_reconstruction_ignores_masked_capacity_but_preserves_route_slots() -> None:
    tokens = np.array([2, 0, 4, 4, 1, 3, 4, 4, -999, -999], dtype=np.int32)
    experts = np.array([5, 7, -999], dtype=np.int32)
    ids = recover_topk(tokens, experts, 8, rows=2, topk=2, block=4, num_experts=8)
    assert ids.tolist() == [[5, 7], [5, 7]]
    for corrupted in (
        np.array([2, 0, 4, 4, 1, 1, 4, 4], dtype=np.int32),
        np.array([2, 0, 4, 4, 1, 4, 4, 4], dtype=np.int32),
        np.array([2, 0, 5, 4, 1, 3, 4, 4], dtype=np.int32),
    ):
        with pytest.raises(ValueError):
            recover_topk(corrupted, experts, 8, rows=2, topk=2, block=4, num_experts=8)
    with pytest.raises(ValueError, match="invalid expert"):
        recover_topk(tokens, np.array([5, 8], dtype=np.int32), 8,
                     rows=2, topk=2, block=4, num_experts=8)


@pytest.mark.parametrize("rows", [8, 16, 32, 64, 128])
def test_wider_fixtures_preserve_real_prefix_without_aliasing(rows: int) -> None:
    x = np.full((8, 4096), 0x3F80, dtype=np.uint16)
    ids = np.arange(64, dtype=np.int32).reshape(8, 8)
    weights = np.full((8, 8), 0.125, dtype=np.float32)
    repeated = expand(x, ids, weights, rows, "repeat")
    spread = expand(x, ids, weights, rows, "spread")
    for name, original in (("input_bf16_bits", x), ("topk_ids", ids), ("topk_weights", weights)):
        assert spread[name][:8].tobytes() == repeated[name][:8].tobytes() == original.tobytes()
        assert not np.shares_memory(spread[name], original)
    assert all(len(np.unique(row)) == 8 for row in spread["topk_ids"])
    if rows > 8:
        assert not np.array_equal(spread["topk_ids"][8:], repeated["topk_ids"][8:])
    with pytest.raises(ValueError):
        expand(x, ids, weights, 129, "repeat")
