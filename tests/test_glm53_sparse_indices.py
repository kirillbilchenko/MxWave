"""Sparse slot diagnosis must preserve multiplicity and separate membership from order."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))

from analyze_sparse_indices import check_slot_multisets


def test_permuted_prefix_preserves_membership_and_checks_invalid_suffix() -> None:
    value = check_slot_multisets(
        np.array([0, 1], dtype=np.int32),
        np.array([[3, 5], [7, 11]], dtype=np.int32),
        np.array([[0, 4, -1, 8], [1, 5, -1, -1]], dtype=np.int32),
        np.array([[20, 12, -1, -1], [45, 29, -1, -1]], dtype=np.int32),
        np.array([2, 2], dtype=np.int32),
        4,
    )
    assert value["all_multisets_match"] and value["all_tails_are_minus_one"]
    assert not value["all_orders_match_input_columns"]


def test_duplicate_loss_and_nonempty_invalid_tail_are_independent_failures() -> None:
    args = (
        np.array([0], dtype=np.int32),
        np.array([[3, 5]], dtype=np.int32),
        np.array([[0, 0, 4, -1]], dtype=np.int32),
    )
    value = check_slot_multisets(
        *args,
        np.array([[12, 20, 20, 0]], dtype=np.int32),
        np.array([3], dtype=np.int32),
        4,
    )
    assert not value["all_multisets_match"]
    assert not value["all_tails_are_minus_one"]


def test_empty_valid_prefix_and_invalid_count_bounds() -> None:
    args = (
        np.array([0], dtype=np.int32),
        np.array([[3]], dtype=np.int32),
        np.array([[-1, 4]], dtype=np.int32),
        np.array([[-1, -1]], dtype=np.int32),
    )
    value = check_slot_multisets(*args, np.array([0], dtype=np.int32), 4)
    assert value["all_multisets_match"] and value["all_orders_match_input_columns"]
    with pytest.raises(ValueError, match="Valid count"):
        check_slot_multisets(*args, np.array([3], dtype=np.int32), 4)
