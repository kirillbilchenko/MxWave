"""Sparse observations must copy actual inputs and in-place outputs without substituting math."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from integrations.glm53_mxfp4.analyze_sparse_prefill import (
    masked_score_difference,
    selection_difference,
)
from integrations.glm53_mxfp4.drafter_sparse_prefill_probe import SparsePrefillCapture


def test_native_inplace_selection_and_scores_are_copied_without_changing_return(tmp_path: Path) -> None:
    capture = SparsePrefillCapture(tmp_path)
    capture.active_layer = 3
    calls = []
    logits = torch.arange(3973 * 8, dtype=torch.float32).reshape(3973, 8)

    def mqa(value: torch.Tensor) -> torch.Tensor:
        calls.append("mqa")
        return value

    def topk(scores: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor,
             output: torch.Tensor, rows: int, stride0: int, stride1: int, k: int) -> None:
        calls.append("topk")
        output.copy_(scores.topk(k, dim=-1).indices)

    module = SimpleNamespace(fp8_fp4_mqa_logits=mqa)
    ops = SimpleNamespace(top_k_per_row_prefill=topk)
    capture.install_scoring(module, ops)
    result = module.fp8_fp4_mqa_logits(logits)
    assert result is logits
    output = torch.full((3973, 2), -1, dtype=torch.int32)
    assert ops.top_k_per_row_prefill(result, torch.zeros(3973, dtype=torch.int32),
                                    torch.full((3973,), 8, dtype=torch.int32),
                                    output, 3973, 8, 1, 2) is None
    assert calls == ["mqa", "topk"]
    report = capture.report()
    assert report["rows"] == 3973 and not report["arithmetic_replaced"]
    key = report["events"][-1]["tensors"][0]["key"]
    assert torch.equal(capture.saved[key], output)
    output.fill_(-1)
    assert torch.equal(capture.saved[key][0], torch.tensor([7, 6], dtype=torch.int32))
    capture.restore()
    assert module.fp8_fp4_mqa_logits is mqa and ops.top_k_per_row_prefill is topk


def test_probe_excludes_other_operations_and_rejects_incomplete_or_oversized_data(tmp_path: Path) -> None:
    capture = SparsePrefillCapture(tmp_path)
    capture.record("layer3.mlp/output", "result", torch.ones(4))
    assert capture.saved == {}
    with pytest.raises(ValueError, match="Missing native"):
        capture.report()
    capture.max_bytes = 2
    with pytest.raises(MemoryError, match="storage budget"):
        capture.record("sparse_prefill.mqa_logits#0/output", "result", torch.ones(4))


def test_score_comparison_masks_unused_columns_and_preserves_signed_zero() -> None:
    first = np.array([[1, 0, np.nan, 99], [99, 2, 3, 99]], dtype=np.float32)
    second = first.copy()
    starts, ends = np.array([0, 1]), np.array([2, 3])
    second[0, 2:] = -55
    report = masked_score_difference(first, second, starts, ends)
    assert report["exact"] and report["nonfinite_pairs"] == 0
    second[0, 1] = -0.0
    report = masked_score_difference(first, second, starts, ends)
    assert report["first_index"] == [0, 1] and report["different_elements"] == 1
    with pytest.raises(ValueError, match="bounds"):
        masked_score_difference(first, second, starts, np.array([5, 3]))


def test_selection_separates_order_membership_duplicates_and_invalid_values() -> None:
    first = np.array([[0, 1, -1], [0, 1, -1], [0, 1, -1], [0, 1, -1]], dtype=np.int32)
    second = np.array([[1, 0, -1], [0, 2, -1], [0, 0, -1], [0, 4, -2]], dtype=np.int32)
    result = selection_difference(first, second, np.array([2, 3, 3, 4]))
    assert result["different_order_rows"] == 4 and result["different_multiset_rows"] == 3
    assert result["order_only_rows"] == 1 and result["first_different_multiset_row"] == 1
    assert result["inventories"][1]["duplicate_entries"] == 1
    assert result["inventories"][1]["causal_or_extent_violations"] == 1
    assert result["inventories"][1]["non_sentinel_negative"] == 1
