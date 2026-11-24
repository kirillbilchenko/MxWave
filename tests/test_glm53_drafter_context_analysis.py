"""Check proposal diagnosis against conditional paths and missing candidates."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SOURCE = Path(__file__).parents[1] / "integrations/glm53_mxfp4/drafter_context_analysis.py"
spec = importlib.util.spec_from_file_location("drafter_context_analysis", SOURCE)
assert spec is not None and spec.loader is not None
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_proposal_diagnosis_distinguishes_selector_error_from_missing_candidate() -> None:
    ids = np.arange(112).reshape(7, 16)
    unary = np.zeros((7, 16), dtype=np.float32)
    unary[:, 0] = 2
    edges = np.broadcast_to(unary[:, None, :], (7, 16, 16)).copy()
    edges[6, 0, 1] = 3
    proposed = ids[:, 0].tolist()
    proposed[6] = int(ids[6, 1])
    expected = ids[:, 0].tolist()
    data = {
        "draft_ids": ids, "draft_values": unary, "draft_scores": edges,
        "draft_proposal": np.array(proposed),
    }
    result = analysis.proposal_review(data, expected)
    assert result["known_common_prefix_matches"] == 6
    assert result["positions"][6]["expected_candidate_rank"] == 1
    assert result["positions"][6]["selected_minus_expected_score"] == 1.0
    assert result["selector_strength_sensitivity"]["0.0"]["known_common_prefix_matches"] == 7
    expected[6] = 1000
    result = analysis.proposal_review(data, expected)
    assert result["positions"][6]["expected_candidate_rank"] is None
    assert result["positions"][6]["selected_minus_expected_score"] is None
    assert all(row["known_common_prefix_matches"] == 6
               for row in result["selector_strength_sensitivity"].values())


def test_hidden_difference_retains_absolute_and_relative_scale() -> None:
    x = np.array([1, 2], dtype=np.float32)
    result = analysis.difference(2 * x, x)
    assert result["unequal_elements"] == 2
    assert result["relative_l2_to_modelopt"] == pytest.approx(1.0)
    assert result["cosine"] == pytest.approx(1.0)
    assert result["max_abs"] == 2
