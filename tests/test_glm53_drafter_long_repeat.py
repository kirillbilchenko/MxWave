"""Repeat diagnostics must compare shared accepted history across different draft grouping."""

from __future__ import annotations

import copy

import numpy as np

from integrations.glm53_mxfp4.review_drafter_long_repeat import (
    LAYERS,
    common_prefix,
    compare_runs,
    difference,
)


def test_bit_comparison_preserves_signed_zero_and_identical_nan_payloads() -> None:
    first = np.array([0, 0x7FC00001], dtype=np.uint32).view(np.float32)
    second = first.copy()
    assert difference(first, second)["exact"]
    second[0] = -0.0
    assert difference(first, second)["first_index"] == [0]


def test_response_branch_bounds_the_feature_comparison() -> None:
    assert common_prefix([1, 2, 3], [1, 9, 3]) == 1
    assert common_prefix([1, 2], [1, 2]) == 2
    values = {f"aux_layer_{layer}": np.arange(6 * 4, dtype=np.uint16).reshape(6, 4)
              for layer in LAYERS}
    frame = {f"aux_layer_{layer}": value[:2] for layer, value in zip(LAYERS, values.values(), strict=True)}
    for key in ("draft_query_ids", "draft_query_positions", "request_seed", "anchor_before_propose",
                "combined_target_hidden", "draft_hidden", "draft_ids", "draft_values",
                "draft_scores", "returned_proposal"):
        frame[key] = np.array([1, 2], dtype=np.uint16)
    first = {"segments": [(0, 6, values)], "frames": [frame], "needed": 6,
             "response": [10, 11, 12, 13, 14],
             "logits": {i: np.array([i, i + 1], dtype=np.uint16) for i in range(6)}}
    second = copy.deepcopy(first)
    second["response"][2] = 99
    # Same history through two response tokens: only positions 0..3 may be compared.
    for value in second["segments"][0][2].values():
        value[4:] += 100
    original = second["segments"][0][2]
    second["segments"] = [(0, 2, {k: v[:2] for k, v in original.items()}),
                          (2, 6, {k: v[2:] for k, v in original.items()})]
    report = compare_runs(first, second, 2)
    assert report["equal_history_feature_rows"] == 4
    assert all(row["different_elements"] == 0 for row in report["shared_history_features"])
    second["segments"][1][2]["aux_layer_5"][1, 0] += 1
    report = compare_runs(first, second, 2)
    assert report["shared_history_features"][0]["first_different_input_position"] == 3
