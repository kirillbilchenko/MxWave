"""Logit diagnostics must preserve ties, numerical meaning and raw-bit distinctions."""

from __future__ import annotations

import numpy as np
import pytest

from integrations.glm53_mxfp4.review_drafter_verification import (
    compare_recurrent_inputs,
    difference,
    logit_comparison,
    numeric,
    recurrent_inputs,
)


def test_bf16_bits_and_negative_zero_are_not_integer_measurements() -> None:
    bits = np.array([0x3F80, 0xBF80, 0x8000], dtype=np.uint16)
    values = numeric(bits, "torch.bfloat16")
    assert values.tolist() == [1.0, -1.0, -0.0]
    assert np.signbit(values[-1])
    assert not difference(values[-1:], np.zeros(1, dtype=np.float32))["bitwise_equal"]


def test_logit_review_retains_all_tied_winners_and_zero_kl() -> None:
    a = np.array([27.625, 27.625, 24.0])
    result = logit_comparison(a, a.copy(), [0, 1])
    assert result["a_argmax_ids"] == result["b_argmax_ids"] == [0, 1]
    assert result["forward_kl_a_to_b"] == 0.0
    assert result["bitwise_equal"]


def test_logit_review_reports_changed_winner_and_rejects_nonfinite_rows() -> None:
    result = logit_comparison(np.array([2., 1.]), np.array([1., 2.]), [0, 1])
    assert result["a_argmax_ids"] == [0] and result["b_argmax_ids"] == [1]
    assert result["forward_kl_a_to_b"] == pytest.approx(np.tanh(0.5))
    with pytest.raises(ValueError, match="finite"):
        logit_comparison(np.array([np.inf, 0]), np.array([1, 0]), [0])


def state_frame(slots: list[int], accepted: int, values: list[list[float]]) -> dict:
    """Build a cache with distinguishable actual and unused speculative states."""
    return {
        "meta": {
            "state_layers": {"0": {"num_spec_decodes": 1, "physical_state_slots": slots}},
            "tensor_dtypes": {"before_layer_0_cache_1": "torch.float32"},
        },
        "x": {
            "before_layer_0_meta_num_accepted_tokens": np.array([accepted]),
            "before_layer_0_meta_spec_state_indices_tensor": np.array([slots]),
            "before_layer_0_cache_1": np.array(values, dtype=np.float32),
        },
    }


def test_recurrent_comparison_uses_accepted_column_not_physical_slot_or_unused_rows() -> None:
    a = state_frame([7, 2], 2, [[99, 99], [1, 2]])
    b = state_frame([11, 12], 1, [[1, 2], [88, 88]])
    result = compare_recurrent_inputs(a, b)["0"]
    assert result["bitwise_equal"] and result["max_abs"] == 0
    assert result["accepted_columns"] == [1, 0]
    assert result["physical_slots"] == [2, 11]


def test_recurrent_comparison_rejects_invalid_accepted_state() -> None:
    with pytest.raises(ValueError, match="accepted-state column"):
        recurrent_inputs(state_frame([7, 2], 3, [[1], [2]]))
