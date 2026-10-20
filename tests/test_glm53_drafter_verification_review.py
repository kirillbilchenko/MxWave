"""Logit diagnostics must preserve ties, numerical meaning and raw-bit distinctions."""

from __future__ import annotations

import numpy as np
import pytest

from integrations.glm53_mxfp4.review_drafter_verification import (
    difference,
    logit_comparison,
    numeric,
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
