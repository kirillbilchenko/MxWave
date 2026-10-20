"""Guard token-versus-route comparisons and preserve timing-control failures."""

from __future__ import annotations

import numpy as np
import pytest

from integrations.glm53_mxfp4.review_expert_wide import changed_prefix, timing_pair


def test_wider_layout_ignores_unshared_rows_but_detects_one_route_ulp() -> None:
    a = np.full((8 * 8, 16), 0x3F80, dtype=np.uint16)
    b = np.full((128 * 8, 16), 0x4000, dtype=np.uint16)
    b[:8 * 8] = a
    assert changed_prefix(a, b, 8, 128, 8) == 0
    b[7 * 8 + 6, 15] += 1
    assert changed_prefix(a, b, 8, 128, 7) == 0
    assert changed_prefix(a, b, 8, 128, 8) == 1
    with pytest.raises(ValueError):
        changed_prefix(a, b[:, :8], 8, 128, 8)


def test_numerically_equal_signed_zero_is_not_bitwise_equal() -> None:
    assert changed_prefix(np.zeros((8, 2), dtype=np.uint16),
                          np.full((16, 2), 0x8000, dtype=np.uint16), 8, 16, 8) == 16


def test_fast_ratio_cannot_override_unstable_identical_control() -> None:
    result = timing_pair([100.0] * 4 + [103.0] * 4, [90.0] * 8)
    assert result['paired_median_overhead'] < 0 and not result['control_pass']
    assert timing_pair([100.0] * 8, [101.0] * 8)['control_pass']
    with pytest.raises(ValueError):
        timing_pair([100.0] * 7, [90.0] * 7)
    with pytest.raises(ValueError):
        timing_pair([100.0] * 8, [float('nan')] * 8)
