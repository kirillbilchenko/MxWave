"""Independent raw-byte analysis distinguishes numerical drift from metadata/FP8 changes."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))

from analyze_mla_internal import raw_metrics, verify_metrics


def test_bf16_metric_preserves_nan_and_signed_zero_bits() -> None:
    reference = np.array([0x0000, 0x3F80, 0x7FC1, 0x7F80], dtype="<u2").tobytes()
    current = np.array([0x8000, 0x4000, 0x7FC1, 0x7F80], dtype="<u2").tobytes()
    value = raw_metrics(current, reference, "torch.bfloat16", numerical=True)
    assert not value["bytes_exact"] and value["changed_elements_by_bits"] == 2
    assert value["finite_pairs"] == value["nonfinite_pairs"] == 2
    assert value["relative_l2_finite"] == 1.0
    assert value["max_absolute_error_finite"] == 1.0


def test_integer_and_fp8_changes_have_byte_metrics_without_float_interpretation() -> None:
    for dtype in ("torch.uint8", "torch.float8_e4m3fn"):
        value = raw_metrics(bytes([0, 255]), bytes([0, 127]), dtype, numerical=False)
        assert value == {
            "bytes_exact": False,
            "changed_bytes": 1,
            "elements": 2,
            "changed_elements_by_bits": 1,
        }
    with pytest.raises(ValueError, match="BF16/F32"):
        raw_metrics(bytes([0]), bytes([0]), "torch.float8_e4m3fn", numerical=True)


def test_verification_never_tolerates_changed_bit_decisions() -> None:
    raw = np.array([0.0, 1.0, 2.0], dtype="<f4").tobytes()
    metrics = raw_metrics(raw, raw, "torch.float32", numerical=True)
    verify_metrics(metrics, {"metric_kind": "numerical", **metrics})
    with pytest.raises(ValueError, match="bit/count"):
        verify_metrics(metrics, {**metrics, "bytes_exact": False})
    with pytest.raises(ValueError, match="equally sized"):
        raw_metrics(b"x", b"", "torch.uint8", numerical=False)
