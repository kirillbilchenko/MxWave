"""Independent source-decoding contract for the installed-kernel diagnostic."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


@pytest.fixture
def decode():
    """Load the CPU decoder without importing any serving engine module."""
    path = Path(__file__).parents[1] / "integrations/glm53_mxfp4/diagnose_mxfp8.py"
    spec = importlib.util.spec_from_file_location("glm53_mxfp8_reference_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.decode


def test_e8m0_bias_and_signed_e4m3_values(decode):
    # E4M3 bytes 0x38 and 0xc0 encode exactly +1 and -2.
    values = torch.tensor([0x38, 0xc0] * 32, dtype=torch.uint8).view(torch.float8_e4m3fn)
    actual = decode(values.reshape(1, 64), torch.tensor([[126, 129]], dtype=torch.uint8))
    assert torch.equal(actual, torch.tensor([[.5, -1] * 16 + [4, -8] * 16]))


def test_every_finite_e8m0_exponent_restores_the_expected_power_of_two(decode):
    weight = torch.ones((255, 32)).to(torch.float8_e4m3fn)
    scales = torch.arange(255, dtype=torch.uint8).reshape(255, 1)
    actual = decode(weight, scales)
    expected = torch.tensor([2.0 ** (i - 127) for i in range(255)])
    assert torch.equal(actual[:, 0], expected)
    assert torch.isfinite(actual).all()


def test_reserved_scale_and_nonfinite_weight_fail_closed(decode):
    weight = torch.ones((1, 32)).to(torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="Expected finite"):
        decode(weight, torch.tensor([[255]], dtype=torch.uint8))
    weight.view(torch.uint8)[0, 0] = 127  # E4M3 NaN.
    with pytest.raises(ValueError, match="non-finite"):
        decode(weight, torch.tensor([[127]], dtype=torch.uint8))


def test_fp4_reference_preserves_nibble_order_sign_and_scale(pytestconfig, monkeypatch):
    directory = Path(__file__).parents[1] / "integrations/glm53_mxfp4"
    monkeypatch.syspath_prepend(str(directory))
    path = directory / "diagnose_moe.py"
    spec = importlib.util.spec_from_file_location("glm53_fp4_reference_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    packed = torch.tensor([[0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe] * 2],
                          dtype=torch.uint8)
    result = module.decode(packed, torch.tensor([[128]], dtype=torch.uint8))
    expected = torch.tensor([[0, 1, 2, 3, 4, 6, 8, 12, -0., -1, -2, -3, -4, -6, -8, -12] * 2])
    assert torch.equal(result, expected)
    assert torch.signbit(result[0, 8])
    with pytest.raises(ValueError, match="finite"):
        module.decode(packed, torch.tensor([[255]], dtype=torch.uint8))
