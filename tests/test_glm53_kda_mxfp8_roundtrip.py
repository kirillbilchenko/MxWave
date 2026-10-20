"""Verify the prospective weight-rounding recipe before any model mutation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

PATH = Path(__file__).parents[1] / "integrations/glm53_mxfp4/kda_mxfp8_roundtrip.py"
SPEC = importlib.util.spec_from_file_location("kda_mxfp8_roundtrip", PATH)
assert SPEC is not None and SPEC.loader is not None
codec = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(codec)


def test_independent_codebook_and_scale_boundaries() -> None:
    """Check random BF16, zero, ties, subnormals and upward scale boundaries."""
    generator = torch.Generator().manual_seed(20261020)
    random = torch.randn((128, 256), generator=generator).bfloat16()
    special = torch.tensor(
        [0, -0.0, 1, -1, 1.0625, 1.1875, 448, 450,
         0.0009765625, 0.001953125, 0.0029296875, 0.00390625,
         224, 226, 896, 900] * 2, dtype=torch.bfloat16,
    ).reshape(1, 32)
    values = [
        random,
        special,
        torch.zeros((2, 32), dtype=torch.bfloat16),
        torch.ldexp(random.float(), torch.tensor(-40)).bfloat16(),
        torch.ldexp(random.float(), torch.tensor(40)).bfloat16(),
    ]
    for source in values:
        actual = codec.roundtrip(source)
        expected = codec.numpy_reference(source.float().numpy())
        for left, right in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(left.cpu().numpy(), right)
        assert torch.equal(actual[0], actual[0].bfloat16().float())
        assert torch.equal(codec.roundtrip(actual[0])[0], actual[0])


def test_blocks_and_rows_do_not_share_scales() -> None:
    """Quantizing TP-aligned slices must match whole-matrix quantization."""
    source = torch.arange(4 * 128, dtype=torch.float32).reshape(4, 128).bfloat16()
    source[:, 64:] *= 1e-3
    full = codec.roundtrip(source)
    for start, stop in ((0, 64), (64, 128)):
        partial = codec.roundtrip(source[:, start:stop])
        torch.testing.assert_close(partial[0], full[0][:, start:stop], rtol=0, atol=0)
        assert torch.equal(partial[1], full[1][:, start:stop])
        assert torch.equal(partial[2], full[2][:, start // 32:stop // 32])


@pytest.mark.parametrize("source", [
    torch.ones(32),
    torch.ones(1, 33),
    torch.full((1, 32), float("nan")),
    torch.full((1, 32), float("inf")),
])
def test_invalid_source_is_rejected(source: torch.Tensor) -> None:
    """Do not silently reshape or saturate invalid inputs."""
    with pytest.raises(ValueError):
        codec.roundtrip(source)
