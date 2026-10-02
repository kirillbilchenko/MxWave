"""Tests for lossless MxWave MXFP4 GGUF export primitives."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import Tensor

from mxwave.gguf import (
    Qwen35LinearAttentionConfig,
    repack_mxfp4_blocks,
    transform_qwen35_mxfp4,
)


def _pack_codes(codes: Tensor) -> Tensor:
    pairs = codes.reshape(*codes.shape[:-1], codes.shape[-1] // 2, 2)
    return (pairs[..., 0] | (pairs[..., 1] << 4)).to(torch.uint8)


def _unpack_codes(packed: Tensor) -> Tensor:
    return torch.stack((packed & 0x0F, (packed >> 4) & 0x0F), dim=-1).reshape(
        *packed.shape[:-1], packed.shape[-1] * 2
    )


def test_repack_mxfp4_blocks_matches_ggml_layout_byte_for_byte() -> None:
    codes = torch.tensor([list(range(16)) + list(reversed(range(16)))], dtype=torch.uint8)
    packed = _pack_codes(codes)
    scale = torch.tensor([[137]], dtype=torch.uint8)

    result = repack_mxfp4_blocks(packed, scale)

    expected_codes = [int(codes[0, index] | (codes[0, index + 16] << 4)) for index in range(16)]
    expected = np.array([[137, *expected_codes]], dtype=np.uint8)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    ("packed", "scale", "error"),
    [
        (
            torch.zeros((1, 16), dtype=torch.int8),
            torch.zeros((1, 1), dtype=torch.uint8),
            TypeError,
        ),
        (
            torch.zeros(16, dtype=torch.uint8),
            torch.zeros((1, 1), dtype=torch.uint8),
            ValueError,
        ),
        (
            torch.zeros((1, 15), dtype=torch.uint8),
            torch.zeros((1, 1), dtype=torch.uint8),
            ValueError,
        ),
        (
            torch.zeros((2, 16), dtype=torch.uint8),
            torch.zeros((2, 2), dtype=torch.uint8),
            ValueError,
        ),
    ],
)
def test_repack_mxfp4_blocks_rejects_invalid_inputs(
    packed: Tensor,
    scale: Tensor,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        repack_mxfp4_blocks(packed, scale)


def test_qwen35_qkv_reorders_only_value_rows() -> None:
    config = Qwen35LinearAttentionConfig(2, 4, 2, 2)
    packed = torch.arange(16, dtype=torch.uint8).reshape(16, 1).expand(-1, 16).clone()
    scale = torch.arange(16, dtype=torch.uint8).reshape(16, 1)

    transformed, transformed_scale = transform_qwen35_mxfp4(
        "model.layers.0.linear_attn.in_proj_qkv.weight",
        packed,
        scale,
        config,
    )

    expected_rows = torch.tensor([*range(8), 8, 9, 12, 13, 10, 11, 14, 15])
    torch.testing.assert_close(transformed[:, 0], expected_rows.to(torch.uint8))
    torch.testing.assert_close(transformed_scale[:, 0], expected_rows.to(torch.uint8))


@pytest.mark.parametrize(
    ("suffix", "value_head_dim", "expected_rows"),
    [
        ("in_proj_z", 2, [0, 1, 4, 5, 2, 3, 6, 7]),
        ("in_proj_a", 1, [0, 2, 1, 3]),
        ("in_proj_b", 1, [0, 2, 1, 3]),
    ],
)
def test_qwen35_auxiliary_projections_reorder_rows(
    suffix: str,
    value_head_dim: int,
    expected_rows: list[int],
) -> None:
    config = Qwen35LinearAttentionConfig(2, 4, 2, value_head_dim)
    rows = len(expected_rows)
    packed = torch.arange(rows, dtype=torch.uint8).reshape(rows, 1).expand(-1, 16).clone()
    scale = torch.arange(rows, dtype=torch.uint8).reshape(rows, 1)

    transformed, transformed_scale = transform_qwen35_mxfp4(
        f"model.layers.0.linear_attn.{suffix}.weight",
        packed,
        scale,
        config,
    )

    expected = torch.tensor(expected_rows, dtype=torch.uint8)
    torch.testing.assert_close(transformed[:, 0], expected)
    torch.testing.assert_close(transformed_scale[:, 0], expected)


def test_qwen35_out_projection_reorders_aligned_mxfp4_blocks() -> None:
    config = Qwen35LinearAttentionConfig(2, 4, 8, 32)
    codes = torch.cat(
        tuple(torch.full((32,), value, dtype=torch.uint8) for value in (1, 2, 3, 4))
    ).reshape(1, 128)
    packed = _pack_codes(codes)
    scale = torch.tensor([[10, 20, 30, 40]], dtype=torch.uint8)

    transformed, transformed_scale = transform_qwen35_mxfp4(
        "model.layers.0.linear_attn.out_proj.weight",
        packed,
        scale,
        config,
    )

    transformed_codes = _unpack_codes(transformed).reshape(4, 32)
    assert transformed_codes[:, 0].tolist() == [1, 3, 2, 4]
    assert transformed_scale.tolist() == [[10, 30, 20, 40]]


def test_qwen35_out_projection_rejects_unaligned_head_permutation() -> None:
    config = Qwen35LinearAttentionConfig(2, 4, 8, 16)
    packed = torch.zeros((2, 32), dtype=torch.uint8)
    scale = torch.zeros((2, 2), dtype=torch.uint8)

    with pytest.raises(ValueError, match="not aligned"):
        transform_qwen35_mxfp4(
            "model.layers.0.linear_attn.out_proj.weight",
            packed,
            scale,
            config,
        )
