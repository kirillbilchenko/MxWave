"""Contract checks for exact arbitrary-byte palettes and raw fallback."""
from __future__ import annotations

import numpy as np
import pytest

from integrations.glm53_nvfp4.codec import decode, encode


@pytest.mark.parametrize('symbols', [1, 2, 16, 17, 31, 32, 33, 256])
def test_all_selector_positions_and_symbol_boundary(symbols: int) -> None:
    """Cover every symbol in every vector lane, including low/high-plane transitions."""
    palette = np.linspace(0, 255, symbols, dtype=np.uint8)
    codes = (np.arange(16 * 1024).reshape(-1, 16) + np.arange(16)) % symbols
    raw = palette[codes].reshape(-1)
    result = encode(raw)
    np.testing.assert_array_equal(decode(result), raw)
    assert bool(result.raw.size) == (symbols > 32)
    assert result.nbytes == (raw.size if symbols > 32 else raw.size * 5 // 8 + 32)


def test_strided_source_and_owned_fallback() -> None:
    """Encoding owns its payload even when the caller later changes a strided source."""
    raw = np.arange(256, dtype=np.uint8).reshape(16, 16).T
    expected = raw.copy().reshape(-1)
    result = encode(raw)
    raw[:] = 0
    np.testing.assert_array_equal(decode(result), expected)


@pytest.mark.parametrize('base', [0, 112, 224, 230, 255])
def test_affine_byte_range_and_no_carry(base: int) -> None:
    """Include the top byte boundary and gaps without requantizing scale values."""
    width = min(32, 256 - base)
    raw = ((np.arange(16 * 32) * 7) % width + base).astype(np.uint8)
    packed = encode(raw, affine=True)
    assert not packed.raw.size
    np.testing.assert_array_equal(decode(packed), raw)
    assert int(raw.max()) <= 255


def test_affine_rejects_wide_sparse_palette() -> None:
    """Few symbols alone do not authorize an affine byte-offset decoder."""
    raw = np.tile(np.array([0, 255], dtype=np.uint8), 16)
    assert not encode(raw).raw.size
    packed = encode(raw, affine=True)
    assert packed.raw.size == raw.size
    np.testing.assert_array_equal(decode(packed), raw)


@pytest.mark.parametrize('raw', [np.empty(0, np.uint8), np.zeros(17, np.uint8),
                                 np.zeros(16, np.int16)])
def test_invalid_input_rejected(raw: np.ndarray) -> None:
    """Reject incomplete vectors and accidental numeric reinterpretation."""
    with pytest.raises(ValueError):
        encode(raw)
