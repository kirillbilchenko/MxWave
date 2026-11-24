"""Exact five-bit palette storage for post-Marlin NVFP4 scale bytes."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class EncodedPart:
    """Aligned low/high bit planes, or exact raw fallback; no floating conversion."""

    low: NDArray[np.uint32]
    high: NDArray[np.uint16]
    palette: NDArray[np.uint8]
    raw: NDArray[np.uint8]
    size: int

    @property
    def nbytes(self) -> int:
        """Return encoded payload bytes, excluding the separate 32-byte descriptor."""
        return self.low.nbytes + self.high.nbytes + self.palette.nbytes + self.raw.nbytes


def encode(raw: NDArray[np.uint8], *, affine: bool = False) -> EncodedPart:
    """Encode a positive multiple of 16 bytes; use raw fallback above 32 symbols."""
    if raw.dtype != np.uint8 or raw.size == 0 or raw.size % 16:
        raise ValueError('Expected nonempty uint8 scale bytes in whole 16-byte vectors')
    values = np.ascontiguousarray(raw).reshape(-1)
    palette, inverse = np.unique(values, return_inverse=True)
    if palette.size > 32 or (affine and int(palette[-1]) - int(palette[0]) >= 32):
        return EncodedPart(np.empty((0, 2), np.uint32), np.empty(0, np.uint16),
                           np.empty(0, np.uint8), values.copy(), values.size)
    codes = ((values.astype(np.uint32) - int(palette[0])) if affine
             else inverse.astype(np.uint32)).reshape(-1, 16)
    low = np.bitwise_or.reduce(
        (codes.reshape(-1, 2, 8) & 15) << (4 * np.arange(8, dtype=np.uint32)), axis=-1)
    high = np.bitwise_or.reduce((codes >> 4) << np.arange(16, dtype=np.uint32),
                               axis=-1).astype(np.uint16)
    padded = np.zeros(32, dtype=np.uint8)
    if affine:
        padded[:] = np.arange(32, dtype=np.uint8) + palette[0]
    else:
        padded[:palette.size] = palette
    return EncodedPart(low, high, padded, np.empty(0, np.uint8), values.size)


def decode(part: EncodedPart) -> NDArray[np.uint8]:
    """Recover bytes independently from the stored bit planes and palette."""
    if part.raw.size:
        return part.raw.copy()
    codes = ((part.low[:, :, None] >> (4 * np.arange(8, dtype=np.uint32))) & 15)
    codes = codes.reshape(-1, 16)
    codes |= ((part.high[:, None].astype(np.uint32) >> np.arange(16, dtype=np.uint32)) & 1) << 4
    return part.palette[codes].reshape(-1)
