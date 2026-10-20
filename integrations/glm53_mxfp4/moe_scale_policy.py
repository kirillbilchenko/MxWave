"""Validate complete scale ranges before selecting an exact shared-memory layout."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

STAGE_BASE_SHIFT = 32
STAGE_NIBBLES = 1 << 40
STAGE_VERIFIED = 1 << 41


def sparse_stage_descriptor(
    raw: NDArray[np.uint8], palette_descriptor: int,
) -> tuple[int, int | None]:
    """Bind a nibble base or U8 fallback to all source bytes, including exceptions.

    The original palette and selector format are unchanged. Upper descriptor bits
    carry the verified staging policy; older consumers ignore these upper bits.
    No source value is clipped and no expanded global-memory buffer is retained.
    """
    if (raw.dtype != np.uint8 or raw.ndim != 1 or not raw.flags.c_contiguous
            or not raw.size or raw.size % 16):
        raise ValueError("Require contiguous, nonempty U8 whole 16-byte vectors")
    if not 0 <= palette_descriptor < 1 << 26:
        raise ValueError("Unexpected or already extended palette descriptor")
    lower, upper = int(raw.min()), int(raw.max())
    base = (lower & ~15) if lower // 16 == upper // 16 else None
    descriptor = palette_descriptor | STAGE_VERIFIED
    if base is not None:
        descriptor |= STAGE_NIBBLES | base << STAGE_BASE_SHIFT
    return descriptor, base
