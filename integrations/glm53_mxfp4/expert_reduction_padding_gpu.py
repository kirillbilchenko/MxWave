"""Original GPU metadata padding; expert weights and real routing entries are unchanged."""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def pad_dispatch(
    src_tokens, src_experts, src_count, dst_tokens, dst_experts, dst_count,
    ROUTES: tl.constexpr, SOURCE_CAPACITY: tl.constexpr, CAPACITY: tl.constexpr,
    BLOCK: tl.constexpr, QUANTUM: tl.constexpr, WIDTH: tl.constexpr,
):
    """Copy consumed routing entries and append empty expert-zero blocks with invalid row IDs."""
    count = tl.load(src_count)
    multiple = BLOCK * QUANTUM
    padded = tl.cdiv(count, multiple) * multiple
    i = tl.arange(0, WIDTH)
    token = tl.load(src_tokens + i, (i < count) & (i < SOURCE_CAPACITY), other=ROUTES)
    tl.store(dst_tokens + i, token, i < CAPACITY)
    expert = tl.load(src_experts + i, (i < count // BLOCK)
                     & (i < SOURCE_CAPACITY // BLOCK), other=0)
    tl.store(dst_experts + i, expert, i < CAPACITY // BLOCK)
    tl.store(dst_count, padded)
