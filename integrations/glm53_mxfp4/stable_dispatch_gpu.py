"""In-place deterministic token ordering inside already selected expert segments."""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _sort_segment(TOKENS, start, count, WIDTH: tl.constexpr):
    offsets = tl.arange(0, WIDTH)
    values = tl.load(TOKENS + start + offsets, offsets < count, other=2147483647)
    ordered = tl.sort(values, descending=False)
    tl.store(TOKENS + start + offsets, ordered, offsets < count)


@triton.jit
def _canonicalize(TOKENS, EXPERTS, PADDED, BLOCK: tl.constexpr, CAPACITY: tl.constexpr):
    expert = tl.program_id(0)
    blocks = tl.load(PADDED) // BLOCK
    lo = tl.full((), 0, tl.int32)
    hi = blocks
    while lo < hi:
        mid = (lo + hi) // 2
        value = tl.load(EXPERTS + mid)
        before = value < expert
        lo = tl.where(before, mid + 1, lo)
        hi = tl.where(before, hi, mid)
    first = lo
    hi = blocks
    while lo < hi:
        mid = (lo + hi) // 2
        value = tl.load(EXPERTS + mid)
        before = value <= expert
        lo = tl.where(before, mid + 1, lo)
        hi = tl.where(before, hi, mid)
    start = first * BLOCK
    count = (lo - first) * BLOCK
    tl.device_assert(count <= CAPACITY, "Expert segment exceeds unique-top-k token bound")
    if count > 0:
        if CAPACITY <= 32:
            _sort_segment(TOKENS, start, count, 32)
        elif CAPACITY <= 128:
            if count <= 32:
                _sort_segment(TOKENS, start, count, 32)
            else:
                _sort_segment(TOKENS, start, count, 128)
        elif CAPACITY <= 512:
            if count <= 32:
                _sort_segment(TOKENS, start, count, 32)
            elif count <= 128:
                _sort_segment(TOKENS, start, count, 128)
            else:
                _sort_segment(TOKENS, start, count, 512)
        else:
            if count <= 32:
                _sort_segment(TOKENS, start, count, 32)
            elif count <= 128:
                _sort_segment(TOKENS, start, count, 128)
            elif count <= 512:
                _sort_segment(TOKENS, start, count, 512)
            else:
                _sort_segment(TOKENS, start, count, CAPACITY)
