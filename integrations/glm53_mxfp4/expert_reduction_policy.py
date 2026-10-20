"""Bounded routing padding for full-K tiles in the pinned grouped Marlin kernel."""

from __future__ import annotations

from math import gcd
from typing import Any


def block_quantum(n: int, tile_n: int, sms: int, blocks_per_sm: int) -> int:
    """Return the smallest routing-block multiple giving whole output-tile launch waves."""
    if min(n, tile_n, sms, blocks_per_sm) < 1 or n % tile_n:
        raise ValueError("Require positive, exactly tiled output geometry")
    grid = sms * blocks_per_sm
    return grid // gcd(grid, n // tile_n)


def capacity_for(capacity: int, block: int, quantum: int) -> int:
    """Bound padding for every legal consumed length of the producer's allocation."""
    if min(capacity, block, quantum) < 1 or capacity % block:
        raise ValueError("Require positive capacity in complete routing blocks")
    multiple = block * quantum
    return ((capacity + multiple - 1) // multiple) * multiple


def bind_padding(
    routing: tuple[Any, Any, Any], routes: int, block: int, quantum: int,
) -> tuple[tuple[Any, Any, Any], Any]:
    """Allocate bounded metadata once and bind a stream-ordered, capture-safe padding launch."""
    import torch
    import triton
    from expert_reduction_padding_gpu import pad_dispatch

    tokens, experts, count = routing
    if (routes < 1 or block not in (8, 16, 32, 48, 64)
            or any(v.dtype != torch.int32 or v.ndim != 1 or not v.is_contiguous()
                   or v.device != tokens.device for v in routing)
            or tokens.device.type != "cuda" or count.numel() != 1
            or experts.numel() < tokens.numel() // block):
        raise ValueError("Require valid contiguous native routing metadata on one GPU")
    capacity = capacity_for(tokens.numel(), block, quantum)
    output = (
        torch.empty(capacity, dtype=torch.int32, device=tokens.device),
        torch.empty(capacity // block, dtype=torch.int32, device=tokens.device),
        torch.empty_like(count),
    )

    def launch() -> None:
        pad_dispatch[(1,)](
            *routing, *output, routes, tokens.numel(), capacity, block, quantum,
            triton.next_power_of_2(capacity), num_warps=4,
        )

    return output, launch
