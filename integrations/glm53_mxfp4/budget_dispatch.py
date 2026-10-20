"""Extend only the integer dispatch shape guard for the bounded 4,096-row experiment."""
from __future__ import annotations

import torch


def canonicalize(
    tokens: torch.Tensor, experts: torch.Tensor, padded: torch.Tensor, *,
    num_tokens: int, num_experts: int, block: int,
) -> None:
    """Sort each consumed expert segment on stream; require ascending bins and unique top-k ids.

    These producer invariants are checked in component qualification, not by
    synchronizing GPU metadata inside each model call. EP mappings, redundant
    experts, more than 4096 tokens and arbitrary dispatch producers are outside
    this experiment's contract.
    """
    values = (tokens, experts, padded)
    if (not 0 <= num_tokens <= 4096 or num_experts < 1
            or block not in (8, 16, 32, 48, 64)):
        raise ValueError("Unsupported stable-dispatch geometry")
    if (any(v.dtype != torch.int32 or v.ndim != 1 or not v.is_contiguous() for v in values)
            or tokens.device.type != "cuda" or any(v.device != tokens.device for v in values)
            or padded.numel() != 1):
        raise ValueError("Require contiguous int32 dispatch metadata on one CUDA device")
    if num_tokens <= 1:
        return
    import triton
    from stable_dispatch_gpu import _canonicalize

    capacity = triton.next_power_of_2(max(32, triton.cdiv(num_tokens, block) * block))
    _canonicalize[(num_experts,)](
        tokens, experts, padded, block, capacity, num_warps=4,
    )


def install() -> None:
    """Keep the pinned producer and Triton implementation; extend the checked row bound."""
    import stable_dispatch

    stable_dispatch.canonicalize = canonicalize
