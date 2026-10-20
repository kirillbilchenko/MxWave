"""Opt-in deterministic token order for TP-only Marlin MoE qualification."""
from __future__ import annotations

import functools
from typing import Any

import torch

ENABLED = True


def canonicalize(sorted_ids: torch.Tensor, expert_ids: torch.Tensor,
                 padded_count: torch.Tensor, *, token_count: int, block_size: int,
                 num_experts: int) -> torch.Tensor:
    """Sort flattened token IDs within ascending expert segments, preserving padding."""
    positions = torch.arange(sorted_ids.numel(), device=sorted_ids.device)
    owners = expert_ids[torch.div(positions, block_size, rounding_mode="floor")]
    base = token_count + 1
    keys = owners.to(torch.int64) * base + sorted_ids.to(torch.int64)
    tail_key = (num_experts + 1) * base + token_count
    keys = torch.where(positions < padded_count[0], keys, tail_key)
    ordered = torch.sort(keys).values
    return torch.remainder(ordered, base).to(sorted_ids.dtype)


def install() -> None:
    """Wrap only Marlin alignment in disposable TP-only diagnostic workers."""
    from vllm.model_executor.layers.fused_moe.experts import marlin_moe

    original = marlin_moe.moe_align_block_size
    if getattr(original, "_mxwave_stable_alignment", False):
        return

    @functools.wraps(original)
    def aligned(topk_ids: torch.Tensor, block_size: int, num_experts: int,
                expert_map: torch.Tensor | None = None, **kwargs: Any) -> Any:
        if expert_map is not None:
            raise ValueError("Deterministic diagnostic supports TP-only expert ordering")
        result = original(topk_ids, block_size, num_experts, expert_map, **kwargs)
        if not ENABLED:
            return result
        sorted_ids, expert_ids, count = result
        ordered = canonicalize(sorted_ids, expert_ids, count, token_count=topk_ids.numel(),
                               block_size=block_size, num_experts=num_experts)
        return ordered, expert_ids, count

    aligned._mxwave_stable_alignment = True
    marlin_moe.moe_align_block_size = aligned
