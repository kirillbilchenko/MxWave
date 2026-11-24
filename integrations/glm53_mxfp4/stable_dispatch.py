"""Opt-in stable dispatch for the pinned TP-only, unique-top-k Marlin experiment."""

from __future__ import annotations

import functools
import hashlib
from pathlib import Path
from typing import Any

import torch
from quality_worker import QualityWorker

_INSTALLED = False
_CALLS: dict[int, int] = {}


def canonicalize(
    tokens: torch.Tensor, experts: torch.Tensor, padded: torch.Tensor, *,
    num_tokens: int, num_experts: int, block: int,
) -> None:
    """Sort each consumed expert segment on stream; require ascending bins and unique top-k ids.

    These producer invariants are checked in component qualification, not by
    synchronizing GPU metadata inside each model call. EP mappings, redundant
    experts, more than 2048 tokens and arbitrary dispatch producers are outside
    this experiment's contract.
    """
    values = (tokens, experts, padded)
    if (not 0 <= num_tokens <= 2048 or num_experts < 1
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
    """Wrap only the pinned Marlin producer; the caller must explicitly enable this experiment."""
    global _INSTALLED
    if _INSTALLED:
        return
    from vllm.model_executor.layers.fused_moe.experts import marlin_moe

    original = marlin_moe.moe_align_block_size

    @functools.wraps(original)
    def align(
        topk_ids: torch.Tensor, block_size: int, num_experts: int,
        expert_map: torch.Tensor | None = None, **kwargs: Any,
    ) -> Any:
        if expert_map is not None or num_experts != 288:
            raise ValueError("Stable-dispatch qualification covers TP-only GLM with 288 experts")
        result = original(topk_ids, block_size, num_experts, expert_map, **kwargs)
        count = int(topk_ids.shape[0])
        if count > 1:
            canonicalize(*result, num_tokens=count, num_experts=num_experts, block=block_size)
            _CALLS[count] = _CALLS.get(count, 0) + 1
        return result

    marlin_moe.moe_align_block_size = align
    _INSTALLED = True


def report() -> dict[str, Any]:
    """Return plain proof metadata; never copy GPU routing buffers into a timed model path."""
    return {
        "installed": _INSTALLED,
        "calls_by_tokens": {str(k): v for k, v in sorted(_CALLS.items())},
        "source_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("stable_dispatch.py", "stable_dispatch_gpu.py")
        },
        "scope": "Pinned TP-only Marlin, 288 experts, unique top-k, at most 2048 tokens",
    }


class StableWorker(QualityWorker):
    """Retain every quality/source check and attest that the opt-in dispatch path ran."""

    def mxwave_quality_source_report(self) -> dict[str, Any]:
        """Attach actual dispatch-call coverage to the unchanged full source-scale proof."""
        result = QualityWorker.mxwave_quality_source_report(self)
        result["stable_dispatch"] = report()
        if not result["stable_dispatch"]["installed"]:
            raise RuntimeError("Requested stable-dispatch qualification was not installed")
        return result
