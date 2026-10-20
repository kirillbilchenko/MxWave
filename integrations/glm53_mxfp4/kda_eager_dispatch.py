"""Experimental eager-only host dispatch; captured and compiled work retains the qualified op."""

from __future__ import annotations

from typing import Any

import kda_runtime
import torch
from kda_direct_dispatch import linear as direct_linear

_QUALIFIED_APPLY = kda_runtime.apply
_MODE = "qualified"


def linear(
    x: torch.Tensor, buffers: list[torch.Tensor], pool: torch.Tensor,
    m: int, k: int, common: int, rare: int, exponent: int, rows: int,
) -> torch.Tensor:
    """Use one dispatcher eagerly while leaving compilation and every new capture unchanged."""
    if torch.compiler.is_compiling():
        return kda_runtime.kda_linear(x, buffers, pool, m, k, common, rare, exponent, rows)
    if x.device.type == "cuda":
        with torch.cuda.device(x.device):
            if torch.cuda.is_current_stream_capturing():
                return kda_runtime.kda_linear(x, buffers, pool, m, k, common, rare, exponent, rows)
    return direct_linear(x, buffers, pool, m, k, common, rare, exponent, rows)


def _eager_apply(layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    from mxwave_zipserv import runtime

    buffers = [getattr(layer, "mxwave_kda_" + name) for name in runtime.BUFFER_NAMES]
    return linear(x, buffers, layer.mxwave_kda_pool, *layer.mxwave_kda_metadata)


def select(mode: str) -> dict[str, Any]:
    """Switch only between closed benchmark groups; preserve weights, pools and existing graphs."""
    global _MODE
    if mode not in ("qualified", "eager-direct"):
        raise ValueError("Unknown KDA dispatch mode")
    if kda_runtime.FUSED_MAX_TOKENS != 8:
        raise ValueError("Eager dispatch comparison requires the qualified eight-token policy")
    if kda_runtime.apply not in (_QUALIFIED_APPLY, _eager_apply):
        raise RuntimeError("Another integration replaced the KDA apply function")
    if not torch.cuda.is_initialized():
        raise RuntimeError("Dispatch switching belongs to an initialized, idle GPU worker")
    torch.cuda.synchronize()
    kda_runtime.apply = _QUALIFIED_APPLY if mode == "qualified" else _eager_apply
    _MODE = mode
    return status()


def status() -> dict[str, Any]:
    """Report plain RPC values without inferring end-to-end speed or quality qualification."""
    return {"mode": _MODE, "compiled_and_captured_operation": "mxwave_glm53::kda_linear",
            "existing_graphs_recaptured": False, "weights_or_pools_replaced": False,
            "cuda_allocated_bytes": int(torch.cuda.memory_allocated())}
