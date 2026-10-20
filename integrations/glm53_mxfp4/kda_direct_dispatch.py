"""Opt-in single-dispatch KDA prototype using the unchanged, hash-pinned native BF16 ABI."""

from __future__ import annotations

from typing import Any

import torch
from kda_runtime import FUSED_MAX_TOKENS, kda_linear, validate_runtime

_LIB = torch.library.Library("mxwave_glm53", "FRAGMENT")
_LIB.define(
    "kda_fused_direct(Tensor x, Tensor[] buffers, int m, int k, "
    "int common, int rare, int exponent, int rows) -> Tensor"
)


def _validate(x: torch.Tensor, m: int, k: int, rows: int) -> int:
    if (
        x.device.type != "cuda" or x.dtype != torch.bfloat16 or x.ndim < 1
        or x.shape[-1] != k or x.requires_grad
        or m <= 0 or k <= 0 or m % 64 or k % 64 or m * k > 2**31 - 4097
        or not 0 < rows <= m or x.numel() // k > 8
    ):
        raise ValueError("Require audited inference-only BF16 CUDA KDA geometry at up to eight tokens")
    return x.numel() // k


def _cuda_linear(
    x: torch.Tensor, buffers: list[torch.Tensor],
    m: int, k: int, common: int, rare: int, exponent: int, rows: int,
) -> torch.Tensor:
    from mxwave_zipserv import runtime

    if FUSED_MAX_TOKENS != 8:
        raise ValueError("Direct KDA prototype requires the qualified eight-token fused policy")
    tokens = _validate(x, m, k, rows)
    runtime._validate_buffers(buffers, x.device)
    validate_runtime()
    if tokens == 0:
        return x.new_empty((*x.shape[:-1], rows))
    with torch.cuda.device(x.device):
        stream = torch.cuda.current_stream(x.device)
        # The public native call bypasses ATen; explicitly retain input allocation
        # lifetimes when a caller uses a non-default stream and releases inputs.
        x.record_stream(stream)
        for buffer in buffers:
            buffer.record_stream(stream)
        flat = x.reshape(tokens, k).contiguous()
        if tokens != 8:
            padded = x.new_zeros((8, k))
            padded[:tokens].copy_(flat)
            flat = padded
        output = x.new_empty((8, m))
        runtime._check(runtime._library().mxzip_mm(
            stream.cuda_stream, *(buffer.data_ptr() for buffer in buffers),
            common, rare, exponent, flat.data_ptr(), output.data_ptr(),
            m, 8, k, 0, 1,
        ))
        return output[:tokens, :rows].contiguous().reshape((*x.shape[:-1], rows))


_LIB.impl("kda_fused_direct", _cuda_linear, "CUDA")


@torch.library.register_fake("mxwave_glm53::kda_fused_direct")
def _fake(
    x: torch.Tensor, buffers: list[torch.Tensor],
    m: int, k: int, common: int, rare: int, exponent: int, rows: int,
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], rows))


def linear(
    x: torch.Tensor, buffers: list[torch.Tensor], pool: torch.Tensor,
    m: int, k: int, common: int, rare: int, exponent: int, rows: int,
) -> torch.Tensor:
    """Prototype a readonly fused op; retain the qualified mutable-scratch fallback above eight."""
    if k <= 0:
        raise ValueError("KDA input width must be positive")
    if x.numel() // k > 8:
        return kda_linear(x, buffers, pool, m, k, common, rare, exponent, rows)
    operation: Any = torch.ops.mxwave_glm53.kda_fused_direct
    return operation(x, buffers, m, k, common, rare, exponent, rows)
