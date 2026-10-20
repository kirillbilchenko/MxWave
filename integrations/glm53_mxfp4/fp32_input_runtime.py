"""Explore FP32 input partials using the qualified model's fixed stream-owned scratch pool."""

from __future__ import annotations

import ctypes
import functools
import hashlib
import os
from pathlib import Path
from typing import Any

import torch
from fp32_input_policy import (
    INPUT_GEOMETRY,
    LIBRARY_SHA256,
    RUNTIME_SHA256,
    SPLIT,
    scratch_bytes,
    selected,
)
from fp32_input_policy import policy as describe_policy
from zipserv_contract import require_qualified_exponent_base

_PRIMED = False


def prime() -> None:
    """Initialize the native handle during eager warmup, never during graph capture or import."""
    import kda_runtime as kda

    global _PRIMED
    if _PRIMED:
        return
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Prime FP32 input dispatch during eager warmup before graph capture")
    kda.validate_runtime()
    library(os.environ["MXWAVE_GLM53_FP32_LIBRARY"])
    _PRIMED = True


@functools.lru_cache(maxsize=1)
def library(path: str) -> Any:
    """Load only the retained FP32 ABI, keeping it separate from the qualified GEMM handle."""
    from mxwave_zipserv import runtime

    source = Path(path)
    if not source.is_absolute() or hashlib.sha256(source.read_bytes()).hexdigest() != LIBRARY_SHA256:
        raise ValueError("Require the measured FP32 native library")
    handle = ctypes.CDLL(str(source))
    if hasattr(handle, "mxzip_mm"):
        raise ValueError("FP32 library must not impersonate the original GEMM ABI")
    handle.mxzip_mm_fp32.argtypes = [*runtime._library().mxzip_mm.argtypes, ctypes.c_uint64]
    handle.mxzip_mm_fp32.restype = ctypes.c_int
    return handle


def workspace(pool: torch.Tensor, slot: int, tokens: int) -> torch.Tensor:
    """View one owned BF16 pool prefix as FP32 without allocating or converting values."""
    needed = scratch_bytes(tokens)
    if (pool.dtype != torch.bfloat16 or pool.ndim != 2 or pool.shape[0] != 2
            or not pool.is_contiguous() or slot not in (0, 1)
            or pool.shape[1] * pool.element_size() < needed):
        raise ValueError("Invalid FP32 view of the fixed two-slot pool")
    prefix = pool[slot, :needed // 2]
    if prefix.data_ptr() % 16:
        raise ValueError("FP32 partial workspace alignment changed")
    result = prefix.view(torch.float32)
    if result.data_ptr() % 16 or result.numel() * result.element_size() != needed:
        raise ValueError("FP32 partial workspace alignment or capacity changed")
    return result


@torch.library.custom_op("mxwave_glm53::kda_input_fp32", mutates_args=("pool",), device_types="cuda")
def input_linear(x: torch.Tensor, buffers: list[torch.Tensor], pool: torch.Tensor,
                 m: int, k: int, common: int, rare: int, exponent: int, rows: int) -> torch.Tensor:
    """Keep untested input sizes on the native route and lease typed scratch for tested sizes."""
    import kda_runtime as kda
    from mxwave_zipserv import runtime

    if (x.dtype != torch.bfloat16 or x.ndim < 1 or x.shape[-1] != k
            or (m, k, rows) != INPUT_GEOMETRY or pool.device != x.device
            or pool.dtype != torch.bfloat16 or pool.ndim != 2
            or pool.shape != (2, m * k) or not pool.is_contiguous()):
        raise ValueError("Invalid source-bound FP32 input geometry or shared scratch pool")
    runtime._validate_buffers(buffers, x.device)
    require_qualified_exponent_base(exponent, where="FP32 input experiment")
    tokens = x.numel() // k
    if not selected(tokens, m, k, rows):
        return kda.kda_linear(x, buffers, pool, m, k, common, rare, exponent, rows)
    prime()
    handle = library(os.environ["MXWAVE_GLM53_FP32_LIBRARY"])
    with torch.cuda.device(x.device):
        slot, lock = kda._slot(pool)
        with lock:
            scratch = workspace(pool, slot, tokens)
            n = runtime.padded_tokens(tokens)
            flat = x.reshape(tokens, k).contiguous()
            if tokens != n:
                padded = torch.zeros((n, k), device=x.device, dtype=x.dtype)
                padded[:tokens].copy_(flat)
            else:
                padded = flat
            output = torch.empty((n, m), device=x.device, dtype=x.dtype)
            status = handle.mxzip_mm_fp32(torch.cuda.current_stream(x.device).cuda_stream,
                *(buffer.data_ptr() for buffer in buffers), common, rare, exponent,
                padded.data_ptr(), output.data_ptr(), m, n, k, scratch.data_ptr(), SPLIT,
                scratch.numel() * scratch.element_size())
            if status:
                raise RuntimeError(f"FP32 input native call failed: {status}")
            return output[:tokens, :rows].contiguous().view(*x.shape[:-1], rows)


@input_linear.register_fake
def _fake(x: torch.Tensor, buffers: list[torch.Tensor], pool: torch.Tensor,
          m: int, k: int, common: int, rare: int, exponent: int, rows: int) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], rows))


def install() -> dict[str, Any]:
    """Opt in only within a disposable, frozen exact16 candidate before model capture."""
    import kda_runtime as kda
    from input_row16_dispatch import install as install_exact16

    if os.environ.get("MXWAVE_GLM53_KDA_INPUT_FP32_SPLIT2") != "1":
        raise ValueError("FP32 input dispatch requires an explicit experiment opt-in")
    if hashlib.sha256(Path(kda.__file__).read_bytes()).hexdigest() != RUNTIME_SHA256:
        raise ValueError("Require the qualified frozen KDA runtime")
    install_exact16()
    native = Path(os.environ["MXWAVE_GLM53_FP32_LIBRARY"])
    if not native.is_absolute() or hashlib.sha256(native.read_bytes()).hexdigest() != LIBRARY_SHA256:
        raise ValueError("Require the measured FP32 native library")
    if getattr(kda.apply, "_mxwave_input_fp32", False):
        return describe_policy()
    original = kda.apply

    @functools.wraps(original)
    def apply(layer: Any, x: torch.Tensor) -> torch.Tensor:
        """Keep shape selection inside the opaque input operation for dynamic model tracing."""
        from mxwave_zipserv import runtime

        metadata = layer.mxwave_kda_metadata
        if (metadata[0], metadata[1], metadata[-1]) != INPUT_GEOMETRY:
            return original(layer, x)
        buffers = [getattr(layer, "mxwave_kda_" + name) for name in runtime.BUFFER_NAMES]
        return input_linear(x, buffers, layer.mxwave_kda_pool, *metadata)

    apply._mxwave_input_fp32 = True  # type: ignore[attr-defined]
    kda.apply = apply
    return describe_policy()
