"""Opaque selective KDA operation with fixed per-stream prefill storage and exact packing."""

from __future__ import annotations

import hashlib
import os
import threading
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from bf16_prefetch import Projection, decode_into
from kda_policy import Target
from kda_row_routing import use_fused
from prepare_bf16_attention import LIBRARY_SHA, RUNTIME_SHA, tensor_digest
from zipserv_contract import require_qualified_exponent_base

SLOTS = 2
FUSED_MAX_TOKENS = int(os.environ.get("MXWAVE_GLM53_KDA_FUSED_MAX_TOKENS", "1"))
if FUSED_MAX_TOKENS not in (1, 8):
    raise ValueError("Selective KDA fused execution supports the audited limits 1 or 8")
_OUTPUT_SIXTEEN = os.environ.get("MXWAVE_GLM53_KDA_OUTPUT_FUSED_16", "0")
if _OUTPUT_SIXTEEN not in ("0", "1"):
    raise ValueError("The output-only sixteen-row route must be explicitly zero or one")
OUTPUT_FUSED_16 = _OUTPUT_SIXTEEN == "1"
_POOLS: dict[int, torch.Tensor] = {}
_ENTRIES: list[dict[str, Any]] = []
_OWNERS: list[weakref.ReferenceType[torch.nn.Module]] = []
_VALIDATED = False
_GLOBAL_LOCK = threading.RLock()


@dataclass
class _PoolState:
    owner: weakref.ReferenceType[torch.Tensor]
    streams: dict[int, tuple[int, Any]] = field(default_factory=dict)
    captured_streams: set[int] = field(default_factory=set)


_STATES: dict[tuple[int, int], _PoolState] = {}


def validate_runtime() -> None:
    """Bind the existing qualified library and Python ABI before using the adapter."""
    global _VALIDATED
    if _VALIDATED:
        return
    from mxwave_zipserv import runtime

    path = Path(os.environ["MXWAVE_ZIPSERV_LIBRARY"])
    if (hashlib.sha256(path.read_bytes()).hexdigest() != LIBRARY_SHA
            or hashlib.sha256(Path(runtime.__file__).read_bytes()).hexdigest() != RUNTIME_SHA
            or torch.cuda.get_device_capability() != (12, 1)):
        raise ValueError("Require the pinned ZipServ ABI and SM121 library")
    _VALIDATED = True


def pool_for(device: torch.device, words: int) -> torch.Tensor:
    """Allocate one fixed two-slot pool per worker GPU; never grow an existing pool."""
    assert device.index is not None
    with _GLOBAL_LOCK:
        if device.index not in _POOLS:
            if words * SLOTS * 2 > 256 * 1024**2:
                raise ValueError("KDA prefill pool exceeds the declared 256 MiB bound")
            _POOLS[device.index] = torch.empty((SLOTS, words), device=device, dtype=torch.bfloat16)
        pool = _POOLS[device.index]
        if pool.shape != (SLOTS, words):
            raise ValueError("KDA pool dimensions changed within one worker")
        return pool


def _slot(pool: torch.Tensor) -> tuple[int, Any]:
    stream = torch.cuda.current_stream(pool.device)
    key = (pool.device.index, pool.data_ptr())
    with _GLOBAL_LOCK:
        for old in [k for k, state in _STATES.items() if state.owner() is None]:
            del _STATES[old]
        state = _STATES.setdefault(key, _PoolState(weakref.ref(pool)))
        if stream.cuda_stream not in state.streams:
            if len(state.streams) >= pool.shape[0]:
                raise ValueError("KDA prefill supports at most two inference streams per worker")
            occupied = {slot for slot, _ in state.streams.values()}
            slot = next(index for index in range(pool.shape[0]) if index not in occupied)
            state.streams[stream.cuda_stream] = (slot, threading.RLock())
        if torch.cuda.is_current_stream_capturing():
            state.captured_streams.add(stream.cuda_stream)
        pool.record_stream(stream)
        return state.streams[stream.cuda_stream]


def retire_idle_eager_slots() -> dict[str, Any]:
    """Release startup-only stream leases after device completion, preserving graph-owned slots.

    The pinned model-capture lifecycle calls this before serving is ready;
    no inference thread may acquire a pool lease concurrently. Never call it
    inside capture or use it to recycle storage referenced by a live graph.
    """
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Retire startup scratch leases only outside CUDA graph capture")
    retired, pinned = 0, 0
    with _GLOBAL_LOCK:
        devices = {pool.device for pool in _POOLS.values()}
        for device in devices:
            torch.cuda.synchronize(device)
        for state in _STATES.values():
            retained = {stream: value for stream, value in state.streams.items()
                        if stream in state.captured_streams}
            retired += len(state.streams) - len(retained)
            pinned += len(retained)
            state.streams = retained
    return {"status": "complete", "retired_eager_stream_leases": retired,
            "retained_captured_stream_leases": pinned, "pool_slots": SLOTS,
            "maximum_pool_bytes": 256 * 1024**2, "startup_only": True}


@torch.library.custom_op("mxwave_glm53::kda_linear", mutates_args=("pool",), device_types="cuda")
def kda_linear(x: torch.Tensor, buffers: list[torch.Tensor], pool: torch.Tensor,
               m: int, k: int, common: int, rare: int, exponent: int, rows: int) -> torch.Tensor:
    """Use split-K=1 at the opt-in token limit and stock GEMM with bounded scratch otherwise."""
    from mxwave_zipserv import runtime

    if (x.dtype != torch.bfloat16 or x.ndim < 1 or x.shape[-1] != k
            or pool.device != x.device or pool.dtype != torch.bfloat16
            or pool.ndim != 2 or pool.shape[1] < m * k or not 0 < rows <= m):
        raise ValueError("Invalid selective KDA operation input or prefill pool")
    runtime._validate_buffers(buffers, x.device)
    require_qualified_exponent_base(exponent, where="selective KDA operation")
    tokens = x.numel() // k
    if tokens == 0:
        return x.new_empty((*x.shape[:-1], rows))
    packed = runtime.PackedWeight(tuple(buffers), m, k, common, rare, exponent)
    if use_fused(tokens, m, k, rows, limit=FUSED_MAX_TOKENS, output_sixteen=OUTPUT_FUSED_16):
        return runtime.linear(x, packed, split_k=1)[..., :rows].contiguous()
    slot, lock = _slot(pool)
    with lock:
        weight = pool[slot, :m * k].view(m, k)
        decode_into(Projection("selective-kda", packed, rows), weight)
        return torch.nn.functional.linear(x, weight[:rows])


@kda_linear.register_fake
def _fake(x: torch.Tensor, buffers: list[torch.Tensor], pool: torch.Tensor,
          m: int, k: int, common: int, rare: int, exponent: int, rows: int) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], rows))


def prepare(layer: torch.nn.Module, target: Target, max_words: int) -> None:
    """Pack the fully loaded BF16 parameter, verify every decoded byte and release it."""
    from mxwave_zipserv import runtime

    validate_runtime()
    if getattr(layer, "mxwave_kda_metadata", None) is not None:
        raise ValueError("KDA parameter preparation is one-shot; refitting is unsupported")
    if (layer.tp_size != 2 or layer.tp_rank not in (0, 1)
            or layer.weight.dtype != torch.bfloat16
            or tuple(layer.weight.shape) != (target.rows, target.width)
            or layer.weight.device.type != "cuda"
            or layer.mxwave_kda_loaded_shards != target.expected_shards):
        raise ValueError("KDA TP2 shape, dtype or complete source loading gate failed")
    cpu = layer.weight.detach().cpu().contiguous()
    source_sha = tensor_digest(cpu)
    padded = torch.nn.functional.pad(cpu, (0, 0, 0, target.native_rows - target.rows))
    packed = runtime.compress(padded)
    require_qualified_exponent_base(packed.start_exp, where=target.prefix)
    packed = packed.to(layer.weight.device)
    raw_bytes = cpu.numel() * cpu.element_size()
    if packed.storage_bytes >= raw_bytes:
        raise ValueError("Selected BF16 KDA matrix does not shrink")
    decoded = runtime.decompress(packed).cpu()
    if tensor_digest(decoded) != tensor_digest(padded):
        raise RuntimeError("Prepared KDA bytes differ after complete GPU reconstruction")
    pool = pool_for(layer.weight.device, max_words)
    for name, buffer in zip(runtime.BUFFER_NAMES, packed.buffers, strict=True):
        layer.register_buffer("mxwave_kda_" + name, buffer)
    layer.register_buffer("mxwave_kda_pool", pool, persistent=False)
    layer.mxwave_kda_metadata = (packed.m, packed.k, packed.max_common,
                                packed.max_rare, packed.start_exp, target.rows)
    entry = {"prefix": target.prefix, "rank": layer.tp_rank, "kind": target.kind,
             "fused_max_tokens": FUSED_MAX_TOKENS,
             "fused_exact_rows": [16] if OUTPUT_FUSED_16 and target.kind == "output" else [],
             "logical_shape": [target.rows, target.width],
             "native_shape": [packed.m, packed.k], "raw_bytes": raw_bytes,
             "native_metadata": [packed.m, packed.k, packed.max_common,
                                 packed.max_rare, packed.start_exp],
             "packed_bytes": packed.storage_bytes, "loaded_weight_sha256": source_sha,
             "gpu_decode_bits_exact": True, "loaded_shards": sorted(
                 shard for shard in layer.mxwave_kda_loaded_shards if shard is not None),
             "row_parallel_source_loaded": None in layer.mxwave_kda_loaded_shards}
    layer.mxwave_kda_qualification = entry
    layer.mxwave_kda_original_parameter = weakref.ref(layer.weight)
    layer.register_parameter("weight", None)
    _ENTRIES.append(entry)
    _OWNERS.append(weakref.ref(layer))


def apply(layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Execute the opaque operation while exposing shared scratch mutation to compilers."""
    from mxwave_zipserv import runtime

    buffers = [getattr(layer, "mxwave_kda_" + name) for name in runtime.BUFFER_NAMES]
    return kda_linear(x, buffers, layer.mxwave_kda_pool, *layer.mxwave_kda_metadata)


def report() -> dict[str, Any]:
    """Report retained packed bytes and count the shared pool only once."""
    live = [entry for entry, owner in zip(_ENTRIES, _OWNERS, strict=True) if owner() is not None]
    return {"layers": list(_ENTRIES), "fused_max_tokens": FUSED_MAX_TOKENS,
            "output_fused_16": OUTPUT_FUSED_16,
            "prepared_layers": len(_ENTRIES),
            "resident_layers": len(live),
            "raw_bytes": sum(e["raw_bytes"] for e in live),
            "packed_bytes": sum(e["packed_bytes"] for e in live),
            "pool_bytes": sum(p.numel() * p.element_size() for p in _POOLS.values()),
            "pool_streams": [len(state.streams) for state in _STATES.values()
                             if state.owner() is not None],
            "captured_pool_streams": [len(state.captured_streams) for state in _STATES.values()
                                     if state.owner() is not None],
            "cuda_allocated_bytes": torch.cuda.memory_allocated(),
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated()}
