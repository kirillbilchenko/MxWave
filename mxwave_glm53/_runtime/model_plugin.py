"""Opt-in full-model pilot: retain exact expert scales and stock MoE routing/activation.

The default is stock execution. Only explicitly prepared MXFP4 expert tensors
use the experiment's opaque GEMM operation; mixed MXFP8 linears stay on vLLM.
"""

from __future__ import annotations

import functools
import importlib
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from .moe_runtime import NativeMoe, pack_scales

_INSTALLED = False
_NATIVE: dict[int, NativeMoe] = {}
_STORAGE: dict[int, dict[str, Any]] = {}
_SCRATCH: dict[tuple[int, int], torch.Tensor] = {}
_LAYERS: list[dict[str, Any]] = []


def native(kind: int) -> NativeMoe:
    """Load the immutable raw/aligned libraries lazily inside a GPU worker."""
    if kind not in _NATIVE:
        key = "MXWAVE_MOE_RAW_LIBRARY" if kind == 0 else "MXWAVE_MOE_ALIGNED_LIBRARY"
        _NATIVE[kind] = NativeMoe(Path(os.environ[key]))
    return _NATIVE[kind]


def _launch_moe(
    x: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor,
    output: torch.Tensor, temporary: torch.Tensor, locks: torch.Tensor,
    sorted_ids: torch.Tensor, expert_ids: torch.Tensor, padded: torch.Tensor,
    routing_weights: torch.Tensor, m: int, n: int, k: int, experts: int,
    topk: int, block: int, multiply: bool, tile_k: int, tile_n: int,
    blocks_per_sm: int, kind: int,
) -> None:
    """Launch one exact grouped GEMM on the caller's CUDA stream without allocating."""
    tensors = (x, weight, scales, output, temporary, locks, sorted_ids, expert_ids,
               padded, routing_weights)
    library = native(kind).library
    status = library.mxwave_moe_run(
        *(t.data_ptr() for t in tensors), m, n, k, experts, topk, block, int(multiply),
        x.device.index, tile_k, tile_n, blocks_per_sm,
        torch.cuda.current_stream(x.device).cuda_stream,
    )
    if status:
        raise RuntimeError(library.mxwave_moe_last_error().decode())


_MUTATIONS = ("output", "temporary", "locks")
moe_gemm = torch.library.custom_op(
    "mxwave_glm53::moe_gemm", _launch_moe, mutates_args=_MUTATIONS, device_types="cuda",
)


@moe_gemm.register_fake
def _fake_moe_gemm(*args: Any, **kwargs: Any) -> None:
    return None


# Keep the Library alive: its registrations otherwise disappear with its owner.
# The schema preserves the same write aliases and fake behavior as custom_op.
# The native launch still reads all dynamic tensor pointers on every invocation.
_DISPATCH_LIBRARY = torch.library.Library("mxwave_glm53", "FRAGMENT")
_DISPATCH_LIBRARY.define(
    "moe_gemm_dispatch" + torch.library.infer_schema(_launch_moe, mutates_args=_MUTATIONS),
)
_DISPATCH_LIBRARY.impl("moe_gemm_dispatch", _launch_moe, "CUDA")
torch.library.register_fake("mxwave_glm53::moe_gemm_dispatch")(_fake_moe_gemm)


def moe_operation(mode: str) -> Callable[..., None]:
    """Select a pre-registered launch path without changing its kernel or pointer lifetime."""
    if mode == "custom":
        return moe_gemm
    if mode == "dispatcher":
        return torch.ops.mxwave_glm53.moe_gemm_dispatch.default
    raise ValueError("MXWAVE_MOE_DISPATCH must be custom or dispatcher")


_GEMM_OP = moe_operation("custom")


@functools.lru_cache(maxsize=1024)
def _plan(m: int, n: int, k: int, block: int, topk: int, experts: int,
          device: int) -> tuple[int, int, int]:
    return native(0).plan(m, n, k, block, topk, experts, device)


def _temporary(x: torch.Tensor) -> torch.Tensor:
    stream = torch.cuda.current_stream(x.device).cuda_stream
    key = (x.device.index, stream)
    if key not in _SCRATCH:
        sms = torch.cuda.get_device_properties(x.device).multi_processor_count
        # Maximum stock non-atomic FP32 reduction scratch for every supported
        # block (the eight-token case doubles its smaller scratch requirement).
        _SCRATCH[key] = torch.empty(sms * 4 * 64 * 256, dtype=torch.float32, device=x.device)
    return _SCRATCH[key]


def _replace_config_scales(method: Any, old: torch.Tensor, replacement: torch.Tensor) -> int:
    """Replace every shared/cloned quant descriptor without retaining the expanded tensor."""
    queue, seen, count = [method.moe_quant_config, method.moe_kernel], set(), 0
    pointer = old.data_ptr()
    while queue:
        value = queue.pop()
        if id(value) in seen or value is None:
            continue
        seen.add(id(value))
        if len(seen) > 512:
            raise RuntimeError("Unexpected MoE descriptor object graph")
        if isinstance(value, (torch.Tensor, str, bytes, int, float, bool, type)):
            continue
        if isinstance(value, dict):
            queue.extend(value.values())
        elif isinstance(value, (tuple, list)):
            queue.extend(value)
        elif type(value).__module__.startswith("vllm.model_executor.layers.fused_moe"):
            for key, child in vars(value).items():
                if key == "scale" and isinstance(child, torch.Tensor) and child.data_ptr() == pointer:
                    setattr(value, key, replacement)
                    count += 1
                elif not isinstance(child, torch.Tensor):
                    queue.append(child)
    return count


def prepare_layer(method: Any, layer: torch.nn.Module, mode: str) -> None:
    """After stock repacking, qualify each whole scale tensor and release eligible raw storage."""
    if method.use_cutlass_mxfp4:
        raise ValueError("The exact-scale pilot requires the Marlin MoE backend")
    label = getattr(layer, "prefix", getattr(method.moe, "layer_name", "unknown"))
    entry: dict[str, Any] = {"layer": str(label), "mode": mode, "scales": []}
    for name in ("w13_weight_scale", "w2_weight_scale"):
        raw = getattr(layer, name)
        if raw.dtype != torch.float8_e8m0fnu or raw.ndim != 3 or not raw.is_contiguous():
            raise ValueError("Prepared MXFP4 scales are not canonical Marlin E8M0 tensors")
        record: dict[str, Any] = {"name": name, "shape": list(raw.shape),
                                  "raw_bytes": raw.numel(), "retained_bytes": raw.numel(),
                                  "compressed": False}
        if mode == "aligned":
            try:
                packed = pack_scales(raw, Path(os.environ["MXWAVE_MOE_RECORD_CODEC"]),
                                     encoding="sparse_palette")
            except ValueError as exc:
                raise ValueError("The selected profile requires all 84 scale tensors") from exc
            decoded = native(1).decode(packed)
            if not torch.equal(decoded, raw.view(torch.uint8)):
                raise RuntimeError("Prepared expert scale bytes differ after GPU decode")
            del decoded
            replacement = torch.nn.Parameter(packed.descriptor, requires_grad=False)
            references = _replace_config_scales(method, raw, replacement)
            if references < 1:
                raise RuntimeError("No expert quant descriptor owned the original scales")
            layer.register_buffer("mxwave_" + name + "_records", packed.records)
            layer.register_buffer("mxwave_" + name + "_exceptions", packed.exceptions)
            setattr(layer, name, replacement)
            _STORAGE[id(replacement)] = {"shape": tuple(raw.shape), "kind": 1, "packed": packed}
            record.update(compressed=True, retained_bytes=packed.retained_bytes,
                          gpu_bytes_exact=True, quant_descriptor_references=references)
        else:
            _STORAGE[id(raw)] = {"shape": tuple(raw.shape), "kind": 0}
        entry["scales"].append(record)
    _LAYERS.append(entry)
    layer.mxwave_scale_qualification = entry
    print(f"MxWave expert scales prepared: {label} ({mode})", flush=True)


def report() -> dict[str, Any]:
    """Expose ownership, actual allocation and scope to an engine worker RPC."""
    return {"layers": _LAYERS, "layers_seen": len(_LAYERS),
            "moe_dispatch": "dispatcher" if _GEMM_OP is moe_operation("dispatcher") else "custom",
            "scale_raw_bytes": sum(x["raw_bytes"] for e in _LAYERS for x in e["scales"]),
            "scale_retained_bytes": sum(x["retained_bytes"] for e in _LAYERS for x in e["scales"]),
            "scratch_bytes": sum(x.numel() * x.element_size() for x in _SCRATCH.values()),
            "scratch_streams": len(_SCRATCH),
            "cuda_allocated_bytes": torch.cuda.memory_allocated(),
            "cuda_reserved_bytes": torch.cuda.memory_reserved(),
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated()}


class ScaleWorker:
    """Read-only worker RPC extension for experiment allocation and coverage evidence."""

    def mxwave_scale_report(self) -> dict[str, Any]:
        """Return the local rank's exact-scale coverage and current allocations."""
        return report()


def install() -> None:
    """Register the bounded loader and explicitly selected experimental GEMM path."""
    global _INSTALLED, _GEMM_OP
    if _INSTALLED:
        return
    mode = os.environ.get("MXWAVE_MOE_MODE", "stock")
    if mode not in ("stock", "raw", "aligned"):
        raise ValueError("MXWAVE_MOE_MODE must be stock, raw or aligned")
    _GEMM_OP = moe_operation(os.environ.get("MXWAVE_MOE_DISPATCH", "custom"))
    _INSTALLED = True
    if mode == "stock":
        return
    from vllm import _custom_ops as ops

    module = importlib.import_module(
        "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe."
        "compressed_tensors_moe_w4a4_mxfp4")
    original_prepare = module.CompressedTensorsW4A4Mxfp4MoEMethod.process_weights_after_loading

    @functools.wraps(original_prepare)
    def prepare(method: Any, layer: torch.nn.Module) -> None:
        original_prepare(method, layer)
        prepare_layer(method, layer, mode)

    module.CompressedTensorsW4A4Mxfp4MoEMethod.process_weights_after_loading = prepare
    import inspect

    original_gemm = ops.moe_wna16_marlin_gemm
    signature = inspect.signature(original_gemm)

    @functools.wraps(original_gemm)
    def gemm(*args: Any, **kwargs: Any) -> torch.Tensor:
        call = signature.bind(*args, **kwargs)
        call.apply_defaults()
        p = call.arguments
        scales = p["b_scales"]
        if id(scales) not in _STORAGE:
            return original_gemm(*args, **kwargs)
        metadata = _STORAGE[id(scales)]
        x, output, weight = p["input"], p["output"], p["b_qweight"]
        if (any(p[key] is not None and p[key].numel()
                for key in ("b_bias", "a_scales", "global_scale", "b_qzeros"))
                or p["use_atomic_add"] or not p["use_fp32_reduce"] or p["is_zp_float"]
                or x.dtype != torch.bfloat16 or output is None):
            raise ValueError("Grouped GEMM is outside the exact-scale runtime contract")
        m, n, k = (int(p[key]) for key in ("size_m", "size_n", "size_k"))
        experts = metadata["shape"][0]
        block, topk = int(p["moe_block_size"]), int(p["top_k"])
        if (n != metadata["shape"][2] or k != metadata["shape"][1] * 32
                or tuple(weight.shape) != (experts, k // 16, n * 2)):
            raise ValueError("Grouped scale ownership does not match the GEMM weight")
        plan = _plan(m, n, k, block, topk, experts, x.device.index)
        if p["thread_k"] > 0 or p["thread_n"] > 0:
            plan = (p["thread_k"], p["thread_n"], p["blocks_per_sm"])
        _GEMM_OP(x, weight, scales, output, _temporary(x), p["workspace"],
                 p["sorted_token_ids"], p["expert_ids"], p["num_tokens_past_padded"],
                 p["topk_weights"], m, n, k, experts, topk, block,
                 p["mul_topk_weights"], *plan, metadata["kind"])
        return output

    ops.moe_wna16_marlin_gemm = gemm
