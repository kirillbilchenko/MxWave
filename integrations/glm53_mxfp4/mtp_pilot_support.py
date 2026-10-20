"""Keep native MTP metadata and draft scale ownership separate from the qualified base adapter."""

from __future__ import annotations

import functools
import hashlib
import os
import re
import weakref
from typing import Any

_INSTALLED = False
_DRAFTS: list[tuple[str, Any, list[dict[str, Any]]]] = []
_BASE_OWNERS: set[str] = set()


def expert_scope(prefix: str) -> str:
    """Accept only the known base MoE layers or the checkpoint's single MTP layer."""
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.mtp_block)?\.mlp\.experts$", prefix)
    if match is None:
        raise ValueError(f"Unrecognized GLM expert owner: {prefix}")
    layer = int(match[1])
    if 3 <= layer <= 44:
        return "base"
    if layer == 45:
        return "draft"
    raise ValueError(f"Expert layer {layer} is outside the base/MTP pilot scope")


def prepare() -> None:
    """Install per-process draft metadata and bypass lossless scale replacement for layer 45."""
    global _INSTALLED
    if _INSTALLED:
        return
    import model_plugin
    import torch
    from mtp_draft_loader import install as install_draft_loader
    from mtp_mapping import install
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
        compressed_tensors_moe_w4a4_mxfp4 as mxfp4_moe,
    )

    install()
    if int(os.environ.get("MXWAVE_MTP_DEPTH", "0")) > 0:
        install_draft_loader()
    select_original = CompressedTensorsConfig.get_quant_method

    @functools.wraps(select_original)
    def select(config: Any, layer: Any, prefix: str) -> Any:
        method = select_original(config, layer, prefix)
        if isinstance(method, mxfp4_moe.CompressedTensorsW4A4Mxfp4MoEMethod):
            expert_scope(prefix)
            previous = getattr(method, "mxwave_mtp_owner_prefix", prefix)
            if previous != prefix:
                raise RuntimeError("An MXFP4 method is shared between different expert owners")
            method.mxwave_mtp_owner_prefix = prefix
        return method

    CompressedTensorsConfig.get_quant_method = select
    original = model_plugin.prepare_layer

    @functools.wraps(original)
    def prepare_layer(method: Any, layer: Any, mode: str) -> None:
        prefix = str(getattr(method, "mxwave_mtp_owner_prefix", "unknown"))
        if expert_scope(prefix) == "base":
            if prefix in _BASE_OWNERS:
                raise RuntimeError("A base expert owner was prepared twice")
            _BASE_OWNERS.add(prefix)
            original(method, layer, mode)
            return
        if _DRAFTS:
            raise RuntimeError("The pilot supports one stock native MTP expert module per rank")
        scales = []
        for name in ("w13_weight_scale", "w2_weight_scale"):
            value = getattr(layer, name)
            if (value.dtype != torch.float8_e8m0fnu or value.ndim != 3
                    or not value.is_contiguous() or id(value) in model_plugin._STORAGE):
                raise ValueError("MTP draft must retain stock canonical E8M0 scale storage")
            scales.append({"name": name, "shape": list(value.shape), "dtype": str(value.dtype),
                "pointer": value.data_ptr(), "source_sha256": hashlib.sha256(
                    value.detach().view(torch.uint8).cpu().numpy()).hexdigest()})
        _DRAFTS.append((prefix, weakref.ref(layer), scales))
        print(f"MxWave MTP pilot retained stock draft scale/GEMM storage: {prefix}", flush=True)

    model_plugin.prepare_layer = prepare_layer
    _INSTALLED = True


def report(expected_depth: int) -> dict[str, Any]:
    """Prove the draft retains original scale bytes and bypasses the compressed GEMM dispatch."""
    import model_plugin
    import torch
    from mtp_draft_loader import report as loader_report

    if (not _INSTALLED or len(_DRAFTS) != int(expected_depth > 0)
            or {int(re.search(r"layers\.(\d+)", p)[1]) for p in _BASE_OWNERS} != set(range(3, 45))):
        raise RuntimeError("Native MTP draft loading/scope differs from the declared arm")
    rows = []
    for prefix, reference, scales in _DRAFTS:
        layer = reference()
        if layer is None:
            raise RuntimeError("Loaded native MTP expert module disappeared")
        for source in scales:
            value = getattr(layer, source["name"])
            if (value.data_ptr() != source["pointer"] or id(value) in model_plugin._STORAGE
                    or hashlib.sha256(value.detach().view(torch.uint8).cpu().numpy()).hexdigest()
                    != source["source_sha256"]):
                raise RuntimeError("MTP draft scale storage changed or entered compressed dispatch")
        rows.append({"prefix": prefix, "stock_scales_unchanged": True,
            "compressed_dispatch_bypassed": True, "scales": scales,
            "weights": [{"name": name, "shape": list(getattr(layer, name).shape),
                "dtype": str(getattr(layer, name).dtype)} for name in ("w13_weight", "w2_weight")]})
    return {"status": "passed", "depth": expected_depth, "draft_modules": rows,
            "draft_loader": loader_report(expected_depth),
            "base_scale_layers": model_plugin.report()["layers_seen"],
            "base_owner_prefixes": sorted(_BASE_OWNERS),
            "producer": "Existing qualified TP-only 288-expert canonical producer retained"}
