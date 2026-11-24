"""Isolate checkpoint CT policy for the pinned new runner's native MTP model load."""

from __future__ import annotations

import contextvars
import copy
import functools
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

_INSTALLED = False
_CALLS: list[dict[str, Any]] = []
_PADDING: list[dict[str, Any]] = []
_WEIGHT_LOADS: list[dict[str, Any]] = []
_EXPECTED_SOURCE_COUNT: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "mxwave_mtp_expected_source_count", default=None)
CONFIG_SHA = "194d290ebb0f19aedc60d24ea9e236e33c86616e460f4a108a842f6fd5b8b198"


def clone_config_for_draft(config: Any, source_quantization: dict[str, Any]) -> Any:
    """Keep all engine settings but rebuild an independent, unmapped CT config from source."""
    result = copy.copy(config)
    result.quant_config = type(config.quant_config).from_config(copy.deepcopy(source_quantization))
    if result is config or result.quant_config is config.quant_config:
        raise RuntimeError("MTP must not reuse the already mapped target quantization config")
    return result


def pad_bf16_nope_weight(value: Any, kv_rank: int, rope_rows: int) -> Any:
    """Match the base loader's zero rope rows while preserving every source BF16 bit."""
    import torch

    if value.dtype != torch.bfloat16 or value.ndim != 2 or kv_rank < 1 or rope_rows < 1:
        raise ValueError("Require a BF16 NoPE KV projection and positive declared dimensions")
    if value.shape[0] == kv_rank + rope_rows:
        return value
    if value.shape[0] != kv_rank:
        raise ValueError("NoPE KV projection has an unexpected row count")
    return torch.cat((value, value.new_zeros((rope_rows, value.shape[1]))), dim=0)


def bf16_projection_slices(
    weight: Any, q: Any, kv: Any, *, q_rank: int, kv_rank: int, rope_rows: int,
    hidden_size: int, nope: bool,
) -> list[tuple[str, Any, Any]]:
    """Compare source projection rows using the loaded model's declared geometry."""
    import torch

    if min(q_rank, kv_rank, hidden_size) < 1 or rope_rows < 0:
        raise ValueError("Invalid native MTP projection dimensions")
    source_kv_rows = kv_rank if nope else kv_rank + rope_rows
    if (tuple(weight.shape) != (q_rank + kv_rank + rope_rows, hidden_size)
            or tuple(q.shape) != (q_rank, hidden_size)
            or tuple(kv.shape) != (source_kv_rows, hidden_size)
            or any(t.dtype != torch.bfloat16 for t in (weight, q, kv))):
        raise ValueError("Native MTP fused projection differs from declared source geometry")
    stop = q_rank + source_kv_rows
    result = [("q", weight[:q_rank], q), ("kv_source", weight[q_rank:stop], kv)]
    if stop < weight.shape[0]:
        result.append(("zero_rope", weight[stop:], torch.zeros(
            (weight.shape[0] - stop, hidden_size), dtype=torch.bfloat16)))
    return result


def install(source: Path = Path("/model/config.json")) -> None:
    """Wrap only the new runner's draft get_model call; leave target construction untouched."""
    global _INSTALLED
    if _INSTALLED:
        return
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )
    from vllm.models.glm5next.nvidia.mtp import Glm5NextMTP
    from vllm.v1.worker.gpu.spec_decode.eagle import utils

    original_weights = Glm5NextMTP.load_weights

    @functools.wraps(original_weights)
    def load_weights(model: Any, weights: Any) -> Any:
        import torch

        config = model.config
        source_keys: set[str] = set()

        def adapted() -> Any:
            for name, value in weights:
                prefixes = (f"model.language_model.layers.{config.num_hidden_layers}.",
                            f"model.layers.{config.num_hidden_layers}.")
                if name.startswith(prefixes):
                    if name in source_keys:
                        raise ValueError("Duplicate native MTP source tensor")
                    source_keys.add(name)
                suffix = f".layers.{config.num_hidden_layers}.self_attn.kv_a_proj_with_mqa.weight"
                if (getattr(config, "mla_nope", False) and config.qk_rope_head_dim > 0
                        and name.endswith(suffix) and value.dtype == torch.bfloat16):
                    if config.num_nextn_predict_layers != 1 or value.shape[1] != config.hidden_size:
                        raise ValueError("NoPE MTP padding is outside the audited single-layer scope")
                    padded = pad_bf16_nope_weight(value, config.kv_lora_rank, config.qk_rope_head_dim)
                    if padded is not value:
                        _PADDING.append({"name": name, "source_shape": list(value.shape),
                            "padded_shape": list(padded.shape),
                            "source_bf16_sha256": hashlib.sha256(
                                value.detach().contiguous().view(torch.uint8).cpu().numpy()).hexdigest(),
                            "source_values_preserved": True, "rope_rows_zero": True})
                        print("MxWave native MTP: appended zero BF16 NoPE rope rows", flush=True)
                    value = padded
                yield name, value

        loaded = original_weights(model, adapted())
        declared = set(dict(model.named_parameters()))
        allowed = {"model.embed_tokens.weight",
                   f"model.layers.{config.num_hidden_layers}.shared_head.head.weight"}
        missing = declared - loaded
        if missing - allowed:
            raise RuntimeError("Unloaded native MTP parameters: " + repr(sorted(missing - allowed)))
        expected = _EXPECTED_SOURCE_COUNT.get()
        if expected is not None and len(source_keys) != expected:
            raise RuntimeError("Native MTP did not consume all 1,756 audited source tensors")
        _WEIGHT_LOADS.append({"source_tensors": len(source_keys),
            "declared_parameters": len(declared), "loaded_parameters": len(loaded),
            "allowed_unloaded_shared_vocabulary": sorted(missing),
            "all_declared_nonvocabulary_parameters_loaded": True,
            "full_source_count_required": expected is not None})
        return loaded

    Glm5NextMTP.load_weights = load_weights

    original = utils.get_model

    @functools.wraps(original)
    def load(*, vllm_config: Any, model_config: Any = None, **kwargs: Any) -> Any:
        if (model_config is None or model_config.hf_config.architectures != ["Glm5NextMTPModel"]):
            return original(vllm_config=vllm_config, model_config=model_config, **kwargs)
        if not isinstance(vllm_config.quant_config, CompressedTensorsConfig):
            raise TypeError("This MTP bridge supports only the audited CT target")
        raw = source.read_bytes()
        if hashlib.sha256(raw).hexdigest() != CONFIG_SHA:
            raise ValueError("MTP policy source differs from the audited checkpoint")
        source_quantization = json.loads(raw)["quantization_config"]
        before = hashlib.sha256(pickle.dumps(vllm_config.quant_config, protocol=5)).hexdigest()
        independent = clone_config_for_draft(vllm_config, source_quantization)
        print("MxWave native MTP: loading independent source CT policy", flush=True)
        context = _EXPECTED_SOURCE_COUNT.set(1756)
        try:
            model = original(vllm_config=independent, model_config=model_config, **kwargs)
        finally:
            _EXPECTED_SOURCE_COUNT.reset(context)
        after = hashlib.sha256(pickle.dumps(vllm_config.quant_config, protocol=5)).hexdigest()
        if before != after:
            raise RuntimeError("Native MTP loading mutated the target's CT policy")
        _CALLS.append({"source_config_sha256": CONFIG_SHA, "target_policy_unchanged": True,
            "separate_quant_config": True, "target_policy_sha256": before,
            "runtime_helper": "vllm.v1.worker.gpu.spec_decode.eagle.utils.get_model"})
        return model

    utils.get_model = load
    _INSTALLED = True


def report(depth: int) -> dict[str, Any]:
    """Require exactly one isolated draft load when MTP is enabled."""
    if ((depth > 0 and not _INSTALLED) or len(_CALLS) != int(depth > 0)
            or len(_WEIGHT_LOADS) != int(depth > 0)):
        raise RuntimeError("Native MTP loader invocation differs from the declared pilot arm")
    return {"status": "passed", "installed": _INSTALLED, "calls": _CALLS,
            "bf16_nope_padding": _PADDING, "weight_loads": _WEIGHT_LOADS}
