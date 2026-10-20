"""Opt-in fused-attention metadata repair for the pinned GLM runtime overlay.

The runtime loads individual checkpoint projections into fused linears. The
compressed-tensors selector must know those aliases to preserve BF16 exclusions.
This changes mapping metadata only; checkpoint tensors and quantization schemes
are left untouched. Apply before vLLM configures the model's quantization config.
"""

from __future__ import annotations

from typing import Any

FUSED_ATTENTION_COMPONENTS: dict[str, list[str]] = {
    "in_proj_qkvbfg_a": ["q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj", "g_a_proj"],
    "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
    "wk_weights_proj": ["wk", "weights_proj"],
}


def register_model_mapping(model_class: type[Any]) -> dict[str, list[str]]:
    """Add missing fused aliases without mutating another model's inherited mapping."""
    original = getattr(model_class, "packed_modules_mapping", {})
    mapping = {name: list(parts) for name, parts in original.items()}
    for name, parts in FUSED_ATTENTION_COMPONENTS.items():
        if name in mapping and mapping[name] != parts:
            raise ValueError(f"Unexpected existing GLM fused projection mapping: {name}")
        mapping[name] = parts.copy()
    model_class.packed_modules_mapping = mapping
    return mapping


def install() -> None:
    """Register aliases on the architecture class in this private experiment process."""
    from vllm.models.glm5next.nvidia.model import Glm5NextForConditionalGeneration

    register_model_mapping(Glm5NextForConditionalGeneration)
