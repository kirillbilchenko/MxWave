"""Map this checkpoint's explicit CT selectors onto the native text-only MTP namespace."""

from __future__ import annotations

from typing import Any

from attention_mapping import FUSED_ATTENTION_COMPONENTS


def register_mtp_mapping(base: type[Any], draft: type[Any], *,
                         layer_index: int = 45) -> dict[str, list[str]]:
    """Give the draft independent selector metadata without changing base-model policy."""
    mapping = {key: list(value) for key, value in base.packed_modules_mapping.items()}
    if any(mapping.get(key) != values for key, values in FUSED_ATTENTION_COMPONENTS.items()):
        raise ValueError("Register the audited base attention aliases before MTP metadata")
    for key, values in getattr(draft, "packed_modules_mapping", {}).items():
        if key in mapping and mapping[key] != values:
            raise ValueError(f"Native MTP fused alias differs from the base model: {key}")
        mapping[key] = list(values)
    mapper_type = type(base.hf_to_vllm_mapper)
    draft.hf_to_vllm_mapper = mapper_type(orig_to_new_prefix={
        "model.language_model.": "model.", "lm_head": f"model.layers.{layer_index}.head"})
    draft.packed_modules_mapping = mapping
    return mapping


def install() -> None:
    """Install MTP-only metadata in the experiment process before quantization configuration."""
    from attention_mapping import install as install_base
    from vllm.models.glm5next.nvidia.model import Glm5NextForConditionalGeneration
    from vllm.models.glm5next.nvidia.mtp import Glm5NextMTP

    install_base()
    register_mtp_mapping(Glm5NextForConditionalGeneration, Glm5NextMTP)
