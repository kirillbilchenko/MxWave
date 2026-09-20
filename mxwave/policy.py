"""Architecture-aware tensor selection policies.

Quantizing every key ending in ``.weight`` is unsafe: modern multimodal and
hybrid-attention checkpoints contain embeddings, convolutions, state-space
parameters, and projections with different runtime support.  Policies make the
selection explicit and auditable before any tensor data is loaded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from .format import load_config_json
from .shard import TensorInfo

PolicyName = Literal[
    "auto",
    "qwen3.8-27b-mlp",
    "qwen3.8-27b-compatible",
    "xing4-29b-a4b",
    "all-linear",
]
ResolvedPolicyName = Literal[
    "qwen3.8-27b-mlp",
    "qwen3.8-27b-compatible",
    "xing4-29b-a4b",
    "all-linear",
]

_QWEN_MLP = re.compile(
    r"^model\.language_model\.layers\.\d+\.mlp\."
    r"(?:gate_proj|up_proj|down_proj)\.weight$"
)
_QWEN_FULL_ATTN = re.compile(
    r"^model\.language_model\.layers\.\d+\.self_attn\."
    r"(?:q_proj|k_proj|v_proj|o_proj)\.weight$"
)
_QWEN_LINEAR_ATTN = re.compile(
    r"^model\.language_model\.layers\.\d+\.linear_attn\."
    r"(?:in_proj_qkv|in_proj_z|out_proj)\.weight$"
)
_XING_BASE_LAYER = r"(?:[0-9]|[1-3][0-9])"
_XING_MOE_LAYER = r"(?:[2-9]|[1-3][0-9])"
_XING_EXPERT = r"(?:[0-9]|[1-5][0-9]|6[0-3])"
_XING_DENSE_MLP = re.compile(
    r"^model\.layers\.[01]\.mlp\.(?:gate_proj|up_proj|down_proj)\.weight$"
)
_XING_ROUTED_EXPERT = re.compile(
    rf"^model\.layers\.{_XING_MOE_LAYER}\.mlp\.experts\.{_XING_EXPERT}\."
    r"(?:gate_proj|up_proj|down_proj)\.weight$"
)
_XING_SHARED_EXPERT = re.compile(
    rf"^model\.layers\.{_XING_MOE_LAYER}\.mlp\.shared_experts\."
    r"(?:gate_proj|up_proj|down_proj)\.weight$"
)
_XING_ATTN = re.compile(
    rf"^model\.layers\.{_XING_BASE_LAYER}\.self_attn\."
    r"(?:q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj|o_proj)\.weight$"
)
_GENERIC_EXCLUDE = re.compile(
    r"(?:embed|embedding|lm_head|norm|router|conv|patch|position|relative_attention)"
)
_FLOAT_DTYPES = frozenset({"BF16", "F16", "F32"})


@dataclass(frozen=True)
class QuantizationPolicy:
    """Resolved rules for selecting checkpoint tensors."""

    name: ResolvedPolicyName
    description: str

    def matches_name(self, name: str) -> bool:
        """Return whether ``name`` is a weight selected by this policy."""
        if self.name == "qwen3.8-27b-mlp":
            return _QWEN_MLP.fullmatch(name) is not None
        if self.name == "qwen3.8-27b-compatible":
            return any(
                pattern.fullmatch(name) is not None
                for pattern in (_QWEN_MLP, _QWEN_FULL_ATTN, _QWEN_LINEAR_ATTN)
            )
        if self.name == "xing4-29b-a4b":
            return any(
                pattern.fullmatch(name) is not None
                for pattern in (
                    _XING_DENSE_MLP,
                    _XING_ROUTED_EXPERT,
                    _XING_SHARED_EXPERT,
                    _XING_ATTN,
                )
            )
        return name.endswith(".weight") and _GENERIC_EXCLUDE.search(name) is None

    def selects(self, info: TensorInfo) -> bool:
        """Return whether a tensor is both named and shaped for MXFP4."""
        if not self.matches_name(info.name):
            return False
        if self.name == "all-linear":
            return info.dtype in _FLOAT_DTYPES and len(info.shape) == 2 and info.shape[-1] % 32 == 0
        return True

    def gamma_proxy_source(self, target_name: str) -> str | None:
        """Return the RMSNorm tensor that directly scales a target's input.

        Architecture adapters cover only direct, shape-compatible norm-to-linear
        paths. Output projections and MLP down projections consume internal
        activations and therefore have no LayerNorm-sized proxy.
        """
        if self.name.startswith("qwen3.8"):
            for suffix in (".mlp.gate_proj.weight", ".mlp.up_proj.weight"):
                if target_name.endswith(suffix):
                    return f"{target_name.removesuffix(suffix)}.post_attention_layernorm.weight"
            attention_input_suffixes = (
                ".self_attn.q_proj.weight",
                ".self_attn.k_proj.weight",
                ".self_attn.v_proj.weight",
                ".linear_attn.in_proj_qkv.weight",
                ".linear_attn.in_proj_z.weight",
            )
            for suffix in attention_input_suffixes:
                if target_name.endswith(suffix):
                    return f"{target_name.removesuffix(suffix)}.input_layernorm.weight"
            return None

        if self.name == "xing4-29b-a4b":
            layer_prefix, separator, projection = target_name.partition(".mlp.")
            if separator and projection.endswith(("gate_proj.weight", "up_proj.weight")):
                return f"{layer_prefix}.post_attention_layernorm.weight"
            attention_sources = {
                ".self_attn.q_a_proj.weight": ".input_layernorm.weight",
                ".self_attn.kv_a_proj_with_mqa.weight": ".input_layernorm.weight",
                ".self_attn.q_b_proj.weight": ".self_attn.q_a_layernorm.weight",
                ".self_attn.kv_b_proj.weight": ".self_attn.kv_a_layernorm.weight",
            }
            for suffix, source_suffix in attention_sources.items():
                if target_name.endswith(suffix):
                    return f"{target_name.removesuffix(suffix)}{source_suffix}"
        return None


_POLICIES: dict[ResolvedPolicyName, QuantizationPolicy] = {
    "qwen3.8-27b-mlp": QuantizationPolicy(
        name="qwen3.8-27b-mlp",
        description="Qwen3.8-27B language-model MLP projections only",
    ),
    "qwen3.8-27b-compatible": QuantizationPolicy(
        name="qwen3.8-27b-compatible",
        description="Qwen3.8-27B MLP plus kernel-compatible attention projections",
    ),
    "xing4-29b-a4b": QuantizationPolicy(
        name="xing4-29b-a4b",
        description=(
            "Xing4.0-29B-A4B base attention, dense MLP, routed experts, and shared experts"
        ),
    ),
    "all-linear": QuantizationPolicy(
        name="all-linear",
        description="explicit experimental policy for eligible 2D linear weights",
    ),
}


def _is_qwen3_8_dense(config: dict[str, object]) -> bool:
    architectures = config.get("architectures")
    text_config = config.get("text_config")
    return (
        config.get("model_type") == "qwen3_5"
        and isinstance(architectures, list)
        and "Qwen3_5ForConditionalGeneration" in architectures
        and isinstance(text_config, dict)
        and text_config.get("num_hidden_layers") == 64
        and text_config.get("hidden_size") == 5120
        and text_config.get("intermediate_size") == 17408
    )


def _is_xing4_29b_a4b(config: dict[str, object]) -> bool:
    """Return whether config identifies the exact supported Xing4 checkpoint family."""
    architectures = config.get("architectures")
    expected_fields = {
        "num_hidden_layers": 40,
        "hidden_size": 3584,
        "intermediate_size": 9216,
        "moe_intermediate_size": 1024,
        "n_routed_experts": 64,
        "n_shared_experts": 1,
        "num_experts_per_tok": 4,
        "first_k_dense_replace": 2,
        "num_nextn_predict_layers": 1,
        "q_lora_rank": 768,
        "kv_lora_rank": 512,
        "hc_mult": 4,
    }
    return (
        config.get("model_type") == "xing4_0"
        and isinstance(architectures, list)
        and "Xing4_0ForCausalLM" in architectures
        and all(config.get(field) == value for field, value in expected_fields.items())
    )


def resolve_policy(model_dir: str | Path, requested: PolicyName) -> QuantizationPolicy:
    """Resolve ``auto`` from model config or return an explicitly requested policy."""
    config = load_config_json(model_dir)
    if requested == "auto":
        if _is_qwen3_8_dense(config):
            return _POLICIES["qwen3.8-27b-mlp"]
        if _is_xing4_29b_a4b(config):
            return _POLICIES["xing4-29b-a4b"]
        raise ValueError(
            "No safe automatic policy for this architecture; select an explicit policy"
        )

    policy = _POLICIES[requested]
    if policy.name.startswith("qwen3.8") and not _is_qwen3_8_dense(config):
        raise ValueError(f"Policy {policy.name!r} requires Qwen3_5ForConditionalGeneration")
    if policy.name == "xing4-29b-a4b" and not _is_xing4_29b_a4b(config):
        raise ValueError(f"Policy {policy.name!r} requires the exact Xing4.0-29B-A4B config")
    return policy


def expected_target_count(
    model_dir: str | Path,
    policy: QuantizationPolicy,
) -> int | None:
    """Return the architecture-derived target count when it is knowable."""
    if policy.name == "all-linear":
        return None
    config = load_config_json(model_dir)
    if policy.name == "xing4-29b-a4b":
        num_layers = cast(int, config["num_hidden_layers"])
        first_dense = cast(int, config["first_k_dense_replace"])
        routed_experts = cast(int, config["n_routed_experts"])
        dense_mlp = first_dense * 3
        moe_mlp = (num_layers - first_dense) * (routed_experts + 1) * 3
        attention = num_layers * 5
        return dense_mlp + moe_mlp + attention

    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        raise TypeError("Qwen config is missing text_config")
    raw_num_layers = text_config.get("num_hidden_layers")
    if not isinstance(raw_num_layers, int) or raw_num_layers <= 0:
        raise ValueError("Qwen text_config has an invalid num_hidden_layers")
    num_layers = raw_num_layers
    if policy.name == "qwen3.8-27b-mlp":
        return num_layers * 3

    raw_layer_types = text_config.get("layer_types")
    if not isinstance(raw_layer_types, list) or len(raw_layer_types) != num_layers:
        raise ValueError("Qwen text_config has an invalid layer_types list")
    layer_types = cast(list[object], raw_layer_types)
    full_attention = sum(item == "full_attention" for item in layer_types)
    linear_attention = sum(item == "linear_attention" for item in layer_types)
    if full_attention + linear_attention != num_layers:
        raise ValueError("Qwen layer_types contains an unsupported layer type")
    return num_layers * 3 + full_attention * 4 + linear_attention * 3


def validate_selected_tensor(info: TensorInfo, policy: QuantizationPolicy) -> None:
    """Fail early when a policy-selected tensor cannot be encoded as MXFP4."""
    if info.dtype not in _FLOAT_DTYPES:
        raise ValueError(
            f"Policy {policy.name!r} selected {info.name!r} with unsupported dtype {info.dtype}"
        )
    if len(info.shape) != 2:
        raise ValueError(
            f"Policy {policy.name!r} selected {info.name!r} with non-matrix shape {info.shape}"
        )
    if info.shape[-1] % 32 != 0:
        raise ValueError(
            f"Policy {policy.name!r} selected {info.name!r}; input dimension "
            f"{info.shape[-1]} is not divisible by 32"
        )
