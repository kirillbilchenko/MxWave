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
    "all-linear",
]
ResolvedPolicyName = Literal[
    "qwen3.8-27b-mlp",
    "qwen3.8-27b-compatible",
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

        Qwen's gate and up projections consume the output of the same
        post-attention RMSNorm.  Attention input projections consume the input
        RMSNorm output.  Output projections and the MLP down projection consume
        internal activations and therefore have no LayerNorm-sized proxy.
        """
        if not self.name.startswith("qwen3.8"):
            return None
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


_POLICIES: dict[ResolvedPolicyName, QuantizationPolicy] = {
    "qwen3.8-27b-mlp": QuantizationPolicy(
        name="qwen3.8-27b-mlp",
        description="Qwen3.8-27B language-model MLP projections only",
    ),
    "qwen3.8-27b-compatible": QuantizationPolicy(
        name="qwen3.8-27b-compatible",
        description="Qwen3.8-27B MLP plus kernel-compatible attention projections",
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


def resolve_policy(model_dir: str | Path, requested: PolicyName) -> QuantizationPolicy:
    """Resolve ``auto`` from model config or return an explicitly requested policy."""
    config = load_config_json(model_dir)
    if requested == "auto":
        if not _is_qwen3_8_dense(config):
            raise ValueError(
                "No safe automatic policy for this architecture; select an explicit policy"
            )
        return _POLICIES["qwen3.8-27b-mlp"]

    policy = _POLICIES[requested]
    if policy.name.startswith("qwen3.8") and not _is_qwen3_8_dense(config):
        raise ValueError(f"Policy {policy.name!r} requires Qwen3_5ForConditionalGeneration")
    return policy


def expected_target_count(
    model_dir: str | Path,
    policy: QuantizationPolicy,
) -> int | None:
    """Return the architecture-derived target count when it is knowable."""
    if policy.name == "all-linear":
        return None
    config = load_config_json(model_dir)
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
