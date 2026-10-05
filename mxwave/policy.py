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
    "kolibri1-routed-experts",
    "all-linear",
]
ResolvedPolicyName = Literal[
    "qwen3.8-27b-mlp",
    "qwen3.8-27b-compatible",
    "kolibri1-routed-experts",
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
_ROUTER_EXCLUDE = re.compile(r"(?:^|\.)(?:gate|shared_expert_gate|mtp)(?:\.|$)")
_QWEN_AUXILIARY_INPUT = re.compile(
    r"^model\.language_model\.layers\.\d+\.linear_attn\.(?:in_proj_a|in_proj_b)\.weight$"
)
_FLOAT_DTYPES = frozenset({"BF16", "F16", "F32"})
_KOLIBRI_EXPERT = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\.weight$"
)
_KOLIBRI_IGNORE = re.compile(
    r"^(?:(?:model\.embed_tokens|model\.norm|lm_head)|model\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj|q_norm|k_norm)|"
    r"mlp\.(?:gate|shared_experts\.(?:gate_proj|up_proj|down_proj))|"
    r"input_layernorm|post_attention_layernorm|post_attn_norm|post_ffn_norm))\.weight$"
)


@dataclass(frozen=True)
class QuantizationPolicy:
    """Resolved rules for selecting checkpoint tensors."""

    name: ResolvedPolicyName
    description: str

    def matches_name(self, name: str) -> bool:
        """Return whether ``name`` is a weight selected by this policy."""
        if self.name == "kolibri1-routed-experts":
            return _KOLIBRI_EXPERT.fullmatch(name) is not None
        if self.name == "qwen3.8-27b-mlp":
            return _QWEN_MLP.fullmatch(name) is not None
        if self.name == "qwen3.8-27b-compatible":
            return any(
                pattern.fullmatch(name) is not None
                for pattern in (_QWEN_MLP, _QWEN_FULL_ATTN, _QWEN_LINEAR_ATTN)
            )
        return (
            name.endswith(".weight")
            and _GENERIC_EXCLUDE.search(name) is None
            and _ROUTER_EXCLUDE.search(name) is None
        )

    def ignores(self, info: TensorInfo) -> bool:
        """Classify explicit passthrough weights independently of the selected targets.

        Unknown weighted modules in an architecture-specific policy remain
        unclassified, so the coverage check can reject an incomplete policy.
        """
        name = info.name
        if self.name == "kolibri1-routed-experts":
            return _KOLIBRI_IGNORE.fullmatch(name) is not None
        if not name.endswith(".weight"):
            return False
        if _GENERIC_EXCLUDE.search(name) or _ROUTER_EXCLUDE.search(name):
            return True
        if self.name == "all-linear":
            return False
        if name.startswith(("model.visual.", "model.vision_tower.")):
            return True
        if _QWEN_AUXILIARY_INPUT.fullmatch(name):
            return True
        return self.name == "qwen3.8-27b-mlp" and any(
            pattern.fullmatch(name) for pattern in (_QWEN_FULL_ATTN, _QWEN_LINEAR_ATTN)
        )

    @property
    def gamma_proxy_offset(self) -> float:
        """Return the offset used by the policy's stored RMSNorm parameters."""
        return 1.0 if self.name.startswith("qwen3.8") else 0.0

    def selects(self, info: TensorInfo) -> bool:
        """Return whether a tensor is both named and shaped for MXFP4."""
        if not self.matches_name(info.name):
            return False
        if self.name == "all-linear":
            return info.dtype in _FLOAT_DTYPES and len(info.shape) == 2 and info.shape[-1] % 32 == 0
        return True

    def gamma_proxy_source(self, target_name: str) -> str | None:
        """Return the RMSNorm tensor that directly scales a target's input.

        Qwen3.5 RMSNorm stores a zero-centered weight: its effective multiplier
        is ``1 + weight``, as represented by ``gamma_proxy_offset``.
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
    "kolibri1-routed-experts": QuantizationPolicy(
        name="kolibri1-routed-experts",
        description="Kolibri 1 routed experts only; all other weights preserved",
    ),
}


def _kolibri_dimensions(config: dict[str, object]) -> tuple[int, int, int, int]:
    if config.get("model_type") != "kolibri1" or config.get("architectures") != [
        "Kolibri1ForCausalLM"
    ]:
        raise ValueError("Kolibri policy requires Kolibri1ForCausalLM / kolibri1")
    dimensions: list[int] = []
    for key in ("num_hidden_layers", "num_experts", "hidden_size", "moe_intermediate_size"):
        value = config.get(key)
        if type(value) is not int or value <= 0:
            raise ValueError(f"Kolibri config has an invalid {key}")
        dimensions.append(value)
    layers, experts, hidden, intermediate = dimensions
    if hidden % 128 or intermediate % 128:
        raise ValueError("Kolibri expert dimensions must be divisible by 128 for Marlin")
    layer_types = config.get("layer_types")
    if (
        not isinstance(layer_types, list)
        or len(layer_types) != layers
        or any(item not in ("sliding_attention", "full_attention") for item in layer_types)
    ):
        raise ValueError("Kolibri config has an invalid layer_types")
    return layers, experts, hidden, intermediate


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
    if policy.name == "kolibri1-routed-experts":
        _kolibri_dimensions(config)
    if policy.name.startswith("qwen3.8") and not _is_qwen3_8_dense(config):
        raise ValueError(f"Policy {policy.name!r} requires Qwen3_5ForConditionalGeneration")
    return policy


def expected_target_count(
    model_dir: str | Path,
    policy: QuantizationPolicy,
) -> int | None:
    """Return the architecture-derived target count when it is knowable."""
    names = expected_target_names(model_dir, policy)
    return len(names) if names is not None else None


def expected_target_names(
    model_dir: str | Path,
    policy: QuantizationPolicy,
) -> frozenset[str] | None:
    """Derive exact target names from the model's declared decoder layout."""
    if policy.name == "all-linear":
        return None
    config = load_config_json(model_dir)
    if policy.name == "kolibri1-routed-experts":
        layers, experts, _hidden, _intermediate = _kolibri_dimensions(config)
        return frozenset(
            f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"
            for layer in range(layers)
            for expert in range(experts)
            for projection in ("gate_proj", "up_proj", "down_proj")
        )
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        raise TypeError("Qwen config is missing text_config")
    raw_num_layers = text_config.get("num_hidden_layers")
    if not isinstance(raw_num_layers, int) or raw_num_layers <= 0:
        raise ValueError("Qwen text_config has an invalid num_hidden_layers")
    num_layers = raw_num_layers
    names = {
        f"model.language_model.layers.{index}.mlp.{projection}.weight"
        for index in range(num_layers)
        for projection in ("gate_proj", "up_proj", "down_proj")
    }
    if policy.name == "qwen3.8-27b-mlp":
        return frozenset(names)

    raw_layer_types = text_config.get("layer_types")
    if not isinstance(raw_layer_types, list) or len(raw_layer_types) != num_layers:
        raise ValueError("Qwen text_config has an invalid layer_types list")
    layer_types = cast(list[object], raw_layer_types)
    full_attention = sum(item == "full_attention" for item in layer_types)
    linear_attention = sum(item == "linear_attention" for item in layer_types)
    if full_attention + linear_attention != num_layers:
        raise ValueError("Qwen layer_types contains an unsupported layer type")
    for index, layer_type in enumerate(layer_types):
        projections = (
            ("q_proj", "k_proj", "v_proj", "o_proj")
            if layer_type == "full_attention"
            else ("in_proj_qkv", "in_proj_z", "out_proj")
        )
        attention = "self_attn" if layer_type == "full_attention" else "linear_attn"
        names.update(
            f"model.language_model.layers.{index}.{attention}.{projection}.weight"
            for projection in projections
        )
    return frozenset(names)


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


def validate_policy_shapes(
    model_dir: str | Path, policy: QuantizationPolicy, infos: list[TensorInfo]
) -> None:
    """Validate architecture-specific matrix shapes after exact target coverage."""
    if policy.name != "kolibri1-routed-experts":
        return
    _layers, _experts, hidden, intermediate = _kolibri_dimensions(load_config_json(model_dir))
    for info in infos:
        match = _KOLIBRI_EXPERT.fullmatch(info.name)
        if match is None:
            continue
        expected = (
            (hidden, intermediate)
            if match.group("projection") == "down_proj"
            else (intermediate, hidden)
        )
        if info.shape != expected:
            raise ValueError(f"Kolibri expert {info.name!r} has shape {info.shape}, expected {expected}")
