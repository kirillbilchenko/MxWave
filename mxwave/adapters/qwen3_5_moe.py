"""Qwen3.5-MoE fused routed-expert layout adapter."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import cast

from ..core import BLOCK_SIZE
from ..expert_ir import (
    ExpertBank,
    ExpertBankKind,
    ExpertQuantizationLayout,
    LogicalExpertMatrix,
)
from ..shard import TensorInfo

POLICY_NAME = "qwen3.5-moe-routed-experts"
_ARCHITECTURE = "Qwen3_5MoeForConditionalGeneration"
_TEXT_MODEL_TYPE = "qwen3_5_moe_text"
_FLOAT_DTYPES = frozenset({"BF16", "F16", "F32"})
_BANK_PATTERN = re.compile(
    r"^(?P<stack>.+\.layers)\.(?P<layer>\d+)\.mlp\.experts\."
    r"(?P<kind>gate_up_proj|down_proj)$"
)

__all__ = ["POLICY_NAME", "build_expert_layout", "matches"]


def matches(config: Mapping[str, object]) -> bool:
    """Return whether ``config`` declares the supported Qwen3.5-MoE family."""
    architectures = config.get("architectures")
    text_config = config.get("text_config")
    return (
        config.get("model_type") == "qwen3_5_moe"
        and isinstance(architectures, list)
        and _ARCHITECTURE in architectures
        and isinstance(text_config, Mapping)
        and text_config.get("model_type") == _TEXT_MODEL_TYPE
    )


def _text_config(config: Mapping[str, object]) -> Mapping[str, object]:
    raw = config.get("text_config")
    if not isinstance(raw, Mapping):
        raise TypeError("Qwen3.5-MoE layout adapter requires a text_config object")
    return cast(Mapping[str, object], raw)


def _positive_int(config: Mapping[str, object], key: str) -> int:
    value = config.get(key)
    if type(value) is not int or value <= 0:
        raise ValueError(f"Qwen3.5-MoE text_config has an invalid {key}")
    return value


def _kind_from_suffix(suffix: str) -> ExpertBankKind:
    if suffix == "gate_up_proj":
        return "gate_up"
    if suffix == "down_proj":
        return "down"
    raise AssertionError(f"Unexpected matched expert bank suffix: {suffix}")


def build_expert_layout(
    config: Mapping[str, object],
    tensor_infos: Iterable[TensorInfo],
) -> ExpertQuantizationLayout:
    """Build a strict routed-expert layout from config and tensor headers.

    This is a header-only operation.  It recognizes the fused bank layout used
    by Nex-N2.5-mini, validates complete coverage, and describes the logical 2D
    matrices expected by stock compressed-tensors/vLLM checkpoints.
    """
    if not matches(config):
        raise ValueError("Qwen3.5-MoE adapter received an incompatible model config")
    text_config = _text_config(config)
    num_layers = _positive_int(text_config, "num_hidden_layers")
    num_experts = _positive_int(text_config, "num_experts")
    hidden_size = _positive_int(text_config, "hidden_size")
    intermediate_size = _positive_int(text_config, "moe_intermediate_size")
    if hidden_size % BLOCK_SIZE or intermediate_size % BLOCK_SIZE:
        raise ValueError(
            "Qwen3.5-MoE expert widths must be divisible by "
            f"MXFP4 block size {BLOCK_SIZE}"
        )

    infos_by_name: dict[str, TensorInfo] = {}
    banks_by_stack: dict[str, dict[tuple[int, ExpertBankKind], TensorInfo]] = defaultdict(dict)
    for info in tensor_infos:
        if info.name in infos_by_name:
            raise ValueError(f"Duplicate source tensor key: {info.name}")
        infos_by_name[info.name] = info
        match = _BANK_PATTERN.fullmatch(info.name)
        if match is None:
            continue
        layer_index = int(match.group("layer"))
        kind = _kind_from_suffix(match.group("kind"))
        key = (layer_index, kind)
        stack_banks = banks_by_stack[match.group("stack")]
        if key in stack_banks:
            raise ValueError(
                f"Qwen3.5-MoE stack {match.group('stack')!r} repeats expert bank {key}"
            )
        stack_banks[key] = info

    expected_keys = {
        (layer_index, kind)
        for layer_index in range(num_layers)
        for kind in ("gate_up", "down")
    }
    matching_stacks = sorted(
        stack for stack, banks in banks_by_stack.items() if set(banks) == expected_keys
    )
    if len(matching_stacks) != 1:
        observed = {stack: len(banks) for stack, banks in sorted(banks_by_stack.items())}
        raise ValueError(
            "Qwen3.5-MoE adapter requires exactly one complete decoder expert stack; "
            f"matching={matching_stacks}, observed_bank_counts={observed}"
        )
    stack_prefix = matching_stacks[0]
    additional_stacks = {
        stack: len(banks)
        for stack, banks in sorted(banks_by_stack.items())
        if stack != stack_prefix and banks
    }
    if additional_stacks:
        raise ValueError(
            "Qwen3.5-MoE adapter found additional fused expert stacks that require "
            f"explicit classification: {additional_stacks}"
        )

    banks: list[ExpertBank] = []
    for layer_index in range(num_layers):
        for kind in ("gate_up", "down"):
            info = banks_by_stack[stack_prefix][(layer_index, kind)]
            if info.dtype not in _FLOAT_DTYPES:
                raise ValueError(
                    f"Expert bank {info.name!r} has unsupported dtype {info.dtype}"
                )
            expected_shape = (
                (num_experts, intermediate_size * 2, hidden_size)
                if kind == "gate_up"
                else (num_experts, hidden_size, intermediate_size)
            )
            if info.shape != expected_shape:
                raise ValueError(
                    f"Expert bank {info.name!r} has shape {info.shape}, expected {expected_shape}"
                )
            banks.append(
                ExpertBank(
                    source_name=info.name,
                    layer_index=layer_index,
                    kind=kind,
                    shape=expected_shape,
                    dtype=info.dtype,
                )
            )

    matrices: list[LogicalExpertMatrix] = []
    banks_by_key = {(bank.layer_index, bank.kind): bank for bank in banks}
    for layer_index in range(num_layers):
        gate_up = banks_by_key[(layer_index, "gate_up")]
        down = banks_by_key[(layer_index, "down")]
        for expert_index in range(num_experts):
            output_prefix = f"{stack_prefix}.{layer_index}.mlp.experts.{expert_index}"
            matrices.extend(
                (
                    LogicalExpertMatrix(
                        source_name=gate_up.source_name,
                        source_shape=gate_up.shape,
                        layer_index=layer_index,
                        expert_index=expert_index,
                        projection="gate",
                        output_module=f"{output_prefix}.gate_proj",
                        row_start=0,
                        row_stop=intermediate_size,
                    ),
                    LogicalExpertMatrix(
                        source_name=gate_up.source_name,
                        source_shape=gate_up.shape,
                        layer_index=layer_index,
                        expert_index=expert_index,
                        projection="up",
                        output_module=f"{output_prefix}.up_proj",
                        row_start=intermediate_size,
                        row_stop=intermediate_size * 2,
                    ),
                    LogicalExpertMatrix(
                        source_name=down.source_name,
                        source_shape=down.shape,
                        layer_index=layer_index,
                        expert_index=expert_index,
                        projection="down",
                        output_module=f"{output_prefix}.down_proj",
                        row_start=0,
                        row_stop=hidden_size,
                    ),
                )
            )

    layers = "|".join(str(index) for index in range(num_layers))
    target_pattern = (
        r"re:^(?:model\.)?language_model(?:\.model)?\.layers\."
        f"(?:{layers})\\.mlp\\.experts\\.\\d+"
        r"\.(?:gate_proj|up_proj|down_proj)$"
    )
    return ExpertQuantizationLayout(
        architecture=_TEXT_MODEL_TYPE,
        policy_name=POLICY_NAME,
        banks=tuple(banks),
        matrices=tuple(matrices),
        target_patterns=(target_pattern,),
        ignored_patterns=(r"re:.*mtp.*", r"re:.*hyper.*"),
        metadata=(
            ("stack_prefix", stack_prefix),
            ("num_layers", num_layers),
            ("num_experts", num_experts),
            ("hidden_size", hidden_size),
            ("intermediate_size", intermediate_size),
        ),
    )
