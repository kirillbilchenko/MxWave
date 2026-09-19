"""Qwen3.5 dense runtime-operation adapter."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import cast

from ..runtime_ir import (
    ResponsePoint,
    RuntimeGraph,
    RuntimeLinearGroup,
    RuntimeOperation,
    RuntimeWeight,
)

_STACK_PATTERN = re.compile(
    r"^(?P<stack>.+\.layers)\.(?P<layer>\d+)\.mlp\.gate_proj\.weight$"
)


def matches(config: Mapping[str, object]) -> bool:
    """Return whether ``config`` describes a dense Qwen3.5 text backbone."""
    text_config = config.get("text_config")
    if not isinstance(text_config, Mapping):
        return False
    model_type = text_config.get("model_type", "qwen3_5_text")
    return config.get("model_type") == "qwen3_5" and model_type == "qwen3_5_text"


def _text_config(config: Mapping[str, object]) -> Mapping[str, object]:
    raw = config.get("text_config")
    if not isinstance(raw, Mapping):
        raise TypeError("Qwen3.5 runtime adapter requires a text_config object")
    return cast(Mapping[str, object], raw)


def _stack_prefix(tensor_names: frozenset[str], layer_count: int) -> str:
    layers_by_stack: dict[str, set[int]] = defaultdict(set)
    for name in tensor_names:
        match = _STACK_PATTERN.fullmatch(name)
        if match is not None:
            layers_by_stack[match.group("stack")].add(int(match.group("layer")))

    expected_layers = set(range(layer_count))
    matching_stacks = sorted(
        stack for stack, layers in layers_by_stack.items() if layers == expected_layers
    )
    if len(matching_stacks) != 1:
        raise ValueError(
            "Qwen3.5 runtime adapter requires exactly one complete decoder stack; "
            f"matching={matching_stacks}, observed={sorted(layers_by_stack)}"
        )
    return matching_stacks[0]


def _weight(stack: str, layer: int, suffix: str, role: str) -> RuntimeWeight:
    return RuntimeWeight(
        checkpoint_name=f"{stack}.{layer}.{suffix}.weight",
        role=role,
    )


def _group(
    stack: str,
    layer: int,
    runtime_suffix: str,
    members: tuple[tuple[str, str], ...],
) -> RuntimeLinearGroup:
    return RuntimeLinearGroup(
        runtime_name=f"{stack}.{layer}.{runtime_suffix}",
        members=tuple(
            _weight(stack, layer, checkpoint_suffix, role)
            for checkpoint_suffix, role in members
        ),
    )


def _mlp_operation(stack: str, layer: int) -> RuntimeOperation:
    prefix = f"{stack}.{layer}.mlp"
    return RuntimeOperation(
        name=prefix,
        layer_index=layer,
        kind="gated-mlp",
        linear_groups=(
            _group(
                stack,
                layer,
                "mlp.gate_up_proj",
                (("mlp.gate_proj", "gate"), ("mlp.up_proj", "up")),
            ),
            _group(
                stack,
                layer,
                "mlp.down_proj",
                (("mlp.down_proj", "down"),),
            ),
        ),
        response_points=(
            ResponsePoint("post_gate"),
            ResponsePoint("output"),
        ),
    )


def _attention_operation(stack: str, layer: int) -> RuntimeOperation:
    prefix = f"{stack}.{layer}.self_attn"
    return RuntimeOperation(
        name=prefix,
        layer_index=layer,
        kind="self-attention",
        linear_groups=(
            _group(
                stack,
                layer,
                "self_attn.qkv_proj",
                (
                    ("self_attn.q_proj", "query"),
                    ("self_attn.k_proj", "key"),
                    ("self_attn.v_proj", "value"),
                ),
            ),
            _group(
                stack,
                layer,
                "self_attn.o_proj",
                (("self_attn.o_proj", "output"),),
            ),
        ),
        response_points=(
            ResponsePoint("attention_output"),
            ResponsePoint("output"),
        ),
    )


def _gated_delta_operation(stack: str, layer: int) -> RuntimeOperation:
    prefix = f"{stack}.{layer}.linear_attn"
    return RuntimeOperation(
        name=prefix,
        layer_index=layer,
        kind="gated-delta",
        linear_groups=(
            _group(
                stack,
                layer,
                "linear_attn.in_proj_qkvz",
                (
                    ("linear_attn.in_proj_qkv", "qkv"),
                    ("linear_attn.in_proj_z", "z"),
                ),
            ),
            _group(
                stack,
                layer,
                "linear_attn.in_proj_ba",
                (
                    ("linear_attn.in_proj_b", "b"),
                    ("linear_attn.in_proj_a", "a"),
                ),
            ),
            _group(
                stack,
                layer,
                "linear_attn.out_proj",
                (("linear_attn.out_proj", "output"),),
            ),
        ),
        response_points=(
            ResponsePoint("recurrent_state", kind="state"),
            ResponsePoint("post_gate"),
            ResponsePoint("output"),
        ),
    )


def build(
    config: Mapping[str, object],
    tensor_names: Iterable[str],
) -> RuntimeGraph:
    """Build and validate the dense Qwen3.5 runtime-operation graph."""
    if not matches(config):
        raise ValueError("Qwen3.5 adapter received an incompatible model config")
    text_config = _text_config(config)
    layer_count = text_config.get("num_hidden_layers")
    layer_types = text_config.get("layer_types")
    if not isinstance(layer_count, int) or layer_count <= 0:
        raise ValueError("Qwen3.5 text_config has an invalid num_hidden_layers")
    if not isinstance(layer_types, list) or len(layer_types) != layer_count:
        raise ValueError("Qwen3.5 text_config has an invalid layer_types list")
    if any(layer_type not in ("full_attention", "linear_attention") for layer_type in layer_types):
        raise ValueError("Qwen3.5 text_config contains an unsupported layer type")

    names = frozenset(tensor_names)
    stack = _stack_prefix(names, layer_count)
    operations: list[RuntimeOperation] = []
    for layer, layer_type in enumerate(layer_types):
        operations.append(_mlp_operation(stack, layer))
        operations.append(
            _attention_operation(stack, layer)
            if layer_type == "full_attention"
            else _gated_delta_operation(stack, layer)
        )

    graph = RuntimeGraph(
        architecture="qwen3_5_text",
        adapter_version="1",
        operations=tuple(operations),
    )
    missing = sorted(
        checkpoint_name
        for operation in graph.operations
        for checkpoint_name in operation.checkpoint_names
        if checkpoint_name not in names
    )
    if missing:
        raise ValueError(
            f"Qwen3.5 checkpoint is missing {len(missing)} runtime weight(s): {missing[:5]}"
        )
    return graph
