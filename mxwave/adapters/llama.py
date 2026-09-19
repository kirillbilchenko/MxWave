"""Dense Llama runtime-operation adapter."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping

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
    """Return whether ``config`` describes a dense Llama text model."""
    return config.get("model_type") == "llama"


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
            "Llama runtime adapter requires exactly one complete decoder stack; "
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
    return RuntimeOperation(
        name=f"{stack}.{layer}.mlp",
        layer_index=layer,
        kind="gated-mlp",
        linear_groups=(
            _group(
                stack,
                layer,
                "mlp.gate_up_proj",
                (("mlp.gate_proj", "gate"), ("mlp.up_proj", "up")),
            ),
            _group(stack, layer, "mlp.down_proj", (("mlp.down_proj", "down"),)),
        ),
        response_points=(ResponsePoint("post_gate"), ResponsePoint("output")),
    )


def _attention_operation(stack: str, layer: int) -> RuntimeOperation:
    return RuntimeOperation(
        name=f"{stack}.{layer}.self_attn",
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
        response_points=(ResponsePoint("attention_output"), ResponsePoint("output")),
    )


def build(
    config: Mapping[str, object],
    tensor_names: Iterable[str],
) -> RuntimeGraph:
    """Build and validate a dense Llama runtime-operation graph."""
    if not matches(config):
        raise ValueError("Llama adapter received an incompatible model config")
    layer_count = config.get("num_hidden_layers")
    if not isinstance(layer_count, int) or layer_count <= 0:
        raise ValueError("Llama config has an invalid num_hidden_layers")

    names = frozenset(tensor_names)
    stack = _stack_prefix(names, layer_count)
    operations = tuple(
        operation
        for layer in range(layer_count)
        for operation in (_mlp_operation(stack, layer), _attention_operation(stack, layer))
    )
    graph = RuntimeGraph(
        architecture="llama",
        adapter_version="1",
        operations=operations,
    )
    missing = sorted(
        checkpoint_name
        for operation in graph.operations
        for checkpoint_name in operation.checkpoint_names
        if checkpoint_name not in names
    )
    if missing:
        raise ValueError(
            f"Llama checkpoint is missing {len(missing)} runtime weight(s): {missing[:5]}"
        )
    return graph
