"""Tests for runtime-operation IR and architecture adapter discovery."""

from __future__ import annotations

import pytest

from mxwave.runtime_adapters import registered_runtime_adapters, resolve_runtime_graph
from mxwave.runtime_ir import (
    ResponsePoint,
    RuntimeGraph,
    RuntimeLinearGroup,
    RuntimeOperation,
    RuntimeWeight,
)


def _qwen_config() -> dict[str, object]:
    return {
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "num_hidden_layers": 4,
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
        },
    }


def _qwen_tensor_names() -> set[str]:
    stack = "model.language_model.layers"
    names = {
        "model.language_model.embed_tokens.weight",
        # Qwen checkpoints can carry a separate one-layer MTP stack. It must
        # not be mistaken for the complete text decoder stack.
        "mtp.layers.0.mlp.gate_proj.weight",
    }
    for layer in range(4):
        names.update(
            f"{stack}.{layer}.mlp.{projection}.weight"
            for projection in ("gate_proj", "up_proj", "down_proj")
        )
        if layer == 3:
            names.update(
                f"{stack}.{layer}.self_attn.{projection}.weight"
                for projection in ("q_proj", "k_proj", "v_proj", "o_proj")
            )
        else:
            names.update(
                f"{stack}.{layer}.linear_attn.{projection}.weight"
                for projection in (
                    "in_proj_qkv",
                    "in_proj_z",
                    "in_proj_b",
                    "in_proj_a",
                    "out_proj",
                )
            )
    return names


def _llama_config() -> dict[str, object]:
    return {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "num_hidden_layers": 2,
    }


def _llama_tensor_names() -> set[str]:
    names = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    for layer in range(2):
        names.update(
            f"model.layers.{layer}.mlp.{projection}.weight"
            for projection in ("gate_proj", "up_proj", "down_proj")
        )
        names.update(
            f"model.layers.{layer}.self_attn.{projection}.weight"
            for projection in ("q_proj", "k_proj", "v_proj", "o_proj")
        )
    return names


def test_qwen_adapter_builds_fused_runtime_groups_and_response_boundaries() -> None:
    graph = resolve_runtime_graph(_qwen_config(), _qwen_tensor_names())

    assert graph.architecture == "qwen3_5_text"
    assert graph.adapter_version == "1"
    assert len(graph.operations) == 8

    gate = graph.linear_group_for_weight(
        "model.language_model.layers.0.mlp.gate_proj.weight"
    )
    assert gate.runtime_name.endswith(".mlp.gate_up_proj")
    assert [member.role for member in gate.members] == ["gate", "up"]
    assert gate.requires_shared_tensor_scale

    qkvz = graph.linear_group_for_weight(
        "model.language_model.layers.0.linear_attn.in_proj_z.weight"
    )
    assert qkvz.runtime_name.endswith(".linear_attn.in_proj_qkvz")
    assert [member.role for member in qkvz.members] == ["qkv", "z"]

    ba = graph.linear_group_for_weight(
        "model.language_model.layers.0.linear_attn.in_proj_a.weight"
    )
    assert ba.runtime_name.endswith(".linear_attn.in_proj_ba")
    assert [member.role for member in ba.members] == ["b", "a"]

    qkv = graph.linear_group_for_weight(
        "model.language_model.layers.3.self_attn.k_proj.weight"
    )
    assert qkv.runtime_name.endswith(".self_attn.qkv_proj")
    assert [member.role for member in qkv.members] == ["query", "key", "value"]

    gdn = graph.operation_for_weight(
        "model.language_model.layers.0.linear_attn.out_proj.weight"
    )
    assert gdn.kind == "gated-delta"
    assert [(point.name, point.kind) for point in gdn.response_points] == [
        ("recurrent_state", "state"),
        ("post_gate", "activation"),
        ("output", "activation"),
    ]
    assert graph.as_dict()["operations"][0]["kind"] == "gated-mlp"


def test_qwen_adapter_rejects_incomplete_runtime_group() -> None:
    names = _qwen_tensor_names()
    names.remove("model.language_model.layers.1.linear_attn.in_proj_a.weight")

    with pytest.raises(ValueError, match="missing 1 runtime weight"):
        resolve_runtime_graph(_qwen_config(), names)


def test_llama_adapter_builds_dense_fused_runtime_groups() -> None:
    graph = resolve_runtime_graph(_llama_config(), _llama_tensor_names())

    assert graph.architecture == "llama"
    assert graph.adapter_version == "1"
    assert len(graph.operations) == 4
    gate_up = graph.linear_group_for_weight("model.layers.1.mlp.up_proj.weight")
    assert gate_up.runtime_name == "model.layers.1.mlp.gate_up_proj"
    assert [member.role for member in gate_up.members] == ["gate", "up"]
    qkv = graph.linear_group_for_weight("model.layers.0.self_attn.v_proj.weight")
    assert qkv.runtime_name == "model.layers.0.self_attn.qkv_proj"
    assert [member.role for member in qkv.members] == ["query", "key", "value"]


def test_llama_adapter_rejects_incomplete_runtime_group() -> None:
    names = _llama_tensor_names()
    names.remove("model.layers.0.self_attn.v_proj.weight")

    with pytest.raises(ValueError, match="missing 1 runtime weight"):
        resolve_runtime_graph(_llama_config(), names)


def test_runtime_adapter_registry_has_stable_unique_names() -> None:
    adapters = registered_runtime_adapters()

    assert [adapter.name for adapter in adapters] == ["qwen3_5_text", "llama"]
    assert len({adapter.name for adapter in adapters}) == len(adapters)


def test_unknown_architecture_requires_an_explicit_adapter() -> None:
    with pytest.raises(ValueError, match="Registered adapters.*qwen3_5_text.*llama"):
        resolve_runtime_graph({"model_type": "unknown"}, {"model.layers.0.weight"})


def test_runtime_ir_rejects_weight_owned_by_multiple_operations() -> None:
    member = RuntimeWeight("model.layers.0.proj.weight", "projection")
    first = RuntimeOperation(
        name="model.layers.0.first",
        layer_index=0,
        kind="gated-mlp",
        linear_groups=(RuntimeLinearGroup("model.layers.0.first.proj", (member,)),),
        response_points=(ResponsePoint("output"),),
    )
    second = RuntimeOperation(
        name="model.layers.0.second",
        layer_index=0,
        kind="self-attention",
        linear_groups=(RuntimeLinearGroup("model.layers.0.second.proj", (member,)),),
        response_points=(ResponsePoint("output"),),
    )

    with pytest.raises(ValueError, match="belongs to both"):
        RuntimeGraph("test", "1", (first, second))
