"""Tests for fused routed-expert layout discovery and logical views."""

from __future__ import annotations

from collections import deque

import pytest
import torch

from mxwave.adapters import resolve_expert_layout
from mxwave.adapters.qwen3_5_moe import POLICY_NAME, build_expert_layout, matches
from mxwave.expert_ir import ExpertBank, ExpertQuantizationLayout, LogicalExpertMatrix
from mxwave.shard import TensorInfo


def _config(
    *,
    num_layers: int = 40,
    num_experts: int = 256,
    hidden_size: int = 2048,
    intermediate_size: int = 512,
) -> dict[str, object]:
    return {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "num_hidden_layers": num_layers,
            "num_experts": num_experts,
            "num_experts_per_tok": 8,
            "hidden_size": hidden_size,
            "moe_intermediate_size": intermediate_size,
        },
    }


def _info(name: str, shape: tuple[int, ...], dtype: str = "BF16") -> TensorInfo:
    elements = 1
    for dimension in shape:
        elements *= dimension
    return TensorInfo(
        name=name,
        shape=shape,
        dtype=dtype,
        data_offsets=(0, elements * 2),
    )


def _bank_infos(
    *,
    num_layers: int = 40,
    num_experts: int = 256,
    hidden_size: int = 2048,
    intermediate_size: int = 512,
) -> list[TensorInfo]:
    stack = "model.language_model.layers"
    infos = [_info("model.language_model.embed_tokens.weight", (1024, hidden_size))]
    for layer_index in range(num_layers):
        prefix = f"{stack}.{layer_index}.mlp.experts"
        infos.extend(
            (
                _info(
                    f"{prefix}.gate_up_proj",
                    (num_experts, intermediate_size * 2, hidden_size),
                ),
                _info(
                    f"{prefix}.down_proj",
                    (num_experts, hidden_size, intermediate_size),
                ),
            )
        )
    return infos


def test_nex_layout_has_exact_logical_and_output_contract() -> None:
    config = _config()
    layout = build_expert_layout(config, _bank_infos())

    assert matches(config)
    assert resolve_expert_layout(config, _bank_infos()).policy_name == POLICY_NAME
    assert layout.architecture == "qwen3_5_moe_text"
    assert layout.policy_name == POLICY_NAME
    assert dict(layout.metadata)["stack_prefix"] == "model.language_model.layers"
    assert layout.source_bank_count == 80
    assert layout.logical_matrix_count == 30_720
    assert layout.output_tensor_count == 61_440
    assert layout.summary() == {
        "architecture": "qwen3_5_moe_text",
        "policy": "qwen3.5-moe-routed-experts",
        "stack_prefix": "model.language_model.layers",
        "num_layers": 40,
        "num_experts": 256,
        "hidden_size": 2048,
        "intermediate_size": 512,
        "source_banks": 80,
        "logical_matrices": 30_720,
        "output_tensors": 61_440,
    }

    matrices = layout.iter_logical_matrices()
    gate = next(matrices)
    up = next(matrices)
    down = next(matrices)
    assert (gate.projection, gate.shape, gate.output_module) == (
        "gate",
        (512, 2048),
        "model.language_model.layers.0.mlp.experts.0.gate_proj",
    )
    assert [spec.name for spec in gate.output_specs()] == [
        "model.language_model.layers.0.mlp.experts.0.gate_proj.weight_packed",
        "model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale",
    ]
    assert [spec.shape for spec in gate.output_specs()] == [(512, 1024), (512, 64)]
    assert (up.projection, up.shape) == ("up", (512, 2048))
    assert [spec.shape for spec in up.output_specs()] == [(512, 1024), (512, 64)]
    assert (down.projection, down.shape) == ("down", (2048, 512))
    assert [spec.shape for spec in down.output_specs()] == [(2048, 256), (2048, 16)]

    last = deque(layout.iter_logical_matrices(), maxlen=1)[0]
    assert (last.layer_index, last.expert_index, last.projection) == (39, 255, "down")
    assert last.output_module == "model.language_model.layers.39.mlp.experts.255.down_proj"


def test_logical_matrices_are_zero_copy_2d_views_in_deterministic_order() -> None:
    config = _config(num_layers=1, num_experts=2, hidden_size=64, intermediate_size=32)
    layout = build_expert_layout(
        config,
        _bank_infos(num_layers=1, num_experts=2, hidden_size=64, intermediate_size=32),
    )
    matrices = list(layout.iter_logical_matrices())
    assert [(item.expert_index, item.projection) for item in matrices] == [
        (0, "gate"),
        (0, "up"),
        (0, "down"),
        (1, "gate"),
        (1, "up"),
        (1, "down"),
    ]

    gate_up_bank = torch.arange(2 * 64 * 64, dtype=torch.float32).reshape(2, 64, 64)
    down_bank = torch.arange(2 * 64 * 32, dtype=torch.float32).reshape(2, 64, 32)
    gate = matrices[3].view(gate_up_bank)
    up = matrices[4].view(gate_up_bank)
    down = matrices[5].view(down_bank)

    assert torch.equal(gate, gate_up_bank[1, :32, :])
    assert torch.equal(up, gate_up_bank[1, 32:, :])
    assert torch.equal(down, down_bank[1, :, :])
    assert gate.untyped_storage().data_ptr() == gate_up_bank.untyped_storage().data_ptr()
    assert up.untyped_storage().data_ptr() == gate_up_bank.untyped_storage().data_ptr()
    assert down.untyped_storage().data_ptr() == down_bank.untyped_storage().data_ptr()


def test_layout_rejects_incomplete_or_misshaped_banks() -> None:
    config = _config(num_layers=2, num_experts=4, hidden_size=64, intermediate_size=32)
    infos = _bank_infos(num_layers=2, num_experts=4, hidden_size=64, intermediate_size=32)
    infos = [info for info in infos if not info.name.endswith("layers.1.mlp.experts.down_proj")]
    with pytest.raises(ValueError, match="exactly one complete decoder expert stack"):
        build_expert_layout(config, infos)

    infos = _bank_infos(num_layers=2, num_experts=4, hidden_size=64, intermediate_size=32)
    bad_name = "model.language_model.layers.1.mlp.experts.down_proj"
    infos = [
        _info(info.name, (4, 32, 64)) if info.name == bad_name else info for info in infos
    ]
    with pytest.raises(ValueError, match=r"down_proj.*expected \(4, 64, 32\)"):
        build_expert_layout(config, infos)


def test_layout_rejects_an_unclassified_additional_expert_stack() -> None:
    config = _config(num_layers=1, num_experts=2, hidden_size=64, intermediate_size=32)
    infos = _bank_infos(num_layers=1, num_experts=2, hidden_size=64, intermediate_size=32)
    infos.append(
        _info(
            "mtp.layers.0.mlp.experts.gate_up_proj",
            (2, 64, 64),
        )
    )

    with pytest.raises(ValueError, match="additional fused expert stacks"):
        build_expert_layout(config, infos)


def test_layout_rejects_wrong_architecture_and_non_mxfp4_width() -> None:
    incompatible = _config(num_layers=1, num_experts=2, hidden_size=64, intermediate_size=32)
    incompatible["architectures"] = ["Qwen3_5ForConditionalGeneration"]
    assert not matches(incompatible)
    with pytest.raises(ValueError, match="No routed-expert layout adapter"):
        resolve_expert_layout(incompatible, [])
    with pytest.raises(ValueError, match="incompatible model config"):
        build_expert_layout(incompatible, [])

    unaligned = _config(num_layers=1, num_experts=2, hidden_size=64, intermediate_size=48)
    with pytest.raises(ValueError, match="divisible by MXFP4 block size 32"):
        build_expert_layout(
            unaligned,
            _bank_infos(
                num_layers=1,
                num_experts=2,
                hidden_size=64,
                intermediate_size=48,
            ),
        )


def test_logical_view_validates_source_shape() -> None:
    config = _config(num_layers=1, num_experts=2, hidden_size=64, intermediate_size=32)
    layout = build_expert_layout(
        config,
        _bank_infos(num_layers=1, num_experts=2, hidden_size=64, intermediate_size=32),
    )
    gate = next(layout.iter_logical_matrices())
    with pytest.raises(ValueError, match="has shape .* expected"):
        gate.view(torch.zeros((2, 32, 64)))


def test_expert_ir_accepts_adapter_defined_bank_and_projection_names() -> None:
    bank = ExpertBank(
        source_name="decoder.blocks.0.custom_bank",
        layer_index=0,
        kind="custom_fused",
        shape=(2, 32, 64),
        dtype="BF16",
    )
    matrix = LogicalExpertMatrix(
        source_name=bank.source_name,
        source_shape=bank.shape,
        layer_index=0,
        expert_index=1,
        projection="custom_projection",
        output_module="decoder.blocks.0.experts.1.custom_proj",
        row_start=0,
        row_stop=32,
    )
    layout = ExpertQuantizationLayout(
        architecture="synthetic_moe",
        policy_name="synthetic-policy",
        banks=(bank,),
        matrices=(matrix,),
        target_patterns=(r"re:^decoder\.blocks\.0\.experts\.\d+\.custom_proj$",),
    )

    assert list(layout.iter_logical_matrices()) == [matrix]
    assert layout.summary()["logical_matrices"] == 1
