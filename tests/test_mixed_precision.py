"""Tests for fusion-safe mixed MXFP4/FP8 composition."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pytest
import torch
from safetensors.torch import save_file

from mxwave.mixed_precision import (
    build_fp8_composition_plan,
    compose_fp8_checkpoint,
    quantize_fp8_channelwise,
)
from mxwave.output import build_quantization_config
from mxwave.precision_budget import build_precision_budget_plan, load_bucket_modules
from mxwave.shard import ShardFile, shard_tensor_info


def _write_checkpoint(root: Path, tensors: dict[str, torch.Tensor]) -> None:
    root.mkdir()
    shard_name = "model.safetensors"
    save_file(tensors, str(root / shard_name))
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {name: shard_name for name in tensors},
            }
        )
    )


def _config(family: Literal["qwen", "llama"] = "qwen") -> dict[str, object]:
    if family == "llama":
        return {
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "num_hidden_layers": 6,
        }
    return {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": {
            "model_type": "qwen3_5_text",
            "num_hidden_layers": 6,
            "layer_types": ["full_attention"] * 6,
        },
    }


def _target_modules(family: Literal["qwen", "llama"] = "qwen") -> list[str]:
    modules: list[str] = []
    stack = "model.language_model.layers" if family == "qwen" else "model.layers"
    for layer in range(6):
        prefix = f"{stack}.{layer}"
        modules.extend(
            [
                f"{prefix}.mlp.gate_proj",
                f"{prefix}.mlp.up_proj",
                f"{prefix}.mlp.down_proj",
                f"{prefix}.self_attn.q_proj",
                f"{prefix}.self_attn.k_proj",
                f"{prefix}.self_attn.v_proj",
                f"{prefix}.self_attn.o_proj",
            ]
        )
    return modules


def _make_checkpoints(
    tmp_path: Path,
    family: Literal["qwen", "llama"] = "qwen",
) -> tuple[Path, Path]:
    primary = tmp_path / "primary"
    dense = tmp_path / "dense"
    targets = _target_modules(family)
    model_root = "model.language_model" if family == "qwen" else "model"
    ignored = [
        f"{model_root}.embed_tokens",
        f"{model_root}.norm",
        "lm_head",
    ]
    primary_tensors: dict[str, torch.Tensor] = {
        f"{model_root}.embed_tokens.weight": torch.randn(8, 32, dtype=torch.bfloat16),
        f"{model_root}.norm.weight": torch.ones(32, dtype=torch.bfloat16),
        "lm_head.weight": torch.randn(8, 32, dtype=torch.bfloat16),
    }
    dense_tensors = dict(primary_tensors)
    for index, module in enumerate(targets):
        primary_tensors[f"{module}.weight_packed"] = torch.full(
            (2, 16), index % 255, dtype=torch.uint8
        )
        primary_tensors[f"{module}.weight_scale"] = torch.full((2, 1), 127, dtype=torch.uint8)
        dense_tensors[f"{module}.weight"] = torch.randn(2, 32, dtype=torch.bfloat16)
    _write_checkpoint(primary, primary_tensors)
    _write_checkpoint(dense, dense_tensors)

    primary_config = _config(family)
    primary_config["quantization_config"] = build_quantization_config(targets, ignored)
    (primary / "config.json").write_text(json.dumps(primary_config))
    (dense / "config.json").write_text(json.dumps(_config(family)))
    (primary / "tokenizer.json").write_text('{"version":"1"}')
    manifest = {
        "manifest_version": 1,
        "producer": {"name": "MxWave", "version": "test"},
        "source": {"repository": "test/model", "revision": "abc"},
        "method": "mse",
        "weight_scale_selection": "mse-activation-block-hessian",
        "policy": "test",
        "source_data_bytes": 100_000,
        "target_source_bytes": len(targets) * 128,
        "source_tensors": len(primary_tensors),
        "target_tensors": len(targets),
        "passthrough_tensors": len(ignored),
        "activation_calibration": {
            "objective": "block-hessian",
            "num_sequences": 2,
            "weighted_tensors": len(targets),
        },
    }
    (primary / "mxwave-manifest.json").write_text(json.dumps(manifest))
    return primary, dense


def test_channelwise_fp8_quantization_handles_zero_rows() -> None:
    weight = torch.tensor(
        [[0.0, 0.0, 0.0], [-2.0, -0.25, 1.0]],
        dtype=torch.bfloat16,
    )

    quantized, scale = quantize_fp8_channelwise(weight)

    assert quantized.dtype == torch.float8_e4m3fn
    assert quantized.device.type == "cpu"
    assert scale.dtype == torch.float32
    assert scale.shape == (2, 1)
    assert torch.isfinite(scale).all()
    reconstructed = quantized.float() * scale
    assert torch.equal(reconstructed[0], torch.zeros(3))
    assert torch.allclose(reconstructed[1], weight[1].float(), atol=0.02, rtol=0.02)


def test_composition_rejects_split_runtime_fusion(tmp_path: Path) -> None:
    primary, dense = _make_checkpoints(tmp_path)
    with pytest.raises(ValueError, match="splits fused runtime group"):
        build_fp8_composition_plan(
            primary,
            dense,
            ["model.language_model.layers.1.self_attn.q_proj"],
        )


def test_fp8_composition_emits_verified_mixed_checkpoint(tmp_path: Path) -> None:
    primary, dense = _make_checkpoints(tmp_path)
    output = tmp_path / "output"
    selected = [
        f"model.language_model.layers.1.self_attn.{projection}"
        for projection in ("q_proj", "k_proj", "v_proj")
    ]
    plan = build_fp8_composition_plan(primary, dense, selected)

    manifest = compose_fp8_checkpoint(plan, output, verbose=False)

    info = shard_tensor_info(ShardFile(output / "model.safetensors", {}))
    for module in selected:
        assert info[f"{module}.weight"].dtype == "F8_E4M3"
        assert info[f"{module}.weight_scale"].shape == (2, 1)
        assert f"{module}.weight_packed" not in info
    config = json.loads((output / "config.json").read_text())["quantization_config"]
    assert config["format"] == "mixed-precision"
    assert config["config_groups"]["group_0"]["format"] == "mxfp4-pack-quantized"
    assert config["config_groups"]["group_1"]["format"] == "float-quantized"
    assert config["config_groups"]["group_1"]["weights"]["strategy"] == "channel"
    assert manifest["fp8_target_tensors"] == 3
    assert manifest["mxfp4_target_tensors"] == len(_target_modules()) - 3
    assert manifest["target_source_bytes"] == len(_target_modules()) * 128
    assert plan.premium_bytes > 0


def test_precision_budget_builds_twelve_bounded_fusion_safe_buckets(
    tmp_path: Path,
) -> None:
    primary, dense = _make_checkpoints(tmp_path)

    plan = build_precision_budget_plan(
        primary,
        dense,
        layers_per_bucket=1,
        max_premium_bytes=10_000,
    )

    assert len(plan.buckets) == 12
    assert {bucket.family for bucket in plan.buckets} == {
        "mlp-input",
        "mlp-output",
        "sequence-input",
        "sequence-output",
    }
    sequence_inputs = [bucket for bucket in plan.buckets if bucket.family == "sequence-input"]
    assert all(len(bucket.selected_modules) == 3 for bucket in sequence_inputs)
    document = plan.as_dict()
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(document))
    first = plan.buckets[0]
    assert load_bucket_modules(plan_path, first.name) == first.selected_modules
    assert len(document["plan_sha256"]) == 64


def test_precision_budget_uses_registered_llama_adapter(tmp_path: Path) -> None:
    primary, dense = _make_checkpoints(tmp_path, "llama")

    plan = build_precision_budget_plan(
        primary,
        dense,
        layers_per_bucket=1,
        max_premium_bytes=10_000,
    )

    assert len(plan.buckets) == 12
    sequence_inputs = [bucket for bucket in plan.buckets if bucket.family == "sequence-input"]
    assert all(len(bucket.selected_modules) == 3 for bucket in sequence_inputs)
    assert all(
        module.startswith("model.layers.")
        for bucket in plan.buckets
        for module in bucket.selected_modules
    )
