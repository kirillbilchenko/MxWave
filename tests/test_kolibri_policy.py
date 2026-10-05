"""Kolibri routed-expert conversion and strict checkpoint coverage contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from mxwave.engine import QuantizeConfig, plan_model, quantize_model


def _model(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    config = {
        "model_type": "kolibri1",
        "architectures": ["Kolibri1ForCausalLM"],
        "hidden_size": 128,
        "moe_intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_experts": 2,
        "layer_types": ["sliding_attention", "full_attention"],
    }
    (source / "config.json").write_text(json.dumps(config))
    tensors = {
        f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight":
        torch.randn(128, 128, dtype=torch.bfloat16)
        for layer in range(2)
        for expert in range(2)
        for projection in ("gate_proj", "up_proj", "down_proj")
    }
    tensors.update({
        "model.embed_tokens.weight": torch.randn(16, 128, dtype=torch.bfloat16),
        "lm_head.weight": torch.randn(16, 128, dtype=torch.bfloat16),
        "model.layers.0.mlp.gate.weight": torch.randn(2, 128, dtype=torch.bfloat16),
        "model.layers.0.moe.router.expert_bias": torch.randn(2, dtype=torch.bfloat16),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(128, 128, dtype=torch.bfloat16),
        "model.layers.0.mlp.shared_experts.gate_proj.weight":
        torch.randn(128, 128, dtype=torch.bfloat16),
        "model.layers.0.post_ffn_norm.weight": torch.randn(128, dtype=torch.bfloat16),
    })
    save_file(tensors, str(source / "model.safetensors"))
    return source


def _config(source: Path, **kwargs: object) -> QuantizeConfig:
    return QuantizeConfig(
        model_dir=source, output_dir=source.parent / "output", device="cpu",
        policy="kolibri1-routed-experts", verbose=False, **kwargs,
    )


def test_kolibri_emission_preserves_nonexperts_and_declares_weight_only(tmp_path: Path) -> None:
    source = _model(tmp_path)
    config = _config(source, method="rtn", verify_sqnr=True)
    plan = plan_model(config)
    assert len(plan.target_names) == 12
    assert len(plan.config_target_modules) == 1
    assert plan.config_target_modules[0].startswith("re:^")
    assert not any("shared_experts" in name for name in plan.target_names)
    quantize_model(config)
    original = load_file(str(source / "model.safetensors"))
    output = load_file(str(source.parent / "output/model.safetensors"))
    for name, tensor in original.items():
        if name not in plan.target_names:
            assert torch.equal(output[name], tensor), name
    cfg = json.loads((source.parent / "output/config.json").read_text())
    group = cfg["quantization_config"]["config_groups"]["group_0"]
    assert "input_activations" not in group
    assert group["targets"] == plan.config_target_modules
    manifest = json.loads((source.parent / "output/mxwave-manifest.json").read_text())
    assert manifest["activation_quantization"] == "none"
    assert "moe=marlin" in manifest["required_backends"]
    assert output["model.layers.0.mlp.experts.0.down_proj.weight_scale"].dtype == torch.uint8


@pytest.mark.parametrize("kind", ["missing", "foreign", "shape", "unknown_module"])
def test_kolibri_plan_rejects_incomplete_or_incompatible_layout(
    tmp_path: Path, kind: str,
) -> None:
    source = _model(tmp_path)
    path = source / "model.safetensors"
    tensors = load_file(str(path))
    name = "model.layers.0.mlp.experts.0.gate_proj.weight"
    if kind == "missing":
        del tensors[name]
    elif kind == "foreign":
        tensors[name.replace("experts.0", "experts.2")] = tensors.pop(name)
    elif kind == "shape":
        tensors[name] = torch.randn(256, 128, dtype=torch.bfloat16)
    else:
        tensors["model.layers.0.unknown.weight"] = torch.ones(128, 128)
    save_file(tensors, str(path))
    with pytest.raises(ValueError):
        plan_model(_config(source))


def test_kolibri_policy_rejects_wrong_architecture_and_fp8_source(tmp_path: Path) -> None:
    source = _model(tmp_path)
    path = source / "config.json"
    config = json.loads(path.read_text())
    config["model_type"] = "qwen3_moe"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Kolibri policy requires"):
        plan_model(_config(source))
    config["model_type"] = "kolibri1"
    config["quantization_config"] = {"quant_method": "fp8"}
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="only plain"):
        plan_model(_config(source))
