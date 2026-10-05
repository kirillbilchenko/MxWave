"""Verify mixed-checkpoint assembly preserves payloads and rejects unsafe schemas."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from mxwave.engine import QuantizeConfig, quantize_model
from mxwave.shard import ShardFile, tensor_payload_sha256


def _fixtures(tmp_path: Path) -> tuple[Path, Path]:
    source, experts, reference = (tmp_path / name for name in ("bf16", "experts", "fp8"))
    source.mkdir()
    reference.mkdir()
    config = {
        "model_type": "kolibri1", "architectures": ["Kolibri1ForCausalLM"],
        "hidden_size": 128, "moe_intermediate_size": 128,
        "num_hidden_layers": 1, "num_experts": 2, "num_experts_per_tok": 1,
        "layer_types": ["full_attention"], "vocab_size": 16,
        "num_attention_heads": 1, "num_key_value_heads": 1, "head_dim": 128,
    }
    tensors = {
        f"model.layers.0.mlp.experts.{expert}.{projection}.weight":
        torch.randn(128, 128, dtype=torch.bfloat16)
        for expert in range(2) for projection in ("gate_proj", "up_proj", "down_proj")
    }
    for kind, projections in (
        ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
        ("mlp.shared_experts", ("gate_proj", "up_proj", "down_proj")),
    ):
        for projection in projections:
            tensors[f"model.layers.0.{kind}.{projection}.weight"] = (
                torch.randn(128, 128, dtype=torch.bfloat16)
            )
    tensors["model.layers.0.mlp.gate.weight"] = torch.randn(2, 128, dtype=torch.bfloat16)
    tensors["model.layers.0.moe.router.expert_bias"] = torch.randn(2, dtype=torch.bfloat16)
    tensors["lm_head.weight"] = torch.randn(16, 128, dtype=torch.bfloat16)
    tensors["model.embed_tokens.weight"] = torch.randn(16, 128, dtype=torch.bfloat16)
    (source / "config.json").write_text(json.dumps(config))
    save_file(tensors, str(source / "model.safetensors"))
    quantize_model(QuantizeConfig(
        model_dir=source, output_dir=experts, device="cpu",
        policy="kolibri1-routed-experts", method="rtn", verbose=False,
    ))
    gold = {}
    for name, value in tensors.items():
        if ".mlp.experts." in name:
            continue
        if "self_attn." in name or "shared_experts." in name:
            gold[name] = value.to(torch.float8_e4m3fn)
            gold[name.removesuffix(".weight") + ".weight_scale_inv"] = torch.tensor([[0.125]])
        else:
            # Deliberately different bytes prove the chosen reference is preserved.
            gold[name] = value + 0.125
    save_file(gold, str(reference / "model.safetensors"))
    config["quantization_config"] = {
        "quant_method": "fp8", "activation_scheme": "dynamic", "weight_block_size": [128, 128],
        "modules_to_not_convert": ["model.layers.0.mlp.gate"],
    }
    (reference / "config.json").write_text(json.dumps(config))
    return experts, reference


def _script(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parent.parent / "scripts"))
    return importlib.import_module("assemble_kolibri_mixed")


def test_mixed_assembly_verifies_all_payloads_including_renamed_fp8_scales(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _script(monkeypatch)
    experts, reference = _fixtures(tmp_path)
    config, payloads = script.plan(experts, reference)
    output = tmp_path / "mixed"
    result = script.assemble(experts, reference, output)
    assert result["status"] == "complete"
    assert result["payload_counts"] == {"mxfp4_expert": 12, "fp8_scale": 7, "fp8_backbone": 11}
    assert config["quantization_config"]["format"] == "mixed-precision"
    groups = config["quantization_config"]["config_groups"]
    assert len(groups["fp8_backbone"]["targets"]) == 7
    assert not groups["mxfp4_experts"].get("input_activations")
    for payload in payloads:
        assert tensor_payload_sha256(ShardFile(payload.shard, {}), payload.info.name) == (
            tensor_payload_sha256(ShardFile(output / "model.safetensors", {}), payload.name)
        )
    with pytest.raises(FileExistsError):
        script.assemble(experts, reference, output)


@pytest.mark.parametrize("kind", ["scale_dtype", "scale_shape", "missing_scale", "format"])
def test_mixed_plan_rejects_invalid_fp8_schema_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str,
) -> None:
    script = _script(monkeypatch)
    experts, reference = _fixtures(tmp_path)
    path = reference / "model.safetensors"
    tensors = load_file(str(path))
    scale = "model.layers.0.self_attn.q_proj.weight_scale_inv"
    if kind == "scale_dtype":
        tensors[scale] = tensors[scale].to(torch.uint8)
    elif kind == "scale_shape":
        tensors[scale] = torch.ones(2, 1)
    elif kind == "missing_scale":
        del tensors[scale]
    else:
        config_path = reference / "config.json"
        config = json.loads(config_path.read_text())
        config["quantization_config"]["weight_block_size"] = [32, 32]
        config_path.write_text(json.dumps(config))
    save_file(tensors, str(path))
    output = tmp_path / "mixed"
    with pytest.raises((ValueError, KeyError)):
        script.assemble(experts, reference, output)
    assert not output.exists()
