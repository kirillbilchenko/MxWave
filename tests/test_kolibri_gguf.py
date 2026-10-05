"""Check complete Kolibri mapping and decode exported bytes with the independent GGUF reader."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from mxwave.core import dequant_mxfp4
from mxwave.kolibri_gguf import (
    _chunks,
    _verify_output,
    _write_tensors,
    export_kolibri_gguf,
    plan_kolibri_gguf,
)


def _fixture(directory: Path) -> tuple[Path, dict[str, torch.Tensor]]:
    model = directory / "model"
    model.mkdir()
    config: dict[str, Any] = {
        "model_type": "kolibri1",
        "architectures": ["Kolibri1ForCausalLM"],
        "hidden_size": 160,
        "num_hidden_layers": 1,
        "num_experts": 12,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 32,
        "shared_expert_intermediate_size": 32,
        "vocab_size": 320,
        "num_attention_heads": 3,
        "num_key_value_heads": 1,
        "head_dim": 128,
        "layer_types": ["sliding_attention"],
        "hidden_act": "silu",
        "max_position_embeddings": 512,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "norm_topk_prob": False,
        "sliding_window": 64,
        "eos_token_id": 257,
        "pad_token_id": 256,
        "bos_token_id": None,
        "attention_bias": False,
        "tie_word_embeddings": False,
    }
    prefix = "model.layers.0."
    tensors = {
        "model.embed_tokens.weight": torch.ones(320, 160, dtype=torch.bfloat16),
        "model.norm.weight": torch.full((160,), 0.125, dtype=torch.bfloat16),
        "lm_head.weight": torch.ones(320, 160, dtype=torch.bfloat16),
        prefix + "mlp.gate.weight": torch.ones(12, 160, dtype=torch.float32),
        prefix + "moe.router.expert_bias": torch.arange(12, dtype=torch.float32),
    }
    for suffix, size in (
        ("input_layernorm.weight", 160),
        ("post_attention_layernorm.weight", 160),
        ("post_attn_norm.weight", 160),
        ("post_ffn_norm.weight", 160),
        ("self_attn.q_norm.weight", 128),
        ("self_attn.k_norm.weight", 128),
    ):
        tensors[prefix + suffix] = torch.ones(size, dtype=torch.bfloat16)
    fp_modules = []
    for suffix, shape in (
        ("self_attn.q_proj", (384, 160)),
        ("self_attn.k_proj", (128, 160)),
        ("self_attn.v_proj", (128, 160)),
        ("self_attn.o_proj", (160, 384)),
        ("mlp.shared_experts.gate_proj", (32, 160)),
        ("mlp.shared_experts.up_proj", (32, 160)),
        ("mlp.shared_experts.down_proj", (160, 32)),
    ):
        name = prefix + suffix
        tensors[name + ".weight"] = torch.ones(shape, dtype=torch.float8_e4m3fn)
        scale_shape = tuple((dimension + 127) // 128 for dimension in shape)
        tensors[name + ".weight_scale"] = torch.arange(
            1,
            int(np.prod(scale_shape)) + 1,
            dtype=torch.float32,
        ).reshape(scale_shape)
        fp_modules.append(name)
    ignore = [
        name.removesuffix(".weight")
        for name in tensors
        if name.endswith(".weight") and name.removesuffix(".weight") not in fp_modules
    ]
    for expert in range(12):
        for projection, shape in (
            ("gate_proj", (32, 160)),
            ("up_proj", (32, 160)),
            ("down_proj", (160, 32)),
        ):
            module = prefix + f"mlp.experts.{expert}.{projection}"
            tensors[module + ".weight_packed"] = torch.full(
                (shape[0], shape[1] // 2),
                expert | ((15 - expert) << 4),
                dtype=torch.uint8,
            )
            tensors[module + ".weight_scale"] = torch.full(
                (shape[0], shape[1] // 32),
                127,
                dtype=torch.uint8,
            )
    config["quantization_config"] = {
        "quant_method": "compressed-tensors",
        "format": "mixed-precision",
        "config_groups": {
            "experts": {
                "format": "mxfp4-pack-quantized",
                "targets": [
                    r"re:^model\.layers\.\d+\.mlp\.experts\.\d+\.(?:gate_proj|up_proj|down_proj)$"
                ],
                "weights": {
                    "num_bits": 4,
                    "type": "float",
                    "symmetric": True,
                    "group_size": 32,
                    "strategy": "group",
                    "dynamic": False,
                    "scale_dtype": "torch.uint8",
                },
            },
            "backbone": {
                "format": "float-quantized",
                "targets": fp_modules,
                "weights": {
                    "num_bits": 8,
                    "type": "float",
                    "symmetric": True,
                    "dynamic": False,
                    "strategy": "block",
                    "block_structure": [128, 128],
                },
            },
        },
        "ignore": ignore,
    }
    (model / "config.json").write_text(json.dumps(config))
    save_file(tensors, str(model / "model.safetensors"))
    manifest = {
        "payload_sha256": {
            name: hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()
            for name, value in tensors.items()
        }
    }
    (model / "mxwave-manifest.json").write_text(json.dumps(manifest))
    return model, tensors


def test_header_only_plan_preserves_numeric_expert_order(tmp_path: Path) -> None:
    model, _ = _fixture(tmp_path)
    plan = plan_kolibri_gguf(model)
    assert len(plan.tensors) == 21
    assert plan.source_payloads == 97
    gate = next(item for item in plan.tensors if item.name == "blk.0.ffn_gate_exps.weight")
    assert gate.shape == (12, 32, 160)
    assert [part[0].info.name.split(".")[5] for part in gate.parts] == [str(i) for i in range(12)]
    destination = tmp_path / "output.gguf"
    result = export_kolibri_gguf(model, destination, dry_run=True)
    assert result["status"] == "planned"
    assert result["payload_bytes"] == sum(item.nbytes for item in plan.tensors)
    assert not destination.exists()


@pytest.mark.parametrize(
    "missing",
    [
        "model.layers.0.moe.router.expert_bias",
        "model.layers.0.post_ffn_norm.weight",
        "model.layers.0.mlp.experts.10.gate_proj.weight_scale",
    ],
)
def test_incomplete_kolibri_checkpoint_is_rejected(tmp_path: Path, missing: str) -> None:
    model, tensors = _fixture(tmp_path)
    del tensors[missing]
    save_file(tensors, str(model / "model.safetensors"))
    with pytest.raises(ValueError, match="Missing source"):
        plan_kolibri_gguf(model)


def test_fp8_block_scales_are_multipliers_and_cover_partial_edge_blocks(tmp_path: Path) -> None:
    model, original = _fixture(tmp_path)
    plan = plan_kolibri_gguf(model)
    tensor = next(item for item in plan.tensors if item.name == "blk.0.attn_q.weight")
    expected = json.loads((model / "mxwave-manifest.json").read_text())["payload_sha256"]
    with (model / "model.safetensors").open("rb") as stream:
        raw = b"".join(_chunks(tensor, {model / "model.safetensors": stream}, expected))
    restored = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(384, 160)
    scales = original["model.layers.0.self_attn.q_proj.weight_scale"]
    # Independently index block IDs, including the second-column partial block.
    correct = scales[torch.arange(384)[:, None] // 128, torch.arange(160)[None, :] // 128]
    torch.testing.assert_close(restored.float(), correct, rtol=0, atol=0)


def test_changed_source_payload_is_rejected(tmp_path: Path) -> None:
    model, _ = _fixture(tmp_path)
    plan = plan_kolibri_gguf(model)
    tensors = load_file(str(model / "model.safetensors"))
    tensors["model.embed_tokens.weight"][0, 0] += 1
    save_file(tensors, str(model / "model.safetensors"))
    expected = json.loads((model / "mxwave-manifest.json").read_text())["payload_sha256"]
    with (
        (model / "model.safetensors").open("rb") as stream,
        pytest.raises(ValueError, match="differs from its manifest"),
    ):
        list(_chunks(plan.tensors[0], {model / "model.safetensors": stream}, expected))


def test_gguf_reader_decodes_all_experts_and_preserves_router_precision(tmp_path: Path) -> None:
    gguf = pytest.importorskip("gguf")
    model, original = _fixture(tmp_path)
    plan = plan_kolibri_gguf(model)
    expected = json.loads((model / "mxwave-manifest.json").read_text())["payload_sha256"]
    output = tmp_path / "fixture.gguf"
    writer = gguf.GGUFWriter(None, "kolibri1")
    try:
        hashes = _write_tensors(plan, writer, output, gguf, expected)
    finally:
        writer.close()
    _verify_output(output, plan, hashes, gguf)
    tensors = {tensor.name: tensor for tensor in gguf.GGUFReader(output).tensors}
    expert = tensors["blk.0.ffn_gate_exps.weight"]
    restored = gguf.dequantize(expert.data, gguf.GGMLQuantizationType.MXFP4).reshape(12, 32, 160)
    for index in range(12):
        name = f"model.layers.0.mlp.experts.{index}.gate_proj"
        reference = dequant_mxfp4(
            original[name + ".weight_packed"], original[name + ".weight_scale"], (32, 160)
        )
        np.testing.assert_array_equal(restored[index], reference.float().numpy())
    router = tensors["blk.0.ffn_gate_inp.weight"]
    assert router.tensor_type == gguf.GGMLQuantizationType.F32
    np.testing.assert_array_equal(router.data, original["model.layers.0.mlp.gate.weight"].numpy())
    np.testing.assert_array_equal(
        tensors["blk.0.exp_probs_b.bias"].data, np.arange(12, dtype=np.float32)
    )
