"""Unit tests for the streaming engine end-to-end."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from mxstream.engine import QuantizeConfig, quantize_model, quantize_shard
from mxstream.shard import discover_shards


def _make_fake_model(tmp_path: Path) -> Path:
    """Write a tiny 2-shard model with a mix of weight + non-weight tensors."""
    # Shard 1: a quantizable weight (in_features divisible by 32) + an embed.
    w1 = torch.randn(64, 128)
    embed = torch.randn(16, 64)
    save_file(
        {
            "model.layers.0.mlp.gate_proj.weight": w1,
            "model.embed_tokens.weight": embed,
        },
        str(tmp_path / "model-00001-of-00002.safetensors"),
    )
    # Shard 2: another quantizable weight + a norm (non-weight, passthrough).
    w2 = torch.randn(32, 64)
    norm = torch.randn(64)
    save_file(
        {
            "model.layers.1.mlp.gate_proj.weight": w2,
            "model.layers.1.input_layernorm.weight": norm,
        },
        str(tmp_path / "model-00002-of-00002.safetensors"),
    )
    # Index + config
    index = {
        "metadata": {"total_size": 0},
        "weight_map": {
            "model.layers.0.mlp.gate_proj.weight": "model-00001-of-00002.safetensors",
            "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
            "model.layers.1.mlp.gate_proj.weight": "model-00002-of-00002.safetensors",
            "model.layers.1.input_layernorm.weight": "model-00002-of-00002.safetensors",
        },
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "test"}))
    return tmp_path


def test_discover_shards_uses_index(tmp_path: Path):
    model_dir = _make_fake_model(tmp_path)
    shards, weight_map = discover_shards(model_dir)
    assert len(shards) == 2
    assert weight_map is not None
    assert "model.embed_tokens.weight" in weight_map


def test_quantize_shard_emits_packed_and_passthrough(tmp_path: Path):
    model_dir = _make_fake_model(tmp_path)
    shards, _ = discover_shards(model_dir)
    cfg = QuantizeConfig(model_dir=model_dir, device="cpu")
    out = quantize_shard(shards[0], cfg)
    # Packed weight + scale emitted
    assert "model.layers.0.mlp.gate_proj.weight_packed" in out
    assert "model.layers.0.mlp.gate_proj.weight_scale" in out
    # Passthrough non-target tensors preserved verbatim
    assert "model.embed_tokens.weight" in out


def test_quantize_model_writes_output_shards(tmp_path: Path):
    model_dir = _make_fake_model(tmp_path)
    out_dir = tmp_path / "out"
    cfg = QuantizeConfig(model_dir=model_dir, output_dir=out_dir, device="cpu")
    n = quantize_model(cfg)
    assert n == 2
    assert (out_dir / "model-00001-of-00002.safetensors").exists()
    assert (out_dir / "model-00002-of-00002.safetensors").exists()


def test_quantize_model_with_hadamard_rotation(tmp_path: Path):
    model_dir = _make_fake_model(tmp_path)
    out_dir = tmp_path / "out_rot"
    cfg = QuantizeConfig(
        model_dir=model_dir,
        output_dir=out_dir,
        device="cpu",
        rotation="hadamard",
    )
    n = quantize_model(cfg)
    assert n == 2
    assert (out_dir / "model-00001-of-00002.safetensors").exists()


def test_should_quantize_excludes_norms_and_embeds():
    from mxstream.engine import _should_quantize

    assert _should_quantize("model.layers.0.mlp.gate_proj.weight", r".*self_attn.*")
    assert not _should_quantize("model.layers.0.input_layernorm.weight", r".*norm.*")
    assert not _should_quantize("model.embed_tokens.weight", r".*embed_tokens.*")
    assert not _should_quantize("model.lm_head.weight", r".*lm_head.*")
