"""Unit tests for the streaming engine and checkpoint contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

import mxwave.engine as engine_module
from mxwave.calibration import save_calibration_data
from mxwave.core import QuantizationMethod
from mxwave.core import quantize_mxfp4 as core_quantize_mxfp4
from mxwave.engine import QuantizeConfig, plan_model, quantize_model, quantize_shard
from mxwave.output import record_runtime_validation
from mxwave.shard import discover_shards


def _make_fake_model(tmp_path: Path, *, quantized_config: bool = False) -> Path:
    """Write a tiny two-shard dense model plus tokenizer assets."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    save_file(
        {
            "model.layers.0.mlp.gate_proj.weight": torch.randn(64, 128),
            "model.embed_tokens.weight": torch.randn(16, 64),
        },
        str(model_dir / "model-00001-of-00002.safetensors"),
    )
    save_file(
        {
            "model.layers.1.mlp.gate_proj.weight": torch.randn(32, 64),
            "model.layers.1.input_layernorm.weight": torch.randn(64),
        },
        str(model_dir / "model-00002-of-00002.safetensors"),
    )
    index = {
        "metadata": {"total_size": 0},
        "weight_map": {
            "model.layers.0.mlp.gate_proj.weight": "model-00001-of-00002.safetensors",
            "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
            "model.layers.1.mlp.gate_proj.weight": "model-00002-of-00002.safetensors",
            "model.layers.1.input_layernorm.weight": "model-00002-of-00002.safetensors",
        },
    }
    (model_dir / "model.safetensors.index.json").write_text(json.dumps(index))
    config: dict[str, object] = {"model_type": "test"}
    if quantized_config:
        config["quantization_config"] = {"quant_method": "gptq", "bits": 4}
    (model_dir / "config.json").write_text(json.dumps(config))
    (model_dir / "tokenizer.json").write_text('{"version":"1.0"}')
    (model_dir / "chat_template.jinja").write_text("{{ messages }}")
    return model_dir


def _config(model_dir: Path, output_dir: Path | None = None, **kwargs: object) -> QuantizeConfig:
    return QuantizeConfig(
        model_dir=model_dir,
        output_dir=output_dir or model_dir.parent / "out",
        device="cpu",
        policy="all-linear",
        verbose=False,
        **kwargs,
    )


def _make_qwen_model(tmp_path: Path) -> Path:
    model_dir = tmp_path / "qwen"
    model_dir.mkdir()
    tensors: dict[str, torch.Tensor] = {}
    weight_map: dict[str, str] = {}
    shard_name = "model.safetensors"
    for layer in range(64):
        input_norm_name = f"model.language_model.layers.{layer}.input_layernorm.weight"
        tensors[input_norm_name] = torch.ones(32, dtype=torch.bfloat16)
        weight_map[input_norm_name] = shard_name
        norm_name = f"model.language_model.layers.{layer}.post_attention_layernorm.weight"
        tensors[norm_name] = torch.ones(32, dtype=torch.bfloat16)
        weight_map[norm_name] = shard_name
        for projection in ("gate_proj", "up_proj", "down_proj"):
            name = f"model.language_model.layers.{layer}.mlp.{projection}.weight"
            tensors[name] = torch.zeros(1, 32, dtype=torch.bfloat16)
            weight_map[name] = shard_name
        attention_kind = "self_attn" if layer % 4 == 3 else "linear_attn"
        projections = (
            ("q_proj", "k_proj", "v_proj", "o_proj")
            if attention_kind == "self_attn"
            else ("in_proj_qkv", "in_proj_z", "out_proj")
        )
        for projection in projections:
            name = f"model.language_model.layers.{layer}.{attention_kind}.{projection}.weight"
            tensors[name] = torch.zeros(1, 32, dtype=torch.bfloat16)
            weight_map[name] = shard_name
    save_file(tensors, str(model_dir / shard_name))
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "model_type": "qwen3_5",
                "text_config": {
                    "num_hidden_layers": 64,
                    "hidden_size": 5120,
                    "intermediate_size": 17408,
                    "layer_types": [
                        "full_attention" if layer % 4 == 3 else "linear_attention"
                        for layer in range(64)
                    ],
                },
            }
        )
    )
    return model_dir


def test_discover_shards_uses_index(tmp_path: Path):
    model_dir = _make_fake_model(tmp_path)
    shards, weight_map = discover_shards(model_dir)
    assert len(shards) == 2
    assert weight_map is not None
    assert "model.embed_tokens.weight" in weight_map


def test_plan_is_header_only_and_reports_projection(tmp_path: Path):
    model_dir = _make_fake_model(tmp_path)
    plan = plan_model(_config(model_dir))
    summary = plan.summary()
    assert plan.target_names == frozenset(
        {
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.1.mlp.gate_proj.weight",
        }
    )
    assert summary["target_tensors"] == 2
    assert summary["projected_output_data_bytes"] < summary["source_data_bytes"]


def test_auto_policy_refuses_unknown_architecture(tmp_path: Path):
    model_dir = _make_fake_model(tmp_path)
    with pytest.raises(ValueError, match="No safe automatic policy"):
        plan_model(QuantizeConfig(model_dir=model_dir, device="cpu"))


def test_auto_policy_selects_only_qwen_mlp_phase_one(tmp_path: Path):
    model_dir = _make_qwen_model(tmp_path)
    plan = plan_model(QuantizeConfig(model_dir=model_dir, device="cpu"))
    assert plan.policy.name == "qwen3.8-27b-mlp"
    assert len(plan.target_names) == 192
    assert len(plan.gamma_proxy_sources) == 128
    assert plan.summary()["gamma_proxy_targets"] == 128
    assert (
        "model.language_model.layers.0.mlp.gate_proj.weight",
        "model.language_model.layers.0.post_attention_layernorm.weight",
    ) in plan.gamma_proxy_sources
    assert not any("down_proj" in target for target, _source in plan.gamma_proxy_sources)
    assert "model.language_model.layers.3.self_attn.q_proj.weight" not in plan.target_names


def test_gamma_proxy_can_be_disabled_and_rtn_never_uses_it(tmp_path: Path):
    model_dir = _make_qwen_model(tmp_path)
    disabled = plan_model(QuantizeConfig(model_dir=model_dir, device="cpu", gamma_proxy=False))
    rtn = plan_model(QuantizeConfig(model_dir=model_dir, device="cpu", method="rtn"))
    assert disabled.gamma_proxy_sources == ()
    assert rtn.gamma_proxy_sources == ()


def test_compatible_policy_weights_direct_attention_inputs_with_input_norm(tmp_path: Path):
    model_dir = _make_qwen_model(tmp_path)
    plan = plan_model(
        QuantizeConfig(
            model_dir=model_dir,
            device="cpu",
            policy="qwen3.8-27b-compatible",
        )
    )
    assert len(plan.target_names) == 400
    assert len(plan.gamma_proxy_sources) == 272
    assert (
        "model.language_model.layers.3.self_attn.q_proj.weight",
        "model.language_model.layers.3.input_layernorm.weight",
    ) in plan.gamma_proxy_sources
    assert (
        "model.language_model.layers.0.linear_attn.in_proj_z.weight",
        "model.language_model.layers.0.input_layernorm.weight",
    ) in plan.gamma_proxy_sources
    assert not any("o_proj" in target for target, _ in plan.gamma_proxy_sources)
    assert not any("out_proj" in target for target, _ in plan.gamma_proxy_sources)


def test_qwen_quantization_passes_module_keyed_gamma_and_records_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    model_dir = _make_qwen_model(tmp_path)
    output_dir = tmp_path / "out"
    gamma_calls = 0
    unweighted_calls = 0

    def recording_quantize(
        tensor: torch.Tensor,
        scale_percentile: float = 99.5,
        gamma: torch.Tensor | None = None,
        hessian: torch.Tensor | None = None,
        method: QuantizationMethod = "mse",
        mse_clip_depth: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal gamma_calls, unweighted_calls
        if gamma is None:
            unweighted_calls += 1
        else:
            gamma_calls += 1
            assert gamma.shape == (tensor.shape[-1],)
        return core_quantize_mxfp4(
            tensor,
            scale_percentile=scale_percentile,
            gamma=gamma,
            hessian=hessian,
            method=method,
            mse_clip_depth=mse_clip_depth,
        )

    monkeypatch.setattr(engine_module, "quantize_mxfp4", recording_quantize)
    quantize_model(
        QuantizeConfig(
            model_dir=model_dir,
            output_dir=output_dir,
            device="cpu",
            verbose=False,
            verify_sqnr=True,
        )
    )

    assert gamma_calls == 128
    assert unweighted_calls == 64
    manifest = json.loads((output_dir / "mxwave-manifest.json").read_text())
    assert manifest["weight_scale_selection"] == "mse-layernorm-gamma-proxy"
    assert manifest["gamma_proxy"] == {
        "sources": ["post_attention_layernorm.weight"],
        "weighted_tensors": 128,
        "unweighted_tensors": 64,
    }
    assert manifest["gamma_weighted_sqnr_db"]["count"] == 128
    assert manifest["gamma_weighted_sqnr_db"]["coverage"] == 1.0
    assert "Gamma-weighted SQNR" in (output_dir / "README.md").read_text()


def test_activation_statistics_replace_proxy_and_cover_every_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir = _make_fake_model(tmp_path)
    output_dir = tmp_path / "out"
    stats_path = tmp_path / "activation-stats.safetensors"
    statistics = {
        "model.layers.0.mlp.gate_proj.weight": torch.linspace(0.1, 1.0, 128),
        "model.layers.1.mlp.gate_proj.weight": torch.linspace(0.2, 1.0, 64),
    }
    save_calibration_data(
        stats_path,
        {"mean-abs": statistics},
        {
            "policy": "all-linear",
            "num_sequences": "8",
            "sequence_length": "32",
            "num_tokens": "256",
            "corpus_sha256": "corpus",
            "token_ids_sha256": "tokens",
            "hessian_damp": "0.0",
        },
    )
    observed: dict[str, torch.Tensor] = {}

    def recording_quantize(
        tensor: torch.Tensor,
        scale_percentile: float = 99.5,
        gamma: torch.Tensor | None = None,
        hessian: torch.Tensor | None = None,
        method: QuantizationMethod = "mse",
        mse_clip_depth: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert gamma is not None
        observed[str(tensor.shape[-1])] = gamma
        return core_quantize_mxfp4(
            tensor,
            scale_percentile=scale_percentile,
            gamma=gamma,
            hessian=hessian,
            method=method,
            mse_clip_depth=mse_clip_depth,
        )

    monkeypatch.setattr(engine_module, "quantize_mxfp4", recording_quantize)
    quantize_model(
        _config(
            model_dir,
            output_dir,
            activation_stats=stats_path,
            calibration_objective="mean-abs",
            verify_sqnr=True,
        )
    )

    assert torch.equal(observed["128"], statistics["model.layers.0.mlp.gate_proj.weight"])
    assert torch.equal(observed["64"], statistics["model.layers.1.mlp.gate_proj.weight"])
    manifest = json.loads((output_dir / "mxwave-manifest.json").read_text())
    assert manifest["weight_scale_selection"] == "mse-activation-mean-abs"
    assert manifest["gamma_proxy"] is None
    assert manifest["activation_calibration"]["weighted_tensors"] == 2
    assert manifest["activation_calibration"]["unweighted_tensors"] == 0
    assert manifest["calibration_weighted_sqnr_db"]["count"] == 2
    card = (output_dir / "README.md").read_text()
    assert "corpus-derived `mean-abs` inputs on 2 targets" in card
    assert "Calibration-weighted SQNR" in card


def test_block_hessian_scale_selection_is_wired_and_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir = _make_fake_model(tmp_path)
    output_dir = tmp_path / "out"
    stats_path = tmp_path / "activation-stats.safetensors"
    statistics = {
        "model.layers.0.mlp.gate_proj.weight": torch.eye(32).repeat(4, 1, 1),
        "model.layers.1.mlp.gate_proj.weight": torch.eye(32).repeat(2, 1, 1),
    }
    save_calibration_data(
        stats_path,
        {"block-hessian": statistics},
        {
            "policy": "all-linear",
            "num_sequences": "8",
            "sequence_length": "32",
            "num_tokens": "256",
            "corpus_sha256": "corpus",
            "token_ids_sha256": "tokens",
            "hessian_damp": "1e-6",
        },
    )
    observed_hessians: list[torch.Tensor] = []

    def recording_quantize(
        tensor: torch.Tensor,
        scale_percentile: float = 99.5,
        gamma: torch.Tensor | None = None,
        hessian: torch.Tensor | None = None,
        method: QuantizationMethod = "mse",
        mse_clip_depth: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert hessian is not None
        observed_hessians.append(hessian)
        return core_quantize_mxfp4(
            tensor,
            scale_percentile=scale_percentile,
            gamma=gamma,
            hessian=hessian,
            method=method,
            mse_clip_depth=mse_clip_depth,
        )

    monkeypatch.setattr(engine_module, "quantize_mxfp4", recording_quantize)
    quantize_model(
        _config(
            model_dir,
            output_dir,
            activation_stats=stats_path,
            calibration_objective="block-hessian",
            verify_sqnr=True,
        )
    )

    assert len(observed_hessians) == 2
    assert {tuple(value.shape) for value in observed_hessians} == {
        (4, 32, 32),
        (2, 32, 32),
    }
    manifest = json.loads((output_dir / "mxwave-manifest.json").read_text())
    assert manifest["weight_scale_selection"] == "mse-activation-block-hessian"
    assert manifest["activation_calibration"]["objective"] == "block-hessian"
    assert manifest["calibration_weighted_sqnr_db"]["count"] == 2


def test_prequantized_source_is_rejected(tmp_path: Path):
    model_dir = _make_fake_model(tmp_path, quantized_config=True)
    with pytest.raises(ValueError, match="unknown_quantized"):
        plan_model(_config(model_dir))


def test_quantize_shard_emits_packed_and_passthrough(tmp_path: Path):
    model_dir = _make_fake_model(tmp_path)
    shards, _ = discover_shards(model_dir)
    config = _config(model_dir)
    output = quantize_shard(shards[0], config)
    assert "model.layers.0.mlp.gate_proj.weight_packed" in output
    assert "model.layers.0.mlp.gate_proj.weight_scale" in output
    assert "model.layers.0.mlp.gate_proj.weight" not in output
    assert "model.embed_tokens.weight" in output


def test_tensor_row_chunking_is_byte_identical(tmp_path: Path) -> None:
    model_dir = _make_fake_model(tmp_path)
    shards, _ = discover_shards(model_dir)
    whole = quantize_shard(shards[0], _config(model_dir, tensor_row_chunk_size=1024))
    chunked = quantize_shard(shards[0], _config(model_dir, tensor_row_chunk_size=7))

    assert whole.keys() == chunked.keys()
    assert all(torch.equal(whole[name], chunked[name]) for name in whole)


def test_tensor_row_chunk_size_must_be_positive(tmp_path: Path) -> None:
    model_dir = _make_fake_model(tmp_path)
    with pytest.raises(ValueError, match="tensor_row_chunk_size must be a positive integer"):
        plan_model(_config(model_dir, tensor_row_chunk_size=0))


def test_quantize_model_writes_output_shards_and_assets(tmp_path: Path):
    model_dir = _make_fake_model(tmp_path)
    for generated_name in (
        "mxwave-manifest.json",
        "mxwave-run.json",
        "mxstream-manifest.json",
        "mxstream-run.json",
    ):
        (model_dir / generated_name).write_text("{}")
    output_dir = tmp_path / "out"
    assert quantize_model(_config(model_dir, output_dir)) == 2
    assert (output_dir / "model-00001-of-00002.safetensors").exists()
    assert (output_dir / "model-00002-of-00002.safetensors").exists()
    assert (output_dir / "tokenizer.json").read_text() == '{"version":"1.0"}'
    assert (output_dir / "chat_template.jinja").exists()
    assert (output_dir / "mxwave-manifest.json").exists()
    card = (output_dir / "README.md").read_text()
    assert "compressed-tensors" in card
    assert "--linear-backend marlin" in card
    assert "W4A16" in card
    assert "Not measured yet" in card
    assert not (output_dir / "mxstream-manifest.json").exists()
    assert not (output_dir / "mxstream-run.json").exists()
    assert "mxwave-manifest.json" not in json.loads(
        (output_dir / "mxwave-manifest.json").read_text()
    )["copied_assets"]
    assert "mxwave-run.json" not in json.loads(
        (output_dir / "mxwave-manifest.json").read_text()
    )["copied_assets"]


def test_runtime_validation_regenerates_evidence_in_model_card(tmp_path: Path):
    model_dir = _make_fake_model(tmp_path)
    output_dir = tmp_path / "out"
    quantize_model(_config(model_dir, output_dir))

    manifest = record_runtime_validation(
        output_dir,
        {
            "status": "passed",
            "runtime": "vLLM test",
            "hardware": "SM121",
            "linear_backend": "MarlinMxFp4LinearKernel",
            "execution_mode": "W4A16",
            "smoke": {"passed": 8, "total": 8},
            "evaluation": {
                "suite": "smoke",
                "suite_version": 2,
                "enable_thinking": False,
                "passed": 4,
                "total": 4,
                "run_id": "run-test",
            },
        },
    )

    assert manifest["runtime_validation"]["status"] == "passed"
    card = (output_dir / "README.md").read_text()
    assert "vLLM test, SM121, MarlinMxFp4LinearKernel, W4A16" in card
    assert "8/8 passed" in card
    assert "4/4 passed — smoke v2, thinking disabled, run `run-test`" in card


def test_runtime_validation_rejects_unknown_status(tmp_path: Path):
    model_dir = _make_fake_model(tmp_path)
    output_dir = tmp_path / "out"
    quantize_model(_config(model_dir, output_dir))
    with pytest.raises(ValueError, match="passed, failed, or partial"):
        record_runtime_validation(output_dir, {"status": "maybe"})


def test_rotation_is_rejected_until_fully_folded(tmp_path: Path):
    model_dir = _make_fake_model(tmp_path)
    with pytest.raises(ValueError, match="Rotation is disabled"):
        quantize_model(_config(model_dir, rotation="hadamard"))


def test_global_calibration_is_rejected(tmp_path: Path):
    model_dir = _make_fake_model(tmp_path)
    with pytest.raises(ValueError, match="keyed per target module"):
        plan_model(_config(model_dir, gamma=torch.ones(128)))


def test_quantize_model_emits_loader_safe_config(tmp_path: Path):
    model_dir = _make_fake_model(tmp_path)
    output_dir = tmp_path / "out"
    quantize_model(_config(model_dir, output_dir))

    config = json.loads((output_dir / "config.json").read_text())
    assert config["model_type"] == "test"
    quantization = config["quantization_config"]
    assert quantization["format"] == "mxfp4-pack-quantized"
    assert quantization["quant_method"] == "compressed-tensors"
    assert "version" not in quantization
    assert "transform_config" not in quantization
    targets = quantization["config_groups"]["group_0"]["targets"]
    assert targets == [
        "model.layers.0.mlp.gate_proj",
        "model.layers.1.mlp.gate_proj",
    ]
    assert "model.embed_tokens" in quantization["ignore"]
    assert "model.layers.1.input_layernorm" in quantization["ignore"]
    assert quantization["global_compression_ratio"] > 1.0


def test_index_uses_emitted_tensor_names_and_shapes(tmp_path: Path):
    model_dir = _make_fake_model(tmp_path)
    output_dir = tmp_path / "out"
    quantize_model(_config(model_dir, output_dir))

    index = json.loads((output_dir / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    assert "model.layers.0.mlp.gate_proj.weight" not in weight_map
    assert "model.layers.0.mlp.gate_proj.weight_packed" in weight_map
    assert "model.layers.0.mlp.gate_proj.weight_scale" in weight_map
    assert index["metadata"]["total_size"] > 0

    shard_path = output_dir / weight_map["model.layers.0.mlp.gate_proj.weight_packed"]
    with safe_open(str(shard_path), framework="pt", device="cpu") as shard:
        assert shard.get_tensor("model.layers.0.mlp.gate_proj.weight_packed").shape == (
            64,
            64,
        )
        assert shard.get_tensor("model.layers.0.mlp.gate_proj.weight_scale").shape == (
            64,
            4,
        )


def test_nonempty_output_requires_resume_and_validates_shards(tmp_path: Path):
    model_dir = _make_fake_model(tmp_path)
    output_dir = tmp_path / "out"
    quantize_model(_config(model_dir, output_dir))
    with pytest.raises(FileExistsError, match="not empty"):
        quantize_model(_config(model_dir, output_dir))
    assert quantize_model(_config(model_dir, output_dir, resume=True)) == 2


def test_resume_rejects_changed_scale_search(tmp_path: Path) -> None:
    model_dir = _make_fake_model(tmp_path)
    output_dir = tmp_path / "out"
    quantize_model(_config(model_dir, output_dir, mse_clip_depth=1))
    with pytest.raises(ValueError, match="Cannot safely resume"):
        quantize_model(_config(model_dir, output_dir, mse_clip_depth=4, resume=True))


def test_sqnr_samples_are_recorded(tmp_path: Path):
    model_dir = _make_fake_model(tmp_path)
    output_dir = tmp_path / "out"
    quantize_model(_config(model_dir, output_dir, verify_sqnr=True, sqnr_rows=4))
    manifest = json.loads((output_dir / "mxwave-manifest.json").read_text())
    assert manifest["sqnr_db"]["minimum"] > 5.0
    assert manifest["sqnr_db"]["count"] == 2
    assert manifest["sqnr_db"]["coverage"] == 1.0
    assert len(manifest["sqnr_db"]["per_tensor"]) == 2

    quantize_model(
        _config(
            model_dir,
            output_dir,
            verify_sqnr=True,
            sqnr_rows=4,
            resume=True,
        )
    )
    resumed = json.loads((output_dir / "mxwave-manifest.json").read_text())
    assert resumed["sqnr_db"]["count"] == 2
    assert resumed["sqnr_db"]["coverage"] == 1.0
