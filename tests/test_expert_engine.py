"""Tests for bounded, adapter-driven routed-expert planning and emission."""

from __future__ import annotations

import json
import os
import re
import struct
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from mxwave import expert_cli, expert_engine
from mxwave.core import quantize_mxfp4
from mxwave.expert_cli import build_parser as build_expert_parser
from mxwave.expert_engine import (
    ExpertQuantizationConfig,
    emit_expert_unit,
    plan_expert_model,
    quantize_expert_model,
)
from mxwave.qualification import verify_checkpoint
from mxwave.shard import ShardFile, shard_tensor_info


def _make_moe_model(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    torch.manual_seed(7)
    shard_1 = "model-00001-of-00002.safetensors"
    shard_2 = "model-00002-of-00002.safetensors"
    tensors_1 = {
        "model.language_model.embed_tokens.weight": torch.randn(
            16, 64, dtype=torch.bfloat16
        ),
        "model.language_model.layers.0.mlp.experts.gate_up_proj": torch.randn(
            2, 64, 64, dtype=torch.bfloat16
        ),
        "model.language_model.layers.0.mlp.gate.weight": torch.randn(
            2, 64, dtype=torch.bfloat16
        ),
    }
    tensors_2 = {
        "model.language_model.layers.0.mlp.experts.down_proj": torch.randn(
            2, 64, 32, dtype=torch.bfloat16
        ),
        "model.language_model.norm.weight": torch.randn(64, dtype=torch.bfloat16),
        "model.visual.blocks.0.mlp.weight": torch.randn(64, 64, dtype=torch.bfloat16),
    }
    save_file(tensors_1, str(source / shard_1))
    save_file(tensors_2, str(source / shard_2))
    weight_map = {
        **{name: shard_1 for name in tensors_1},
        **{name: shard_2 for name in tensors_2},
    }
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )
    (source / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                "model_type": "qwen3_5_moe",
                "text_config": {
                    "model_type": "qwen3_5_moe_text",
                    "num_hidden_layers": 1,
                    "num_experts": 2,
                    "num_experts_per_tok": 1,
                    "hidden_size": 64,
                    "moe_intermediate_size": 32,
                    "mtp_num_hidden_layers": 1,
                },
                "vision_config": {"hidden_size": 64},
            }
        )
    )
    (source / "tokenizer.json").write_text('{"version":"test"}')
    (source / "preprocessor_config.json").write_text('{"image_processor_type":"test"}')
    return source


def _config(source: Path, output: Path | None = None, **kwargs: Any) -> ExpertQuantizationConfig:
    return ExpertQuantizationConfig(
        model_dir=source,
        output_dir=output or source.parent / "output",
        device="cpu",
        host_tensor_cap_bytes=16 * 1024,
        verbose=False,
        **kwargs,
    )


def _flip_tensor_payload_byte(shard_path: Path, tensor_name: str) -> None:
    info = shard_tensor_info(ShardFile(path=shard_path, weight_map={}))[tensor_name]
    with shard_path.open("r+b") as stream:
        raw_length = stream.read(8)
        header_length = struct.unpack("<Q", raw_length)[0]
        stream.seek(8 + header_length + info.data_offsets[0])
        original = stream.read(1)
        assert len(original) == 1
        stream.seek(-1, 1)
        stream.write(bytes([original[0] ^ 1]))


def test_expert_plan_has_exact_coverage_and_capped_units(tmp_path: Path) -> None:
    source = _make_moe_model(tmp_path)
    plan = plan_expert_model(_config(source))
    summary = plan.summary()

    assert plan.layout.source_bank_count == 2
    assert plan.layout.logical_matrix_count == 6
    assert plan.layout.output_tensor_count == 12
    assert len(plan.tensors) == 6
    assert plan.emitted_tensor_count == 16
    assert len(plan.units) == 3
    assert [unit.kind for unit in plan.units] == [
        "expert-bank",
        "expert-bank",
        "passthrough",
    ]
    assert plan.maximum_unit_bytes <= 16 * 1024
    assert len(plan.config_target_patterns) == 1
    target_pattern = re.compile(plan.config_target_patterns[0].removeprefix("re:"))
    assert target_pattern.match(
        "model.language_model.layers.0.mlp.experts.1.gate_proj"
    )
    assert target_pattern.match(
        "language_model.model.layers.0.mlp.experts.1.gate_proj"
    )
    assert target_pattern.match(
        "model.language_model.model.layers.0.mlp.experts.1.down_proj"
    )
    assert not target_pattern.match(
        "model.visual.layers.0.mlp.experts.1.gate_proj"
    )
    assert not target_pattern.match(
        "model.language_model.mtp.layers.0.mlp.experts.1.gate_proj"
    )
    assert r"re:.*mtp.*" in plan.config_ignored_patterns
    assert r"re:.*hyper.*" in plan.config_ignored_patterns
    assert len(plan.target_modules) == 6
    assert not any("shared_expert" in target for target in plan.target_modules)
    assert summary["policy_description"] == (
        "adapter-validated routed experts only; RTN baseline"
    )
    assert summary["method"] == "rtn"
    assert "scale_search" not in summary


def test_bank_task_opens_safetensors_once_and_never_reads_full_bank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _make_moe_model(tmp_path)
    config = _config(source, tensor_row_chunk_size=7)
    plan = plan_expert_model(config)
    unit = plan.units[0]
    actual_safe_open = expert_engine.safe_open
    open_count = 0

    def counting_safe_open(*args: Any, **kwargs: Any) -> Any:
        nonlocal open_count
        open_count += 1
        return actual_safe_open(*args, **kwargs)

    def reject_full_tensor(*args: Any, **kwargs: Any) -> torch.Tensor:
        raise AssertionError("Expert-bank emission must not call read_tensor")

    monkeypatch.setattr(expert_engine, "safe_open", counting_safe_open)
    monkeypatch.setattr(expert_engine, "read_tensor", reject_full_tensor)
    emitted = emit_expert_unit(
        unit,
        config,
        shards_by_name={shard.path.name: shard for shard in plan.shards},
    )

    assert open_count == 1
    assert len(emitted) == 8
    assert all(tensor.dtype == torch.uint8 for tensor in emitted.values())
    assert unit.sources[0].info.name not in emitted


@pytest.mark.parametrize("method", ["rtn", "mse"])
def test_expert_row_chunking_is_byte_identical(tmp_path: Path, method: str) -> None:
    source = _make_moe_model(tmp_path)
    whole_config = _config(source, method=method, tensor_row_chunk_size=128)
    chunked_config = _config(source, method=method, tensor_row_chunk_size=7)
    plan = plan_expert_model(whole_config)
    shards = {shard.path.name: shard for shard in plan.shards}
    whole = emit_expert_unit(plan.units[0], whole_config, shards_by_name=shards)
    chunked = emit_expert_unit(plan.units[0], chunked_config, shards_by_name=shards)

    assert whole.keys() == chunked.keys()
    assert all(torch.equal(whole[name], chunked[name]) for name in whole)


@pytest.mark.parametrize("method", ["rtn", "mse"])
def test_expert_emission_matches_core_on_exact_adapter_slices(
    tmp_path: Path,
    method: str,
) -> None:
    source = _make_moe_model(tmp_path)
    config = _config(source, method=method, tensor_row_chunk_size=7)
    plan = plan_expert_model(config)
    shards = {shard.path.name: shard for shard in plan.shards}

    for unit in (item for item in plan.units if item.kind == "expert-bank"):
        planned_source = unit.sources[0]
        with safe_open(
            str(shards[planned_source.shard_name].path),
            framework="pt",
            device="cpu",
        ) as handle:
            source_bank = handle.get_tensor(planned_source.info.name)
        emitted = emit_expert_unit(unit, config, shards_by_name=shards)
        for matrix in planned_source.logical_matrices:
            expected_packed, expected_scales = quantize_mxfp4(
                matrix.view(source_bank),
                method=method,
                scale_percentile=config.scale_percentile,
                mse_clip_depth=config.mse_clip_depth,
            )
            assert torch.equal(
                emitted[f"{matrix.output_module}.weight_packed"], expected_packed
            )
            assert torch.equal(
                emitted[f"{matrix.output_module}.weight_scale"], expected_scales
            )


def test_expert_cli_preserves_rtn_default_and_exposes_tested_mse_search() -> None:
    parser = build_expert_parser()
    default = parser.parse_args(["--model-dir", "source", "--output-dir", "output"])
    assert default.method == "rtn"
    assert default.scale_percentile == 99.5
    assert default.mse_clip_depth == 4

    mse = parser.parse_args(
        [
            "--model-dir",
            "source",
            "--output-dir",
            "output",
            "--method",
            "mse",
            "--scale-percentile",
            "98.25",
            "--mse-clip-depth",
            "3",
        ]
    )
    assert mse.method == "mse"
    assert mse.scale_percentile == 98.25
    assert mse.mse_clip_depth == 3

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--model-dir",
                "source",
                "--output-dir",
                "output",
                "--activation-stats",
                "removed.safetensors",
            ]
        )

def test_expert_cli_forwards_mse_settings(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: list[ExpertQuantizationConfig] = []

    def fake_quantize(config: ExpertQuantizationConfig) -> int:
        captured.append(config)
        return 2

    monkeypatch.setattr(expert_cli, "quantize_expert_model", fake_quantize)
    status = expert_cli.main(
        [
            "--model-dir",
            "source",
            "--output-dir",
            "output",
            "--method",
            "mse",
            "--scale-percentile",
            "98.25",
            "--mse-clip-depth",
            "3",
        ]
    )

    assert status == 0
    assert len(captured) == 1
    assert captured[0].method == "mse"
    assert captured[0].scale_percentile == 98.25
    assert captured[0].mse_clip_depth == 3
    assert "expert MSE shard(s)" in capsys.readouterr().out


@pytest.mark.parametrize("method", ["rtn", "mse"])
def test_expert_quantizer_forwards_selected_scale_search(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    source = _make_moe_model(tmp_path)
    config = _config(
        source,
        method=method,
        scale_percentile=98.25,
        mse_clip_depth=3,
        tensor_row_chunk_size=7,
    )
    plan = plan_expert_model(config)
    calls: list[dict[str, Any]] = []
    actual_quantize = expert_engine.quantize_mxfp4

    def recording_quantize(
        weight: torch.Tensor, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append(kwargs)
        return actual_quantize(weight, **kwargs)

    monkeypatch.setattr(expert_engine, "quantize_mxfp4", recording_quantize)
    emit_expert_unit(
        plan.units[0],
        config,
        shards_by_name={shard.path.name: shard for shard in plan.shards},
    )

    assert calls
    expected = {
        "method": method,
        "scale_percentile": 98.25,
        "mse_clip_depth": 3,
    }
    assert all(call == expected for call in calls)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"method": "unknown"}, "Unsupported quantization method"),
        ({"scale_percentile": 0.0}, "scale_percentile"),
        ({"scale_percentile": float("nan")}, "scale_percentile"),
        ({"mse_clip_depth": -1}, "mse_clip_depth"),
        ({"mse_clip_depth": 9}, "mse_clip_depth"),
        ({"mse_clip_depth": True}, "mse_clip_depth"),
    ],
)
def test_expert_scale_search_options_are_validated_before_planning(
    tmp_path: Path,
    options: dict[str, Any],
    message: str,
) -> None:
    source = _make_moe_model(tmp_path)
    with pytest.raises(ValueError, match=message):
        plan_expert_model(_config(source, **options))


def test_unweighted_mse_metadata_is_explicit_and_calibration_free(tmp_path: Path) -> None:
    source = _make_moe_model(tmp_path)
    output = tmp_path / "mse-output"
    config = _config(
        source,
        output,
        method="mse",
        scale_percentile=98.25,
        mse_clip_depth=3,
    )
    plan = plan_expert_model(config)
    summary = plan.summary()
    scale_search = {
        "scale_percentile": 98.25,
        "mse_clip_depth": 3,
        "includes_no_clipping_candidate": True,
    }
    assert summary["method"] == "unweighted-mse"
    assert summary["scale_search"] == scale_search
    assert "scale_percentile" not in summary
    assert "mse_clip_depth" not in summary
    assert summary["policy_description"] == (
        "adapter-validated routed experts only; unweighted-MSE scale-search candidate"
    )

    assert quantize_expert_model(config) == 3
    run = json.loads((output / "mxwave-run.json").read_text())
    assert run["engine"] == "adapter-driven-expert-quantization-v1"
    assert run["method"] == "unweighted-mse"
    assert run["scale_search"] == scale_search
    assert "scale_percentile" not in run
    assert "mse_clip_depth" not in run
    assert run["calibration"] == {"kind": "none"}

    manifest = json.loads((output / "mxwave-manifest.json").read_text())
    assert manifest["method"] == "unweighted-mse"
    assert manifest["scale_search"] == scale_search
    assert "scale_percentile" not in manifest
    assert "mse_clip_depth" not in manifest
    assert manifest["weight_scale_selection"] == (
        "unweighted-mse-p98.25-depth3-plus-no-clipping"
    )
    assert manifest["experimental_scope"] == (
        "routed-expert unweighted-MSE scale-search candidate; "
        "no activation calibration"
    )
    assert manifest["activation_calibration"] is None
    assert manifest["gamma_proxy"] is None
    assert "No activation calibration was used" in (output / "README.md").read_text()


def test_full_expert_emission_preserves_assets_and_passthrough(tmp_path: Path) -> None:
    source = _make_moe_model(tmp_path)
    output = tmp_path / "output"
    config = _config(
        source,
        output,
        source_repository="nex-agi/Nex-N2.5-mini",
        source_revision="test-revision",
    )
    plan = plan_expert_model(config)
    assert quantize_expert_model(config) == 3

    index = json.loads((output / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    assert len(weight_map) == plan.emitted_tensor_count
    assert "model.language_model.layers.0.mlp.experts.gate_up_proj" not in weight_map
    assert "model.language_model.layers.0.mlp.experts.down_proj" not in weight_map
    assert (
        "model.language_model.layers.0.mlp.experts.1.gate_proj.weight_packed"
        in weight_map
    )
    assert (
        "model.language_model.layers.0.mlp.experts.1.down_proj.weight_scale"
        in weight_map
    )

    for name in (
        "model.language_model.embed_tokens.weight",
        "model.language_model.layers.0.mlp.gate.weight",
        "model.language_model.norm.weight",
        "model.visual.blocks.0.mlp.weight",
    ):
        source_shard = source / json.loads(
            (source / "model.safetensors.index.json").read_text()
        )["weight_map"][name]
        output_shard = output / weight_map[name]
        with safe_open(str(source_shard), framework="pt", device="cpu") as source_handle:
            expected = source_handle.get_tensor(name)
        with safe_open(str(output_shard), framework="pt", device="cpu") as output_handle:
            actual = output_handle.get_tensor(name)
        assert torch.equal(actual, expected)

    emitted_config = json.loads((output / "config.json").read_text())
    group = emitted_config["quantization_config"]["config_groups"]["group_0"]
    assert group["format"] == "mxfp4-pack-quantized"
    assert group["targets"] == plan.config_target_patterns
    assert "input_activations" not in group
    assert r"re:.*mtp.*" in emitted_config["quantization_config"]["ignore"]
    assert emitted_config["vision_config"] == {"hidden_size": 64}
    assert (output / "tokenizer.json").read_text() == '{"version":"test"}'
    assert (output / "preprocessor_config.json").exists()

    manifest = json.loads((output / "mxwave-manifest.json").read_text())
    assert manifest["method"] == "rtn"
    assert "scale_search" not in manifest
    assert "scale_percentile" not in manifest
    assert "mse_clip_depth" not in manifest
    assert manifest["weight_scale_selection"] == "rtn-memoryless-minmax"
    assert manifest["activation_calibration"] is None
    assert manifest["experimental_scope"] == (
        "routed-expert RTN baseline; no activation calibration"
    )
    assert manifest["logical_matrices"] == 6
    assert manifest["quantized_output_tensors"] == 12
    assert manifest["host_memory_contract"]["resident_tensor_cap_bytes"] == 16 * 1024
    payload_verification = manifest["passthrough_payload_verification"]
    assert payload_verification["source_output_match"] is True
    assert payload_verification["tensor_count"] == 4
    assert len(payload_verification["aggregate_sha256"]) == 64
    assert set(payload_verification["per_tensor_sha256"]) == {
        "model.language_model.embed_tokens.weight",
        "model.language_model.layers.0.mlp.gate.weight",
        "model.language_model.norm.weight",
        "model.visual.blocks.0.mlp.weight",
    }
    integrity_path = output / "mxwave-shard-integrity.json"
    integrity = json.loads(integrity_path.read_text())
    assert integrity["run_identity_sha256"] == manifest["shard_integrity"][
        "run_identity_sha256"
    ]
    assert integrity["shards"] == manifest["shard_integrity"]["per_shard"]
    assert manifest["shard_integrity"]["complete"] is True
    assert manifest["shard_integrity"]["shard_count"] == len(plan.units)
    assert manifest["shard_integrity"]["file_bytes"] == sum(
        path.stat().st_size for path in output.glob("*.safetensors")
    )
    assert "mxwave-shard-integrity.json" in manifest[
        "fingerprint_excluded_metadata_assets"
    ]
    assert "mxwave-shard-integrity.json" not in manifest["copied_assets"]
    card = (output / "README.md").read_text()
    assert "weight-only W4A16" in card
    assert "--linear-backend marlin --moe-backend marlin" in card
    assert "No activation calibration was used" in card

    qualification = verify_checkpoint(output)
    assert qualification["status"] == "passed"
    assert qualification["target_modules"] == 6
    assert qualification["ignored_modules"] == 4

    run = json.loads((output / "mxwave-run.json").read_text())
    assert run["engine"] == "adapter-driven-expert-quantization-v1"
    assert run["layout_policy"] == "qwen3.5-moe-routed-experts"
    assert run["method"] == "rtn"
    assert "scale_percentile" not in run
    assert "mse_clip_depth" not in run
    assert run["calibration"] == {"kind": "none"}
    assert manifest["source"] == run["source"]
    source_shards = run["source"]["shards"]
    assert set(source_shards) == {
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    }
    assert all(set(record) == {"bytes", "sha256"} for record in source_shards.values())
    assert all(len(record["sha256"]) == 64 for record in source_shards.values())
    source_integrity = manifest["source_weight_shard_integrity"]
    assert source_integrity["algorithm"] == "sha256"
    assert source_integrity["per_shard"] == source_shards
    assert source_integrity["shard_count"] == 2
    assert source_integrity["file_bytes"] == sum(
        path.stat().st_size for path in source.glob("*.safetensors")
    )
    assert len(source_integrity["aggregate_sha256"]) == 64


def test_resume_validates_each_independent_output_shard(tmp_path: Path) -> None:
    source = _make_moe_model(tmp_path)
    output = tmp_path / "output"
    quantize_expert_model(_config(source, output))
    assert quantize_expert_model(_config(source, output, resume=True)) == 3

    first_shard = min(output.glob("*.safetensors"))
    first_shard.unlink()
    assert quantize_expert_model(_config(source, output, resume=True)) == 3
    assert first_shard.is_file()
    sidecar = json.loads((output / "mxwave-shard-integrity.json").read_text())
    assert sidecar["shards"][first_shard.name]["bytes"] == first_shard.stat().st_size


def test_resume_rejects_source_payload_change_with_restored_metadata(tmp_path: Path) -> None:
    source = _make_moe_model(tmp_path)
    output = tmp_path / "output"
    quantize_expert_model(_config(source, output))

    shard_path = source / "model-00001-of-00002.safetensors"
    before = shard_path.stat()
    _flip_tensor_payload_byte(
        shard_path,
        "model.language_model.layers.0.mlp.experts.gate_up_proj",
    )
    os.utime(shard_path, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = shard_path.stat()
    assert after.st_size == before.st_size
    assert after.st_mtime_ns == before.st_mtime_ns

    with pytest.raises(ValueError, match="source, layout, quantization"):
        quantize_expert_model(_config(source, output, resume=True))


def test_resume_allows_source_mtime_change_when_bytes_are_unchanged(tmp_path: Path) -> None:
    source = _make_moe_model(tmp_path)
    output = tmp_path / "output"
    quantize_expert_model(_config(source, output))

    shard_path = source / "model-00001-of-00002.safetensors"
    before = shard_path.stat()
    os.utime(
        shard_path,
        ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000),
    )

    assert quantize_expert_model(_config(source, output, resume=True)) == 3


def test_resume_rejects_changed_mse_scale_search(tmp_path: Path) -> None:
    source = _make_moe_model(tmp_path)
    output = tmp_path / "output"
    original = _config(
        source,
        output,
        method="mse",
        scale_percentile=99.5,
        mse_clip_depth=4,
    )
    quantize_expert_model(original)
    run = json.loads((output / "mxwave-run.json").read_text())
    assert run["engine"] == "adapter-driven-expert-quantization-v1"
    assert run["method"] == "unweighted-mse"
    assert run["scale_search"] == {
        "scale_percentile": 99.5,
        "mse_clip_depth": 4,
        "includes_no_clipping_candidate": True,
    }
    assert quantize_expert_model(
        _config(
            source,
            output,
            method="mse",
            scale_percentile=99.5,
            mse_clip_depth=4,
            resume=True,
        )
    ) == 3

    with pytest.raises(ValueError, match="quantization settings"):
        quantize_expert_model(
            _config(
                source,
                output,
                method="mse",
                scale_percentile=99.5,
                mse_clip_depth=3,
                resume=True,
            )
        )


def test_interrupted_emission_persists_each_complete_unit_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _make_moe_model(tmp_path)
    output = tmp_path / "output"
    actual_emit = expert_engine.emit_expert_unit
    calls = 0

    def interrupt_second_unit(*args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption before second output")
        return actual_emit(*args, **kwargs)

    monkeypatch.setattr(expert_engine, "emit_expert_unit", interrupt_second_unit)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        quantize_expert_model(_config(source, output))

    sidecar_path = output / "mxwave-shard-integrity.json"
    interrupted = json.loads(sidecar_path.read_text())
    assert list(interrupted["shards"]) == ["model-00001-of-00003.safetensors"]
    assert (output / "model-00001-of-00003.safetensors").is_file()
    assert not (output / "model-00002-of-00003.safetensors").exists()

    monkeypatch.setattr(expert_engine, "emit_expert_unit", actual_emit)
    assert quantize_expert_model(_config(source, output, resume=True)) == 3
    completed = json.loads(sidecar_path.read_text())
    assert set(completed["shards"]) == {
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
    }


def test_resume_rejects_header_valid_expert_payload_corruption(tmp_path: Path) -> None:
    source = _make_moe_model(tmp_path)
    output = tmp_path / "output"
    config = _config(source, output)
    plan = plan_expert_model(config)
    quantize_expert_model(config)

    expert_unit = next(unit for unit in plan.units if unit.kind == "expert-bank")
    packed_name = next(
        name for name in expert_unit.expected_specs() if name.endswith(".weight_packed")
    )
    shard_path = output / expert_unit.filename
    _flip_tensor_payload_byte(shard_path, packed_name)
    assert packed_name in shard_tensor_info(ShardFile(path=shard_path, weight_map={}))

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        quantize_expert_model(_config(source, output, resume=True))


@pytest.mark.parametrize("mutation", ["missing", "wrong-identity", "wrong-hash"])
def test_resume_rejects_missing_or_mismatched_integrity_sidecar(
    tmp_path: Path, mutation: str
) -> None:
    source = _make_moe_model(tmp_path)
    output = tmp_path / "output"
    quantize_expert_model(_config(source, output))
    sidecar_path = output / "mxwave-shard-integrity.json"

    if mutation == "missing":
        sidecar_path.unlink()
        match = "sidecar is missing"
    else:
        sidecar = json.loads(sidecar_path.read_text())
        if mutation == "wrong-identity":
            sidecar["run_identity_sha256"] = "0" * 64
            match = "does not match run identity"
        else:
            first_name = min(sidecar["shards"])
            sidecar["shards"][first_name]["sha256"] = "0" * 64
            match = "SHA-256 mismatch"
        sidecar_path.write_text(json.dumps(sidecar))

    with pytest.raises(ValueError, match=match):
        quantize_expert_model(_config(source, output, resume=True))


def test_resume_rejects_crash_window_shard_without_integrity_record(
    tmp_path: Path,
) -> None:
    source = _make_moe_model(tmp_path)
    output = tmp_path / "output"
    quantize_expert_model(_config(source, output))
    sidecar_path = output / "mxwave-shard-integrity.json"
    sidecar = json.loads(sidecar_path.read_text())
    first_name = min(sidecar["shards"])
    del sidecar["shards"][first_name]
    sidecar_path.write_text(json.dumps(sidecar))

    with pytest.raises(ValueError, match="untrusted crash window"):
        quantize_expert_model(_config(source, output, resume=True))


def test_resume_rejects_passthrough_payload_corruption(tmp_path: Path) -> None:
    source = _make_moe_model(tmp_path)
    output = tmp_path / "output"
    quantize_expert_model(_config(source, output))

    index = json.loads((output / "model.safetensors.index.json").read_text())
    tensor_name = "model.language_model.norm.weight"
    shard_path = output / index["weight_map"][tensor_name]
    _flip_tensor_payload_byte(shard_path, tensor_name)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        quantize_expert_model(_config(source, output, resume=True))


def test_resume_recomputes_complete_sqnr_coverage(tmp_path: Path) -> None:
    source = _make_moe_model(tmp_path)
    output = tmp_path / "output"
    quantize_expert_model(_config(source, output, verify_sqnr=True, sqnr_rows=3))
    quantize_expert_model(
        _config(source, output, resume=True, verify_sqnr=True, sqnr_rows=3)
    )

    manifest = json.loads((output / "mxwave-manifest.json").read_text())
    assert manifest["sqnr_db"]["count"] == 6
    assert manifest["sqnr_db"]["coverage"] == 1.0


def test_host_cap_rejects_an_oversized_bank_before_tensor_reads(tmp_path: Path) -> None:
    source = _make_moe_model(tmp_path)
    with pytest.raises(ValueError, match="Expert bank .* above host tensor cap"):
        plan_expert_model(
            ExpertQuantizationConfig(
                model_dir=source,
                output_dir=tmp_path / "output",
                device="cpu",
                host_tensor_cap_bytes=1024,
                verbose=False,
            )
        )
