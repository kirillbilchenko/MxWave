"""Qualification contract and whole-checkpoint verification tests."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from mxwave.qualification import (
    RUNTIME_EVIDENCE_FORMAT,
    SPEC_FORMAT,
    load_spec,
    run_qualification,
    verify_checkpoint,
)
from mxwave.qualification_cli import main


def _make_checkpoint(
    tmp_path: Path,
    *,
    bad_scale_shape: bool = False,
    extra_uncovered: bool = False,
) -> Path:
    root = tmp_path / "model"
    root.mkdir()
    module = "model.layers.0.mlp.gate_proj"
    ignored = "model.layers.0.mlp.up_proj"
    tensors = {
        f"{module}.weight_packed": torch.zeros((2, 32), dtype=torch.uint8),
        f"{module}.weight_scale": torch.full(
            (2, 3 if bad_scale_shape else 2), 127, dtype=torch.uint8
        ),
        f"{ignored}.weight": torch.zeros((2, 64), dtype=torch.bfloat16),
    }
    if extra_uncovered:
        tensors["model.layers.0.mlp.down_proj.weight"] = torch.zeros(
            (2, 64), dtype=torch.bfloat16
        )
    shard = root / "model.safetensors"
    save_file(tensors, shard)
    tensor_bytes = sum(value.numel() * value.element_size() for value in tensors.values())
    weight_map = {name: shard.name for name in tensors}
    index = {"metadata": {"total_size": tensor_bytes}, "weight_map": weight_map}
    config = {
        "model_type": "synthetic",
        "quantization_config": {
            "format": "mxfp4-pack-quantized",
            "quant_method": "compressed-tensors",
            "quantization_status": "compressed",
            "config_groups": {
                "group_0": {
                    "targets": [module],
                    "weights": {
                        "num_bits": 4,
                        "type": "float",
                        "symmetric": True,
                        "group_size": 32,
                        "strategy": "group",
                        "dynamic": False,
                        "scale_dtype": "torch.uint8",
                    },
                    "input_activations": {
                        "num_bits": 4,
                        "type": "float",
                        "symmetric": True,
                        "group_size": 32,
                        "strategy": "group",
                        "dynamic": True,
                        "scale_dtype": "torch.uint8",
                    },
                    "output_activations": None,
                }
            },
            "ignore": [ignored],
        },
    }
    manifest = {
        "manifest_version": 1,
        "target_modules": [module],
        "ignored_modules": [ignored],
        "actual_output_bytes": shard.stat().st_size,
        "copied_assets": [],
    }
    (root / "config.json").write_text(json.dumps(config))
    (root / "model.safetensors.index.json").write_text(json.dumps(index))
    (root / "mxwave-manifest.json").write_text(json.dumps(manifest))
    return root


def _make_regex_weight_only_checkpoint(
    tmp_path: Path,
    *,
    explicit_null_activations: bool = False,
) -> Path:
    root = tmp_path / "regex-weight-only-model"
    root.mkdir()
    target_pattern = (
        r"re:^model\.layers\.0\.mlp\.experts\.\d+\.(?:gate_proj|down_proj)$"
    )
    targets = sorted(
        f"model.layers.0.mlp.experts.{expert}.{projection}"
        for expert in range(2)
        for projection in ("gate_proj", "down_proj")
    )
    ignored = sorted(
        (
            "model.layers.0.mlp.router",
            "model.visual.blocks.0.mlp",
        )
    )
    tensors: dict[str, torch.Tensor] = {}
    for module in targets:
        tensors[f"{module}.weight_packed"] = torch.zeros((2, 32), dtype=torch.uint8)
        tensors[f"{module}.weight_scale"] = torch.full((2, 2), 127, dtype=torch.uint8)
    for module in ignored:
        tensors[f"{module}.weight"] = torch.zeros((2, 64), dtype=torch.bfloat16)
    shard = root / "model.safetensors"
    save_file(tensors, shard)
    tensor_bytes = sum(value.numel() * value.element_size() for value in tensors.values())
    ignore_patterns = sorted((*ignored, r"re:.*mtp.*", r"re:.*hyper.*"))
    group = {
        "targets": [target_pattern],
        "weights": {
            "num_bits": 4,
            "type": "float",
            "symmetric": True,
            "group_size": 32,
            "strategy": "group",
            "dynamic": False,
            "scale_dtype": "torch.uint8",
        },
    }
    if explicit_null_activations:
        group["input_activations"] = None
    config = {
        "model_type": "synthetic_moe",
        "quantization_config": {
            "format": "mxfp4-pack-quantized",
            "quant_method": "compressed-tensors",
            "quantization_status": "compressed",
            "config_groups": {"group_0": group},
            "ignore": ignore_patterns,
        },
    }
    index = {
        "metadata": {"total_size": tensor_bytes},
        "weight_map": {name: shard.name for name in tensors},
    }
    manifest = {
        "manifest_version": 1,
        "target_modules": targets,
        "ignored_modules": ignored,
        "config_target_patterns": [target_pattern],
        "config_ignored_patterns": ignore_patterns,
        "actual_output_bytes": shard.stat().st_size,
        "copied_assets": [],
    }
    (root / "config.json").write_text(json.dumps(config))
    (root / "model.safetensors.index.json").write_text(json.dumps(index))
    (root / "mxwave-manifest.json").write_text(json.dumps(manifest))
    return root


_PPL_PROTOCOL = "1" * 64
_RUNTIME_PROTOCOL = "2" * 64


def _runtime_evidence(
    path: Path,
    checkpoint_sha256: str,
    *,
    include_concurrency: bool = True,
) -> None:
    samples = [
        {
            "concurrency": 1,
            "ttft_seconds": 0.5,
            "output_tokens_per_second": 12.0,
        }
    ]
    if include_concurrency:
        samples.append(
            {
                "concurrency": 4,
                "ttft_seconds": 0.8,
                "output_tokens_per_second": 30.0,
            }
        )
    path.write_text(
        json.dumps(
            {
                "format": RUNTIME_EVIDENCE_FORMAT,
                "checkpoint_sha256": checkpoint_sha256,
                "protocol_sha256": _RUNTIME_PROTOCOL,
                "runtime": {
                    "name": "vllm",
                    "version": "0.29.0",
                    "image": f"vllm/vllm-openai:v0.29.0@sha256:{'a' * 64}",
                    "stock": True,
                    "client_location": "on-device",
                },
                "load": {
                    "status": "passed",
                    "kernel_backend": "marlin",
                    "fresh_runtime_cache_startup_seconds": 12.0,
                    "warm_restart_seconds": 8.0,
                },
                "memory": {
                    "peak_host_used_bytes": 1000,
                    "peak_accelerator_bytes": 900,
                },
                "performance": {"samples": samples},
                "mtp": {
                    "enabled": True,
                    "proposed_tokens": 100,
                    "accepted_tokens": 60,
                    "token_parity": True,
                },
                "smoke": {"pass": True},
            }
        )
    )


def _ppl_evidence(path: Path, checkpoint_sha256: str, *, perplexity: float = 8.1) -> None:
    mean_nll = math.log(perplexity)
    total_nll = mean_nll * 1000
    chunks = [
        {
            "index": 0,
            "text_sha256": "3" * 64,
            "token_sha256": "4" * 64,
            "scored_tokens": 500,
            "negative_log_likelihood": total_nll / 2,
        },
        {
            "index": 1,
            "text_sha256": "5" * 64,
            "token_sha256": "6" * 64,
            "scored_tokens": 500,
            "negative_log_likelihood": total_nll / 2,
        },
    ]
    path.write_text(
        json.dumps(
            {
                "format": "mxwave-api-perplexity-v1",
                "checkpoint_sha256": checkpoint_sha256,
                "protocol_sha256": _PPL_PROTOCOL,
                "dataset_revision": "frozen-revision",
                "dataset_sha256": "7" * 64,
                "corpus_sha256": "8" * 64,
                "num_chunks": len(chunks),
                "scored_tokens": 1000,
                "negative_log_likelihood": total_nll,
                "mean_negative_log_likelihood": mean_nll,
                "perplexity": perplexity,
                "chunks": chunks,
            }
        )
    )


def _make_mixed_checkpoint(tmp_path: Path) -> Path:
    root = tmp_path / "mixed-model"
    root.mkdir()
    mxfp4_module = "model.layers.0.mlp.gate_proj"
    fp8_module = "model.layers.0.mlp.up_proj"
    tensors = {
        f"{mxfp4_module}.weight_packed": torch.zeros((2, 32), dtype=torch.uint8),
        f"{mxfp4_module}.weight_scale": torch.full((2, 2), 127, dtype=torch.uint8),
        f"{fp8_module}.weight": torch.zeros((2, 64), dtype=torch.float8_e4m3fn),
        f"{fp8_module}.weight_scale": torch.ones((2, 1), dtype=torch.float32),
    }
    shard = root / "model.safetensors"
    save_file(tensors, shard)
    tensor_bytes = sum(value.numel() * value.element_size() for value in tensors.values())
    config = {
        "quantization_config": {
            "format": "mixed-precision",
            "quant_method": "compressed-tensors",
            "quantization_status": "compressed",
            "config_groups": {
                "group_0": {
                    "format": "mxfp4-pack-quantized",
                    "targets": [mxfp4_module],
                    "weights": {
                        "num_bits": 4,
                        "type": "float",
                        "symmetric": True,
                        "group_size": 32,
                        "strategy": "group",
                        "dynamic": False,
                        "scale_dtype": "torch.uint8",
                    },
                    "input_activations": {
                        "num_bits": 4,
                        "type": "float",
                        "symmetric": True,
                        "group_size": 32,
                        "strategy": "group",
                        "dynamic": True,
                        "scale_dtype": "torch.uint8",
                    },
                    "output_activations": None,
                },
                "group_1": {
                    "format": "float-quantized",
                    "targets": [fp8_module],
                    "weights": {
                        "num_bits": 8,
                        "type": "float",
                        "symmetric": True,
                        "group_size": None,
                        "strategy": "channel",
                        "dynamic": False,
                    },
                    "input_activations": None,
                    "output_activations": None,
                },
            },
            "ignore": [],
        }
    }
    index = {
        "metadata": {"total_size": tensor_bytes},
        "weight_map": {name: shard.name for name in tensors},
    }
    manifest = {
        "target_modules": [mxfp4_module, fp8_module],
        "ignored_modules": [],
        "actual_output_bytes": shard.stat().st_size,
        "copied_assets": [],
        "mxfp4_target_modules": [mxfp4_module],
        "fp8_target_modules": [fp8_module],
        "mxfp4_target_tensors": 1,
        "fp8_target_tensors": 1,
        "target_tensors": 2,
        "composition": {"selected_modules": [fp8_module]},
    }
    (root / "config.json").write_text(json.dumps(config))
    (root / "model.safetensors.index.json").write_text(json.dumps(index))
    (root / "mxwave-manifest.json").write_text(json.dumps(manifest))
    return root


def _spec(
    path: Path,
    model: Path,
    ppl: Path,
    runtime: Path,
    *,
    threshold: float = 9.0,
) -> None:
    checkpoint_sha256 = verify_checkpoint(model, hash_shards=True)["checkpoint_sha256"]
    assert isinstance(checkpoint_sha256, str)
    path.write_text(
        json.dumps(
            {
                "format": SPEC_FORMAT,
                "run_id": "synthetic-qualification",
                "model": {
                    "path": str(model),
                    "label": "synthetic",
                    "hash_shards": True,
                },
                "evidence": [
                    {
                        "id": "ppl",
                        "kind": "perplexity",
                        "path": str(ppl),
                        "bindings": [
                            {
                                "pointer": "/checkpoint_sha256",
                                "structure_pointer": "/checkpoint_sha256",
                            },
                            {"pointer": "/protocol_sha256", "value": _PPL_PROTOCOL},
                        ],
                    },
                    {
                        "id": "runtime",
                        "kind": "runtime",
                        "path": str(runtime),
                        "bindings": [
                            {
                                "pointer": "/checkpoint_sha256",
                                "structure_pointer": "/checkpoint_sha256",
                            },
                            {"pointer": "/protocol_sha256", "value": _RUNTIME_PROTOCOL},
                        ],
                    },
                ],
                "gates": [
                    {
                        "id": "ppl-limit",
                        "evidence": "ppl",
                        "pointer": "/perplexity",
                        "operator": "<=",
                        "threshold": threshold,
                    }
                ],
                "required_capabilities": [
                    "structural",
                    "perplexity",
                    "stock_vllm_load",
                    "fresh_runtime_cache_startup",
                    "peak_host_memory",
                    "peak_accelerator_memory",
                    "ttft",
                    "decode_throughput",
                    "concurrency",
                    "mtp_acceptance",
                    "critical_smoke",
                    "colocated_runtime_client",
                ],
                "success_decision": "qualified-optional",
            }
        )
    )


def test_whole_checkpoint_verification_passes_valid_mxfp4(tmp_path: Path) -> None:
    root = _make_checkpoint(tmp_path)

    report = verify_checkpoint(root)

    assert report["status"] == "passed"
    assert report["target_modules"] == 1
    assert report["ignored_modules"] == 1
    assert report["tensor_count"] == 3


@pytest.mark.parametrize("explicit_null_activations", [False, True])
def test_whole_checkpoint_verification_expands_weight_only_expert_regex(
    tmp_path: Path,
    explicit_null_activations: bool,
) -> None:
    root = _make_regex_weight_only_checkpoint(
        tmp_path,
        explicit_null_activations=explicit_null_activations,
    )

    report = verify_checkpoint(root)

    assert report["status"] == "passed"
    assert report["format"] == "mxfp4-pack-quantized"
    assert report["target_modules"] == 4
    assert report["ignored_modules"] == 2
    assert report["tensor_count"] == 10


def test_whole_checkpoint_verification_rejects_invalid_target_regex(tmp_path: Path) -> None:
    root = _make_regex_weight_only_checkpoint(tmp_path)
    config_path = root / "config.json"
    config = json.loads(config_path.read_text())
    config["quantization_config"]["config_groups"]["group_0"]["targets"] = [
        "re:["
    ]
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match="invalid regex"):
        verify_checkpoint(root)


def test_whole_checkpoint_verification_rejects_invalid_ignore_regex(tmp_path: Path) -> None:
    root = _make_regex_weight_only_checkpoint(tmp_path)
    config_path = root / "config.json"
    config = json.loads(config_path.read_text())
    config["quantization_config"]["ignore"] = ["re:("]
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match="invalid regex"):
        verify_checkpoint(root)


def test_whole_checkpoint_verification_rejects_unmatched_target_regex(
    tmp_path: Path,
) -> None:
    root = _make_regex_weight_only_checkpoint(tmp_path)
    config_path = root / "config.json"
    config = json.loads(config_path.read_text())
    config["quantization_config"]["config_groups"]["group_0"]["targets"] = [
        r"re:^model\.does_not_exist$"
    ]
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match="matches no checkpoint modules"):
        verify_checkpoint(root)


def test_whole_checkpoint_verification_rejects_ambiguous_target_selectors(
    tmp_path: Path,
) -> None:
    root = _make_regex_weight_only_checkpoint(tmp_path)
    config_path = root / "config.json"
    config = json.loads(config_path.read_text())
    selectors = config["quantization_config"]["config_groups"]["group_0"]["targets"]
    selectors.append("model.layers.0.mlp.experts.0.gate_proj")
    selectors.sort()
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match="Ambiguous target selector coverage"):
        verify_checkpoint(root)


def test_whole_checkpoint_verification_rejects_target_ignore_overlap(
    tmp_path: Path,
) -> None:
    root = _make_regex_weight_only_checkpoint(tmp_path)
    config_path = root / "config.json"
    config = json.loads(config_path.read_text())
    config["quantization_config"]["ignore"].append(
        "model.layers.0.mlp.experts.0.gate_proj"
    )
    config["quantization_config"]["ignore"].sort()
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match="Ambiguous target/ignore selector coverage"):
        verify_checkpoint(root)


def test_whole_checkpoint_verification_checks_manifest_selector_identity(
    tmp_path: Path,
) -> None:
    root = _make_regex_weight_only_checkpoint(tmp_path)
    manifest_path = root / "mxwave-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["config_target_patterns"] = [r"re:^wrong$"]
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="manifest config_target_patterns differ"):
        verify_checkpoint(root)


def test_whole_checkpoint_verification_rejects_bad_pair_shape(tmp_path: Path) -> None:
    root = _make_checkpoint(tmp_path, bad_scale_shape=True)

    with pytest.raises(ValueError, match="incompatible packed/scale shapes"):
        verify_checkpoint(root)


def test_whole_checkpoint_verification_supports_mixed_mxfp4_fp8(tmp_path: Path) -> None:
    root = _make_mixed_checkpoint(tmp_path)

    report = verify_checkpoint(root)

    assert report["status"] == "passed"
    assert report["format"] == "mixed-precision"
    assert report["target_modules"] == 2


def test_whole_checkpoint_verification_rejects_uncovered_raw_weight(tmp_path: Path) -> None:
    root = _make_checkpoint(tmp_path, extra_uncovered=True)

    with pytest.raises(ValueError, match="Raw-weight coverage disagreement"):
        verify_checkpoint(root)


def test_whole_checkpoint_verification_rejects_wrong_quantization_scheme(
    tmp_path: Path,
) -> None:
    root = _make_checkpoint(tmp_path)
    config_path = root / "config.json"
    config = json.loads(config_path.read_text())
    config["quantization_config"]["config_groups"]["group_0"]["weights"][
        "dynamic"
    ] = True
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match=r"weights\.dynamic"):
        verify_checkpoint(root)


def test_whole_checkpoint_verification_validates_dynamic_mxfp4_activations(
    tmp_path: Path,
) -> None:
    root = _make_checkpoint(tmp_path)
    config_path = root / "config.json"
    config = json.loads(config_path.read_text())
    config["quantization_config"]["config_groups"]["group_0"][
        "input_activations"
    ]["dynamic"] = False
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match=r"input_activations\.dynamic"):
        verify_checkpoint(root)


def test_checkpoint_fingerprint_changes_when_only_payload_changes(tmp_path: Path) -> None:
    root = _make_checkpoint(tmp_path)
    first = verify_checkpoint(root, hash_shards=True)["checkpoint_sha256"]
    shard = root / "model.safetensors"
    with shard.open("r+b") as handle:
        handle.seek(-1, 2)
        original = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([original[0] ^ 1]))

    second = verify_checkpoint(root, hash_shards=True)["checkpoint_sha256"]

    assert isinstance(first, str)
    assert isinstance(second, str)
    assert first != second


def test_checkpoint_fingerprint_includes_declared_model_assets(tmp_path: Path) -> None:
    root = _make_checkpoint(tmp_path)
    tokenizer = root / "tokenizer_config.json"
    tokenizer.write_text('{"version": 1}')
    manifest_path = root / "mxwave-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["copied_assets"] = [tokenizer.name]
    manifest_path.write_text(json.dumps(manifest))
    first = verify_checkpoint(root, hash_shards=True)["checkpoint_sha256"]

    tokenizer.write_text('{"version": 2}')
    second = verify_checkpoint(root, hash_shards=True)["checkpoint_sha256"]

    assert isinstance(first, str)
    assert isinstance(second, str)
    assert first != second


def test_empty_qualification_contract_is_rejected(tmp_path: Path) -> None:
    model = _make_checkpoint(tmp_path)
    spec = tmp_path / "empty.json"
    spec.write_text(
        json.dumps(
            {
                "format": SPEC_FORMAT,
                "run_id": "empty",
                "model": {
                    "path": str(model),
                    "label": "synthetic",
                    "hash_shards": True,
                },
                "evidence": [],
                "gates": [],
                "required_capabilities": [],
            }
        )
    )

    with pytest.raises(ValueError, match="at least one evidence"):
        load_spec(spec)


def test_qualification_aggregates_evidence_and_passes_frozen_gate(tmp_path: Path) -> None:
    model = _make_checkpoint(tmp_path)
    checkpoint_sha256 = verify_checkpoint(model, hash_shards=True)["checkpoint_sha256"]
    assert isinstance(checkpoint_sha256, str)
    ppl = tmp_path / "ppl.json"
    _ppl_evidence(ppl, checkpoint_sha256)
    runtime = tmp_path / "runtime.json"
    _runtime_evidence(runtime, checkpoint_sha256)
    spec = tmp_path / "spec.json"
    _spec(spec, model, ppl, runtime)

    report, report_path = run_qualification(spec, tmp_path / "run")

    assert report_path.is_file()
    assert (tmp_path / "run" / "qualification.md").is_file()
    assert report["decision"] == "qualified-optional"
    assert report["complete"] is True
    assert report["gates"][0]["pass"] is True
    assert report["capabilities"]["mtp_acceptance"] is True


def test_missing_runtime_capability_is_incomplete_not_pass(tmp_path: Path) -> None:
    model = _make_checkpoint(tmp_path)
    checkpoint_sha256 = verify_checkpoint(model, hash_shards=True)["checkpoint_sha256"]
    assert isinstance(checkpoint_sha256, str)
    ppl = tmp_path / "ppl.json"
    _ppl_evidence(ppl, checkpoint_sha256)
    runtime = tmp_path / "runtime.json"
    _runtime_evidence(runtime, checkpoint_sha256, include_concurrency=False)
    spec = tmp_path / "spec.json"
    _spec(spec, model, ppl, runtime)

    report, _ = run_qualification(spec, tmp_path / "run")

    assert report["decision"] == "incomplete"
    assert report["complete"] is False
    assert report["missing_capabilities"] == ["concurrency"]


def test_wrong_checkpoint_binding_is_incomplete_not_rejected(tmp_path: Path) -> None:
    model = _make_checkpoint(tmp_path)
    checkpoint_sha256 = verify_checkpoint(model, hash_shards=True)["checkpoint_sha256"]
    assert isinstance(checkpoint_sha256, str)
    ppl = tmp_path / "ppl.json"
    _ppl_evidence(ppl, "f" * 64)
    runtime = tmp_path / "runtime.json"
    _runtime_evidence(runtime, checkpoint_sha256)
    spec = tmp_path / "spec.json"
    _spec(spec, model, ppl, runtime)

    report, _ = run_qualification(spec, tmp_path / "run")

    assert report["decision"] == "incomplete"
    assert report["complete"] is False
    assert report["gates"][0]["pass"] is False
    assert report["missing_required_evidence"] == ["ppl"]


def test_failed_gate_rejects_and_cli_returns_two(tmp_path: Path) -> None:
    model = _make_checkpoint(tmp_path)
    checkpoint_sha256 = verify_checkpoint(model, hash_shards=True)["checkpoint_sha256"]
    assert isinstance(checkpoint_sha256, str)
    ppl = tmp_path / "ppl.json"
    _ppl_evidence(ppl, checkpoint_sha256)
    runtime = tmp_path / "runtime.json"
    _runtime_evidence(runtime, checkpoint_sha256)
    spec = tmp_path / "spec.json"
    _spec(spec, model, ppl, runtime, threshold=8.0)

    result = main(["run", "--spec", str(spec), "--output-dir", str(tmp_path / "run")])

    assert result == 2
    report = json.loads((tmp_path / "run" / "qualification.json").read_text())
    assert report["decision"] == "rejected"


def test_hook_materializes_versioned_evidence_without_shell(tmp_path: Path) -> None:
    model = _make_checkpoint(tmp_path)
    checkpoint_sha256 = verify_checkpoint(model, hash_shards=True)["checkpoint_sha256"]
    assert isinstance(checkpoint_sha256, str)
    generated = tmp_path / "generated-ppl.json"
    spec = tmp_path / "spec.json"
    payload_path = tmp_path / "payload.json"
    _ppl_evidence(payload_path, checkpoint_sha256, perplexity=8.0)
    payload = payload_path.read_text()
    payload_path.unlink()
    spec.write_text(
        json.dumps(
            {
                "format": SPEC_FORMAT,
                "run_id": "hook-test",
                "model": {
                    "path": str(model),
                    "label": "synthetic",
                    "hash_shards": True,
                },
                "evidence": [
                    {
                        "id": "ppl",
                        "kind": "perplexity",
                        "path": str(generated),
                        "bindings": [
                            {
                                "pointer": "/checkpoint_sha256",
                                "structure_pointer": "/checkpoint_sha256",
                            },
                            {"pointer": "/protocol_sha256", "value": _PPL_PROTOCOL},
                        ],
                        "hook": {
                            "argv": [
                                sys.executable,
                                "-c",
                                (
                                    "from pathlib import Path; import sys; "
                                    "Path(sys.argv[1]).write_text(sys.argv[2])"
                                ),
                                "{artifact}",
                                payload,
                            ],
                            "timeout_seconds": 10,
                        },
                    }
                ],
                "gates": [
                    {
                        "id": "ppl-limit",
                        "evidence": "ppl",
                        "pointer": "/perplexity",
                        "operator": "<=",
                        "threshold": 8.1,
                    }
                ],
                "required_capabilities": ["structural", "perplexity"],
            }
        )
    )

    report, _ = run_qualification(spec, tmp_path / "run")

    assert report["decision"] == "qualified-default"
    assert generated.is_file()
    assert report["hooks"]["ppl"]["return_code"] == 0
