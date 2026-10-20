"""Fail-closed metadata coverage contracts for the GLM reuse audit."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, ClassVar, Self

import pytest

from integrations.glm53_mxfp4.audit_checkpoint import analyze, fetch, validate_header


def _fixture() -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    targets = [
        f"model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}"
        for layer in range(3, 46)
        for expert in range(2)
        for projection in ("gate_proj", "up_proj", "down_proj")
    ]
    group = {
        "weights": {
            "num_bits": 4,
            "group_size": 32,
            "strategy": "group",
            "type": "float",
            "scale_dtype": "torch.uint8",
            "dynamic": False,
        },
        "targets": targets,
        "format": "mxfp4-pack-quantized",
    }
    other = deepcopy(group)
    other["weights"]["num_bits"] = 8
    other["targets"] = ["Linear"]
    other["format"] = "mxfp8-quantized"
    config = {
        "model_type": "glm5_next",
        "architectures": ["Glm5NextForConditionalGeneration"],
        "text_config": {
            "model_type": "glm5_next_text",
            "num_hidden_layers": 45,
            "n_routed_experts": 2,
            "first_k_dense_replace": 3,
            "num_nextn_predict_layers": 1,
            "hidden_size": 32,
            "moe_intermediate_size": 32,
        },
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "config_groups": {"experts": group, "fallback": other},
            "ignore": ["lm_head", "model.language_model.layers.3.mlp.gate"],
        },
    }
    tensors: dict[str, Any] = {}
    offset = 0

    def add(name: str, shape: list[int], dtype: str, size: int) -> None:
        nonlocal offset
        tensors[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size

    for target in targets:
        add(target + ".weight_packed", [32, 16], "U8", 512)
        add(target + ".weight_scale", [32, 1], "U8", 32)
    for module in (
        "model.language_model.layers.0.mlp.up_proj",
        "model.language_model.layers.3.mlp.shared_experts.up_proj",
    ):
        add(module + ".weight", [32, 32], "F8_E4M3", 1024)
        add(module + ".weight_scale", [32, 1], "U8", 32)
    add("lm_head.weight", [10, 32], "BF16", 640)
    add("model.language_model.layers.3.mlp.gate.weight", [2, 32], "BF16", 128)
    return config, dict.fromkeys(tensors, "model.safetensors"), {"model.safetensors": tensors}


def test_audit_resolves_explicit_experts_and_generic_fp8_without_overlap() -> None:
    config, index, shards = _fixture()
    report = analyze(config, index, shards)
    assert report["base_expert_matrices"] == 42 * 2 * 3
    assert report["mtp_expert_matrices"] == 2 * 3
    assert report["other_mxfp8_matrices"] == 2
    assert report["families"]["routers/BF16"]["modules"] == 1
    assert report["families"]["leading_dense_mlp/MXFP8"]["modules"] == 1
    assert report["families"]["shared_experts/MXFP8"]["modules"] == 1
    assert report["families"]["mtp/routed_experts/MXFP4"]["modules"] == 6


@pytest.mark.parametrize("defect", ["missing_expert", "wrong_scale", "ignored_fp8", "index_owner"])
def test_audit_rejects_inconsistent_realizations(defect: str) -> None:
    config, index, shards = _fixture()
    tensors = shards["model.safetensors"]
    module = "model.language_model.layers.3.mlp.experts.0.up_proj"
    if defect == "missing_expert":
        del index[module + ".weight_packed"]
        del tensors[module + ".weight_packed"]
    elif defect == "wrong_scale":
        tensors[module + ".weight_scale"]["shape"] = [32, 2]
    elif defect == "ignored_fp8":
        config["quantization_config"]["ignore"].append("model.language_model.layers.0.mlp.up_proj")
    else:
        index[module + ".weight_packed"] = "wrong.safetensors"
    with pytest.raises(ValueError):
        analyze(config, index, shards)


def test_header_accounting_rejects_overlap_and_truncated_file() -> None:
    header = {"a": {"dtype": "U8", "shape": [8], "data_offsets": [0, 8]}}
    assert len(validate_header(header, 64, 80)) == 1
    with pytest.raises(ValueError, match="file size"):
        validate_header(header, 64, 79)
    header["b"] = {"dtype": "U8", "shape": [4], "data_offsets": [4, 8]}
    with pytest.raises(ValueError, match="overlap"):
        validate_header(header, 64, 80)


def test_range_fetch_rejects_ignored_range_before_reading_body(monkeypatch: Any) -> None:
    class Response:
        status = 200
        headers: ClassVar[dict[str, str]] = {}

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def read(self, _limit: int) -> bytes:
            pytest.fail("An ignored range must be rejected without reading a shard payload")

    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: Response())
    with pytest.raises(ValueError, match="exact metadata range"):
        fetch("https://example.invalid/shard", 0, 8)
