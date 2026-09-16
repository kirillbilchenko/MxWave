"""Unit tests for config.json-based input format detection."""

from __future__ import annotations

import json
from pathlib import Path

from mxwave.format import detect_input_format, scale_suffix_for


def _write_config(tmp_path: Path, quantization_config: dict) -> Path:
    cfg = {"model_type": "qwen3", "quantization_config": quantization_config}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    return tmp_path


def test_detects_fp16_when_no_quant_config(tmp_path: Path):
    fmt = detect_input_format(tmp_path)
    assert fmt.kind == "fp16"


def test_detects_mxfp8(tmp_path: Path):
    d = _write_config(tmp_path, {"quant_method": "compressed-tensors", "format": "mxfp8"})
    fmt = detect_input_format(d)
    assert fmt.kind == "mxfp8"
    assert fmt.block_size == 32


def test_detects_block_fp8(tmp_path: Path):
    d = _write_config(tmp_path, {"quant_method": "fp8", "format": "fp8e4m3"})
    fmt = detect_input_format(d)
    assert fmt.kind == "fp8_block"
    assert scale_suffix_for(fmt) == ".weight_scale_inv"


def test_detects_per_channel_fp8(tmp_path: Path):
    d = _write_config(
        tmp_path,
        {
            "quant_method": "compressed-tensors",
            "format": "float-quantized",
            "strategy": "channel",
        },
    )
    fmt = detect_input_format(d)
    assert fmt.kind == "fp8_per_channel"
    assert scale_suffix_for(fmt) == ".weight_scale"


def test_detects_mxfp4(tmp_path: Path):
    d = _write_config(
        tmp_path, {"quant_method": "compressed-tensors", "format": "mxfp4-pack-quantized"}
    )
    fmt = detect_input_format(d)
    assert fmt.kind == "mxfp4"


def test_detects_modelopt_nvfp4(tmp_path: Path):
    d = _write_config(
        tmp_path,
        {"quant_method": "modelopt", "quant_algo": "NVFP4"},
    )
    fmt = detect_input_format(d)
    assert fmt.kind == "nvfp4"
    assert fmt.block_size == 16


def test_unknown_quantized_config_never_defaults_to_fp16(tmp_path: Path):
    d = _write_config(tmp_path, {"quant_method": "gptq", "bits": 4})
    assert detect_input_format(d).kind == "unknown_quantized"
