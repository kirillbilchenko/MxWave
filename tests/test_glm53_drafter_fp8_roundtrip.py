"""Verify scale retention, all finite FP8 encodings, and rejection of bad exports."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from integrations.glm53_mxfp4.drafter_fp8_roundtrip import convert_rows, digest, run


def test_all_finite_fp8_codes_survive_bf16_with_non_power_of_two_scales() -> None:
    codes = torch.arange(256, dtype=torch.uint8)
    codes = codes[torch.isfinite(codes.view(torch.float8_e4m3fn).float())]
    codes = codes.repeat((256 * 128 + codes.numel() - 1) // codes.numel())
    values = codes[:256 * 128].reshape(256, 128).view(torch.float8_e4m3fn)
    scales = torch.tensor([[0.001234567], [1.234567]])
    dense = convert_rows(values, scales, encode=False)
    restored = convert_rows(dense, scales, encode=True)
    assert dense.dtype == torch.bfloat16
    assert torch.equal(restored.view(torch.uint8), values.view(torch.uint8))


def test_export_refuses_overflow_and_invalid_scales() -> None:
    with pytest.raises(ValueError, match="overflow"):
        convert_rows(torch.full((128, 128), 500.0), torch.ones(1, 1), encode=True)
    with pytest.raises(ValueError, match="positive"):
        convert_rows(torch.ones(128, 128), torch.zeros(1, 1), encode=True)
    with pytest.raises(ValueError, match="complete"):
        convert_rows(torch.ones(127, 128), torch.ones(1, 1), encode=True)


def test_complete_file_roundtrip_and_source_identity(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    path = source / "model.safetensors"
    save_file({
        "layers.0.weight": torch.ones(128, 128).to(torch.float8_e4m3fn),
        "layers.0.weight_scale_inv": torch.tensor([[0.012345]]),
        "fc.weight": torch.arange(8).reshape(2, 4).bfloat16(),
    }, str(path), metadata={"format": "pt"})
    (source / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": "fp8"}}))
    expected = digest(path)
    review = run(source, tmp_path / "roundtrip", expected_sha256=expected)
    assert review["file_bytes_identical"] and review["config_bytes_identical"]
    assert review["different_tensor_bytes"] == 0
    assert review["fp8_element_count"] == 128 * 128
    with pytest.raises(ValueError, match="hash"):
        run(source, tmp_path / "bad", expected_sha256="0" * 64)
