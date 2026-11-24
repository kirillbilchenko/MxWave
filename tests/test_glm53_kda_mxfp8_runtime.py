"""Verify the runtime encoder preserves the independently measured rounding contract."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

SOURCE = Path(__file__).parents[1] / "integrations" / "glm53_mxfp4"


def module():
    """Load the CPU-testable encoder without importing vLLM or initializing CUDA."""
    for name in ("kda_mxfp8_roundtrip", "kda_mxfp8_runtime"):
        spec = importlib.util.spec_from_file_location(name, SOURCE / (name + ".py"))
        assert spec is not None and spec.loader is not None
        loaded = importlib.util.module_from_spec(spec)
        sys.modules[name] = loaded
        spec.loader.exec_module(loaded)
    return sys.modules["kda_mxfp8_runtime"], sys.modules["kda_mxfp8_roundtrip"]


@pytest.mark.parametrize("chunk_rows", [1, 7, 512])
def test_chunking_preserves_quality_recipe_and_source(chunk_rows):
    runtime, codec = module()
    generator = torch.Generator().manual_seed(20261020)
    weight = torch.randn((19, 64), generator=generator).bfloat16()
    weight[0] = 0
    before = weight.clone()
    values, expected_codes, expected_scales = codec.roundtrip(weight)
    codes, scales, decoded, record = runtime.encode_weight(weight, chunk_rows)
    assert torch.equal(weight, before)
    assert torch.equal(codes, expected_codes)
    assert torch.equal(scales, expected_scales)
    assert torch.equal(decoded.float(), values)
    assert codes.device == scales.device == decoded.device == weight.device
    assert record["source_sha256"] == runtime.digest(before)
    assert record["rounded_sha256"] == runtime.digest(values.bfloat16())
    assert record["encoded_bytes"] == weight.numel() + weight.numel() // 32


@pytest.mark.parametrize(
    "weight,chunk_rows",
    [(torch.zeros(2, 64), 4), (torch.zeros(2, 63).bfloat16(), 4),
     (torch.zeros(0, 64).bfloat16(), 4), (torch.zeros(2, 0).bfloat16(), 4),
     (torch.zeros(2, 64).bfloat16(), 0)],
)
def test_rejects_invalid_conversion_contract(weight, chunk_rows):
    runtime, _ = module()
    with pytest.raises(ValueError):
        runtime.encode_weight(weight, chunk_rows)
