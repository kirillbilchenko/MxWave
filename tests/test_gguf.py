"""Tests for lossless MxWave MXFP4 GGUF export primitives."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import Tensor

import mxwave.gguf as gguf_export
from mxwave.gguf import (
    Qwen35LinearAttentionConfig,
    install_llama_cpp_exporter,
    repack_mxfp4_blocks,
    transform_qwen35_mxfp4,
)


def _pack_codes(codes: Tensor) -> Tensor:
    pairs = codes.reshape(*codes.shape[:-1], codes.shape[-1] // 2, 2)
    return (pairs[..., 0] | (pairs[..., 1] << 4)).to(torch.uint8)


def _unpack_codes(packed: Tensor) -> Tensor:
    return torch.stack((packed & 0x0F, (packed >> 4) & 0x0F), dim=-1).reshape(
        *packed.shape[:-1], packed.shape[-1] * 2
    )


def test_repack_mxfp4_blocks_matches_ggml_layout_byte_for_byte() -> None:
    codes = torch.tensor([list(range(16)) + list(reversed(range(16)))], dtype=torch.uint8)
    packed = _pack_codes(codes)
    scale = torch.tensor([[137]], dtype=torch.uint8)

    result = repack_mxfp4_blocks(packed, scale)

    expected_codes = [int(codes[0, index] | (codes[0, index + 16] << 4)) for index in range(16)]
    expected = np.array([[137, *expected_codes]], dtype=np.uint8)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    ("packed", "scale", "error"),
    [
        (
            torch.zeros((1, 16), dtype=torch.int8),
            torch.zeros((1, 1), dtype=torch.uint8),
            TypeError,
        ),
        (
            torch.zeros(16, dtype=torch.uint8),
            torch.zeros((1, 1), dtype=torch.uint8),
            ValueError,
        ),
        (
            torch.zeros((1, 15), dtype=torch.uint8),
            torch.zeros((1, 1), dtype=torch.uint8),
            ValueError,
        ),
        (
            torch.zeros((2, 16), dtype=torch.uint8),
            torch.zeros((2, 2), dtype=torch.uint8),
            ValueError,
        ),
    ],
)
def test_repack_mxfp4_blocks_rejects_invalid_inputs(
    packed: Tensor,
    scale: Tensor,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        repack_mxfp4_blocks(packed, scale)


def test_qwen35_qkv_reorders_only_value_rows() -> None:
    config = Qwen35LinearAttentionConfig(2, 4, 2, 2)
    packed = torch.arange(16, dtype=torch.uint8).reshape(16, 1).expand(-1, 16).clone()
    scale = torch.arange(16, dtype=torch.uint8).reshape(16, 1)

    transformed, transformed_scale = transform_qwen35_mxfp4(
        "model.layers.0.linear_attn.in_proj_qkv.weight",
        packed,
        scale,
        config,
    )

    expected_rows = torch.tensor([*range(8), 8, 9, 12, 13, 10, 11, 14, 15])
    torch.testing.assert_close(transformed[:, 0], expected_rows.to(torch.uint8))
    torch.testing.assert_close(transformed_scale[:, 0], expected_rows.to(torch.uint8))


@pytest.mark.parametrize(
    ("suffix", "value_head_dim", "expected_rows"),
    [
        ("in_proj_z", 2, [0, 1, 4, 5, 2, 3, 6, 7]),
        ("in_proj_a", 1, [0, 2, 1, 3]),
        ("in_proj_b", 1, [0, 2, 1, 3]),
    ],
)
def test_qwen35_auxiliary_projections_reorder_rows(
    suffix: str,
    value_head_dim: int,
    expected_rows: list[int],
) -> None:
    config = Qwen35LinearAttentionConfig(2, 4, 2, value_head_dim)
    rows = len(expected_rows)
    packed = torch.arange(rows, dtype=torch.uint8).reshape(rows, 1).expand(-1, 16).clone()
    scale = torch.arange(rows, dtype=torch.uint8).reshape(rows, 1)

    transformed, transformed_scale = transform_qwen35_mxfp4(
        f"model.layers.0.linear_attn.{suffix}.weight",
        packed,
        scale,
        config,
    )

    expected = torch.tensor(expected_rows, dtype=torch.uint8)
    torch.testing.assert_close(transformed[:, 0], expected)
    torch.testing.assert_close(transformed_scale[:, 0], expected)


def test_qwen35_out_projection_reorders_aligned_mxfp4_blocks() -> None:
    config = Qwen35LinearAttentionConfig(2, 4, 8, 32)
    codes = torch.cat(
        tuple(torch.full((32,), value, dtype=torch.uint8) for value in (1, 2, 3, 4))
    ).reshape(1, 128)
    packed = _pack_codes(codes)
    scale = torch.tensor([[10, 20, 30, 40]], dtype=torch.uint8)

    transformed, transformed_scale = transform_qwen35_mxfp4(
        "model.layers.0.linear_attn.out_proj.weight",
        packed,
        scale,
        config,
    )

    transformed_codes = _unpack_codes(transformed).reshape(4, 32)
    assert transformed_codes[:, 0].tolist() == [1, 3, 2, 4]
    assert transformed_scale.tolist() == [[10, 30, 20, 40]]


def test_qwen35_out_projection_rejects_unaligned_head_permutation() -> None:
    config = Qwen35LinearAttentionConfig(2, 4, 8, 16)
    packed = torch.zeros((2, 32), dtype=torch.uint8)
    scale = torch.zeros((2, 2), dtype=torch.uint8)

    with pytest.raises(ValueError, match="not aligned"):
        transform_qwen35_mxfp4(
            "model.layers.0.linear_attn.out_proj.weight",
            packed,
            scale,
            config,
        )


def _quantization_config() -> dict[str, object]:
    return {
        "quant_method": "compressed-tensors",
        "format": "mxfp4-pack-quantized",
    }


def _install_fake_llama_cpp(monkeypatch: pytest.MonkeyPatch) -> tuple[type[object], object]:
    class FakeModelBase:
        def prepare_tensors(self) -> None:
            self.quantization_seen_by_upstream = self.hparams.get("quantization_config")

    architectures = SimpleNamespace(QWEN35="qwen35", MMPROJ="mmproj")
    fake_modules = {
        "gguf": SimpleNamespace(MODEL_ARCH=architectures),
        "conversion.base": SimpleNamespace(
            ModelBase=FakeModelBase,
            LazyTorchTensor=object(),
            logger=object(),
        ),
    }
    monkeypatch.setattr(
        gguf_export.importlib,
        "import_module",
        lambda name: fake_modules[name],
    )
    install_llama_cpp_exporter()
    return FakeModelBase, architectures


def test_main_export_keeps_strict_mxfp4_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_base, architectures = _install_fake_llama_cpp(monkeypatch)
    converted: list[object] = []
    monkeypatch.setattr(
        gguf_export,
        "_write_compressed_mxfp4",
        lambda model, *_args: converted.append(model),
    )
    raw_quantization = _quantization_config()
    model = SimpleNamespace(
        hparams={"quantization_config": raw_quantization},
        global_config={},
        model_arch=architectures.QWEN35,
        mtp_only=False,
        model_tensors={"model.layers.0.weight_packed": object()},
    )

    model_base.prepare_tensors(model)

    assert converted == [model]
    assert model.quantization_seen_by_upstream is None
    assert model.hparams["quantization_config"] is raw_quantization
    assert model._is_mxfp4 is True


@pytest.mark.parametrize(
    ("model_arch", "mtp_only", "quantization_location"),
    [
        ("qwen35", True, "hparams"),
        ("mmproj", False, "global_config"),
    ],
)
def test_float_only_export_modes_delegate_without_mxwave_quantization(
    monkeypatch: pytest.MonkeyPatch,
    model_arch: str,
    mtp_only: bool,
    quantization_location: str,
) -> None:
    model_base, _architectures = _install_fake_llama_cpp(monkeypatch)
    raw_quantization = _quantization_config()
    hparams: dict[str, object] = {}
    global_config: dict[str, object] = {}
    config = hparams if quantization_location == "hparams" else global_config
    config["quantization_config"] = raw_quantization
    model = SimpleNamespace(
        hparams=hparams,
        global_config=global_config,
        model_arch=model_arch,
        mtp_only=mtp_only,
        model_tensors={"mtp.layers.0.mlp.up_proj.weight": object()},
    )

    model_base.prepare_tensors(model)

    assert model.quantization_seen_by_upstream is None
    assert config["quantization_config"] is raw_quantization
    assert not hasattr(model, "_is_mxfp4")


@pytest.mark.parametrize(("model_arch", "mtp_only"), [("qwen35", True), ("mmproj", False)])
def test_float_only_export_modes_reject_quantized_auxiliary_tensors(
    monkeypatch: pytest.MonkeyPatch,
    model_arch: str,
    mtp_only: bool,
) -> None:
    model_base, _architectures = _install_fake_llama_cpp(monkeypatch)
    model = SimpleNamespace(
        hparams={"quantization_config": _quantization_config()},
        global_config={},
        model_arch=model_arch,
        mtp_only=mtp_only,
        model_tensors={"mtp.layers.0.mlp.up_proj.weight_packed": object()},
    )

    with pytest.raises(ValueError, match="unsupported MXFP4 tensors"):
        model_base.prepare_tensors(model)
