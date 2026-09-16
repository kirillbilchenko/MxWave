"""Unit tests for the MXFP4 quantization core."""

from __future__ import annotations

import pytest
import torch

from mxwave.core import BLOCK_SIZE, compute_block_hessian, dequant_mxfp4, quantize_mxfp4


def test_quantize_shape_and_dtypes():
    w = torch.randn(64, 128)
    packed, scales = quantize_mxfp4(w)
    assert packed.shape == (64, 128 // 2)
    assert scales.shape == (64, 128 // BLOCK_SIZE)
    assert packed.dtype == torch.uint8
    assert scales.dtype == torch.uint8


def test_quantize_reconstruction_is_lossy_but_close():
    w = torch.randn(32, 64)
    packed, scales = quantize_mxfp4(w, scale_percentile=100.0)
    # Dequantize via core helpers
    from mxwave.core import dequant_mxfp4

    recon = dequant_mxfp4(packed, scales, w.shape)
    err = (recon - w).abs().mean().item()
    assert err < 0.5  # 4-bit error should be modest


def test_hessian_weighted_quantize_runs():
    w = torch.randn(64, 128)
    X = torch.randn(32, 128)
    H = compute_block_hessian(X)
    packed, scales = quantize_mxfp4(w, hessian=H)
    assert packed.shape == (64, 64)
    assert scales.shape == (64, 4)


def test_gamma_weighted_quantize_runs():
    w = torch.randn(64, 128)
    gamma = torch.rand(128)
    packed, _ = quantize_mxfp4(w, gamma=gamma)
    assert packed.shape == (64, 64)


def test_diagonal_hessian_matches_rms_channel_weighting():
    generator = torch.Generator().manual_seed(123)
    weight = torch.randn(8, 64, generator=generator)
    rms = torch.rand(64, generator=generator).add(0.1)
    hessian = torch.diag_embed(rms.reshape(2, 32).square())
    gamma_packed, gamma_scales = quantize_mxfp4(
        weight,
        gamma=rms,
        mse_clip_depth=4,
    )
    hessian_packed, hessian_scales = quantize_mxfp4(
        weight,
        hessian=hessian,
        mse_clip_depth=4,
    )
    assert torch.equal(gamma_scales, hessian_scales)
    assert torch.equal(gamma_packed, hessian_packed)


def test_deeper_calibrated_search_can_clip_unimportant_outlier():
    weight = torch.full((1, 32), 0.1)
    weight[0, 0] = 8.0
    magnitudes = torch.ones(32)
    magnitudes[0] = 1e-4

    shallow_packed, shallow_scales = quantize_mxfp4(
        weight,
        gamma=magnitudes,
        mse_clip_depth=1,
    )
    deep_packed, deep_scales = quantize_mxfp4(
        weight,
        gamma=magnitudes,
        mse_clip_depth=4,
    )
    shallow = dequant_mxfp4(shallow_packed, shallow_scales, weight.shape)
    deep = dequant_mxfp4(deep_packed, deep_scales, weight.shape)
    shallow_error = ((shallow - weight).square() * magnitudes.square()).sum()
    deep_error = ((deep - weight).square() * magnitudes.square()).sum()

    assert deep_scales[0, 0] < shallow_scales[0, 0]
    assert deep_error < shallow_error / 10


def test_no_clipping_candidate_protects_important_outlier():
    weight = torch.full((1, 32), 0.1)
    weight[0, 0] = 8.0
    _packed, scales = quantize_mxfp4(weight, mse_clip_depth=8)
    assert scales[0, 0].item() == 128


def test_hessian_and_gamma_are_mutually_exclusive():
    w = torch.randn(32, 64)
    X = torch.randn(16, 64)
    H = compute_block_hessian(X)
    try:
        quantize_mxfp4(w, gamma=torch.rand(64), hessian=H)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_compute_block_hessian_shape():
    X = torch.randn(32, 128)
    H = compute_block_hessian(X)
    assert H.shape == (128 // BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)


def test_rtn_scale_matches_compressed_tensors_reference_edges():
    maxima = torch.tensor([0.0, 1e-7, 1e-6, 6.0, 6.99, 7.0, 10.0])
    weight = torch.zeros(len(maxima), BLOCK_SIZE)
    weight[:, 0] = maxima
    packed, scales = quantize_mxfp4(weight, method="rtn")
    assert scales[:, 0].tolist() == [0, 101, 105, 127, 127, 128, 128]
    reconstructed = dequant_mxfp4(packed, scales, weight.shape)
    assert reconstructed[3, 0] == 6.0
    assert reconstructed[5, 0] == 6.0


def test_rtn_rejects_calibration_weighting():
    with torch.no_grad(), pytest.raises(ValueError, match="rtn does not use"):
        quantize_mxfp4(torch.randn(2, 32), method="rtn", gamma=torch.ones(32))
