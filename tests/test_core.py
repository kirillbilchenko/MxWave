"""Unit tests for the MXFP4 quantization core."""

from __future__ import annotations

import torch

from mxstream.core import BLOCK_SIZE, compute_block_hessian, quantize_mxfp4


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
    from mxstream.core import dequant_mxfp4

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
