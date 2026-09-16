"""Unit tests for the MXFP4 quantization core."""

from __future__ import annotations

import pytest
import torch

from mxstream.core import BLOCK_SIZE, compute_block_hessian, dequant_mxfp4, quantize_mxfp4


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
        hessian_rounding_sweeps=1,
    )
    assert torch.equal(gamma_scales, hessian_scales)
    assert torch.equal(gamma_packed, hessian_packed)


def test_hessian_rounding_reduces_correlated_error_at_fixed_scale():
    weight = torch.zeros(1, BLOCK_SIZE)
    weight[0, :2] = 0.74
    weight[0, 2] = 6.0
    hessian = torch.eye(BLOCK_SIZE).unsqueeze(0)
    hessian[0, 0, 1] = 0.99
    hessian[0, 1, 0] = 0.99

    nearest_packed, nearest_scales = quantize_mxfp4(
        weight,
        scale_percentile=100.0,
        hessian=hessian,
        mse_clip_depth=0,
    )
    refined_packed, refined_scales = quantize_mxfp4(
        weight,
        scale_percentile=100.0,
        hessian=hessian,
        mse_clip_depth=0,
        hessian_rounding_sweeps=1,
    )
    nearest = dequant_mxfp4(nearest_packed, nearest_scales, weight.shape)
    refined = dequant_mxfp4(refined_packed, refined_scales, weight.shape)
    nearest_error = nearest - weight
    refined_error = refined - weight
    nearest_loss = torch.einsum("oi,bij,oj->", nearest_error, hessian, nearest_error)
    refined_loss = torch.einsum("oi,bij,oj->", refined_error, hessian, refined_error)

    assert torch.equal(nearest_scales, refined_scales)
    assert refined_loss < nearest_loss / 10


def test_hessian_error_feedback_reduces_correlated_error_at_fixed_scale():
    weight = torch.zeros(1, BLOCK_SIZE)
    weight[0, :2] = 0.74
    weight[0, 2] = 6.0
    hessian = torch.eye(BLOCK_SIZE).unsqueeze(0)
    hessian[0, 0, 1] = 0.99
    hessian[0, 1, 0] = 0.99

    nearest_packed, nearest_scales = quantize_mxfp4(
        weight,
        scale_percentile=100.0,
        hessian=hessian,
        mse_clip_depth=0,
    )
    feedback_packed, feedback_scales = quantize_mxfp4(
        weight,
        scale_percentile=100.0,
        hessian=hessian,
        mse_clip_depth=0,
        hessian_error_feedback=True,
        hessian_feedback_activation_order=False,
    )
    nearest = dequant_mxfp4(nearest_packed, nearest_scales, weight.shape)
    feedback = dequant_mxfp4(feedback_packed, feedback_scales, weight.shape)
    nearest_error = nearest - weight
    feedback_error = feedback - weight
    nearest_loss = torch.einsum("oi,bij,oj->", nearest_error, hessian, nearest_error)
    feedback_loss = torch.einsum("oi,bij,oj->", feedback_error, hessian, feedback_error)

    assert torch.equal(nearest_scales, feedback_scales)
    assert feedback_loss < nearest_loss / 10


def test_hessian_error_feedback_matches_nearest_for_diagonal_hessian():
    generator = torch.Generator().manual_seed(456)
    weight = torch.randn(8, 64, generator=generator)
    diagonal = torch.rand(2, BLOCK_SIZE, generator=generator).add(0.1)
    hessian = torch.diag_embed(diagonal)

    nearest_packed, nearest_scales = quantize_mxfp4(
        weight,
        hessian=hessian,
        mse_clip_depth=4,
    )
    feedback_packed, feedback_scales = quantize_mxfp4(
        weight,
        hessian=hessian,
        mse_clip_depth=4,
        hessian_error_feedback=True,
    )

    assert torch.equal(nearest_scales, feedback_scales)
    assert torch.equal(nearest_packed, feedback_packed)


def test_hessian_error_feedback_respects_plain_mse_trust_region():
    weight = torch.zeros(1, BLOCK_SIZE)
    weight[0, :2] = 0.74
    weight[0, 2] = 6.0
    hessian = torch.eye(BLOCK_SIZE).unsqueeze(0)
    hessian[0, 0, 1] = 0.99
    hessian[0, 1, 0] = 0.99

    nearest_packed, nearest_scales = quantize_mxfp4(
        weight,
        scale_percentile=100.0,
        hessian=hessian,
        mse_clip_depth=0,
    )
    constrained_packed, constrained_scales = quantize_mxfp4(
        weight,
        scale_percentile=100.0,
        hessian=hessian,
        mse_clip_depth=0,
        hessian_error_feedback=True,
        hessian_feedback_activation_order=False,
        hessian_feedback_max_mse_ratio=1.05,
    )
    relaxed_packed, relaxed_scales = quantize_mxfp4(
        weight,
        scale_percentile=100.0,
        hessian=hessian,
        mse_clip_depth=0,
        hessian_error_feedback=True,
        hessian_feedback_activation_order=False,
        hessian_feedback_max_mse_ratio=1.10,
    )

    assert torch.equal(nearest_scales, constrained_scales)
    assert torch.equal(nearest_scales, relaxed_scales)
    assert torch.equal(nearest_packed, constrained_packed)
    assert not torch.equal(nearest_packed, relaxed_packed)


def test_heldout_hessian_selects_only_feedback_that_generalizes() -> None:
    weight = torch.zeros(1, BLOCK_SIZE)
    weight[0, :2] = 0.74
    weight[0, 2] = 6.0
    training_hessian = torch.eye(BLOCK_SIZE).unsqueeze(0)
    training_hessian[0, 0, 1] = 0.99
    training_hessian[0, 1, 0] = 0.99
    ordinary_heldout_hessian = torch.eye(BLOCK_SIZE).unsqueeze(0)

    nearest_packed, nearest_scales = quantize_mxfp4(
        weight,
        scale_percentile=100.0,
        hessian=training_hessian,
        mse_clip_depth=0,
    )
    training_selected, training_scales = quantize_mxfp4(
        weight,
        scale_percentile=100.0,
        hessian=training_hessian,
        mse_clip_depth=0,
        hessian_error_feedback=True,
        hessian_feedback_activation_order=False,
        feedback_selection_hessian=training_hessian,
    )
    heldout_selected, heldout_scales = quantize_mxfp4(
        weight,
        scale_percentile=100.0,
        hessian=training_hessian,
        mse_clip_depth=0,
        hessian_error_feedback=True,
        hessian_feedback_activation_order=False,
        feedback_selection_hessian=ordinary_heldout_hessian,
    )

    assert torch.equal(nearest_scales, training_scales)
    assert torch.equal(nearest_scales, heldout_scales)
    assert not torch.equal(nearest_packed, training_selected)
    assert torch.equal(nearest_packed, heldout_selected)


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


def test_hessian_rounding_requires_hessian():
    with pytest.raises(ValueError, match="requires method='mse' and a block Hessian"):
        quantize_mxfp4(torch.randn(2, BLOCK_SIZE), hessian_rounding_sweeps=1)


def test_hessian_error_feedback_requires_hessian():
    with pytest.raises(ValueError, match="requires method='mse' and a block Hessian"):
        quantize_mxfp4(torch.randn(2, BLOCK_SIZE), hessian_error_feedback=True)


def test_hessian_refinement_modes_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        quantize_mxfp4(
            torch.randn(2, BLOCK_SIZE),
            hessian=torch.eye(BLOCK_SIZE).unsqueeze(0),
            hessian_rounding_sweeps=1,
            hessian_error_feedback=True,
        )


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
