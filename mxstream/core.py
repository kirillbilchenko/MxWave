"""MXFP4 E2M1 quantization core.

MXFP4 (OCP MX spec): 4 bits per value, E2M1, 8 positive magnitudes
{0, 0.5, 1, 1.5, 2, 3, 4, 6}. Every 32 consecutive input channels share one
e8m0 scale (2**k, integer k). Two 4-bit codes are packed per byte.

Scale selection is MSE-optimal over an explicit candidate set, optionally
weighted by per-channel activation statistics or a block Hessian. Experimental
block-local coordinate refinement and GPTQ-style error feedback can then refine
individual E2M1 codes against the same Hessian. This is a clean-room
implementation against the public MX spec and published quantization methods.
"""

from __future__ import annotations

from typing import Literal

import torch

BLOCK_SIZE = 32
QuantizationMethod = Literal["rtn", "mse"]

# MXFP4 E2M1 representable positive values and their rounding boundaries.
_POS_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
_BOUNDARIES = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32)
_SIGNED_VALUES = torch.cat((_POS_VALUES, -_POS_VALUES))


def compute_block_hessian(
    X: torch.Tensor,
    block_size: int = BLOCK_SIZE,
    damp: float = 1e-6,
) -> torch.Tensor:
    """Compute block-diagonal Hessian H_b = X_b^T @ X_b / N for each block.

    Used for AWQ/GPTQ-style scale selection: minimize reconstruction error
    trace(dW @ H @ dW^T) rather than plain MSE.

    Args:
        X: [N, in_features] activation matrix.
        block_size: Block size for MXFP4 quantization.
        damp: Dampening added to the diagonal for stability.

    Returns:
        [num_blocks, block_size, block_size] symmetric PSD.
    """
    assert X.ndim == 2, f"Expected 2D activation matrix, got shape {X.shape}"
    N, K = X.shape
    assert K % block_size == 0, f"in_features {K} not divisible by block_size {block_size}"

    num_blocks = K // block_size
    X_b = X.float().reshape(N, num_blocks, block_size)
    H = torch.einsum("nbs,nbt->bst", X_b, X_b) / max(N, 1)
    H.diagonal(dim1=-2, dim2=-1).add_(damp)
    return H


def _round_to_mxfp4(scaled: torch.Tensor) -> torch.Tensor:
    """Round values in [-6, 6] to the nearest MXFP4 representable value (dequantized floats)."""
    abs_scaled = scaled.abs().clamp(max=6.0)
    bucket = torch.searchsorted(_BOUNDARIES.to(scaled.device), abs_scaled.reshape(-1))
    dequant_abs = _POS_VALUES.to(scaled.device)[bucket].reshape_as(abs_scaled)
    return dequant_abs * scaled.sign()


def _rtn_scale_exponent(block_max: torch.Tensor) -> torch.Tensor:
    """Return the compressed-tensors RTN E8M0 exponent for each block maximum.

    ``compressed-tensors`` rounds the maximum to an FP4-aware power of two and
    subtracts ``floor(log2(6)) == 2``.  Expressing the rule arithmetically keeps
    mxstream independent from that package while producing the same scale.
    """
    nonzero = block_max > 0
    safe_max = block_max.clamp(min=torch.finfo(torch.float32).tiny)
    base_exp = torch.floor(torch.log2(safe_max))
    significand = safe_max / torch.pow(2.0, base_exp)
    rounded_exp = base_exp + (significand >= 1.75).to(base_exp.dtype)
    raw_exp = rounded_exp - 2.0
    return torch.where(nonzero, raw_exp, torch.full_like(raw_exp, -127.0))


def _nearest_mxfp4_codes(scaled: torch.Tensor) -> torch.Tensor:
    """Return packed-nibble values for nearest representable E2M1 numbers."""
    abs_scaled = scaled.abs().clamp(max=6.0)
    bucket = torch.searchsorted(_BOUNDARIES.to(scaled.device), abs_scaled.reshape(-1))
    sign_mask = (scaled.reshape(-1) < 0).to(torch.uint8) * 8
    return (bucket.to(torch.uint8) + sign_mask).reshape_as(scaled)


def _refine_codes_with_hessian(
    tensor: torch.Tensor,
    scales: torch.Tensor,
    codes: torch.Tensor,
    hessian: torch.Tensor,
    sweeps: int,
) -> torch.Tensor:
    """Minimize block reconstruction loss by exact coordinate updates.

    For a fixed block scale and all other codes fixed, every coordinate tries
    all 16 E2M1 nibble values and chooses the minimum of ``dW.T @ H @ dW``.
    Each update therefore cannot increase the supplied block-Hessian objective.
    """
    values = _SIGNED_VALUES.to(device=tensor.device)
    refined = codes.clone()
    reconstructed = values[refined.long()] * scales.unsqueeze(-1)
    error = reconstructed - tensor
    diagonal = hessian.diagonal(dim1=-2, dim2=-1)

    for _ in range(sweeps):
        for coordinate in range(BLOCK_SIZE):
            hessian_column = hessian[:, :, coordinate]
            cross = torch.einsum("obs,bs->ob", error, hessian_column)
            cross = cross - error[..., coordinate] * diagonal[:, coordinate].unsqueeze(0)
            candidate_error = (
                scales.unsqueeze(-1) * values.reshape(1, 1, -1)
                - tensor[..., coordinate].unsqueeze(-1)
            )
            candidate_cost = (
                diagonal[:, coordinate].reshape(1, -1, 1) * candidate_error.square()
                + 2.0 * cross.unsqueeze(-1) * candidate_error
            )
            best_cost, best = candidate_cost.min(dim=-1)
            current = refined[..., coordinate].long()
            current_cost = candidate_cost.gather(-1, current.unsqueeze(-1)).squeeze(-1)
            best = torch.where(current_cost <= best_cost, current, best)
            refined[..., coordinate] = best.to(torch.uint8)
            error[..., coordinate] = candidate_error.gather(-1, best.unsqueeze(-1)).squeeze(-1)
    return refined


def _codes_with_hessian_error_feedback(
    tensor: torch.Tensor,
    scales: torch.Tensor,
    hessian: torch.Tensor,
    *,
    damp_percent: float,
    activation_order: bool,
) -> torch.Tensor:
    """Quantize fixed MXFP4 grids with block-local GPTQ-style error feedback.

    Each MXFP4 block is independently damped and factorized. Columns are then
    quantized sequentially while their error is propagated to the remaining
    columns through the inverse-Hessian Cholesky factor. Optional activation
    ordering is applied only during this optimization and undone before packing,
    so neither the emitted tensor layout nor inference changes.

    This deliberately operates on the recorded block-diagonal Hessian. It is a
    bounded-memory approximation to full-layer GPTQ, not a claim of equivalence.
    """
    ordered_tensor = tensor
    ordered_hessian = hessian
    permutation: torch.Tensor | None = None
    if activation_order:
        permutation = torch.argsort(
            hessian.diagonal(dim1=-2, dim2=-1),
            dim=-1,
            descending=True,
            stable=True,
        )
        ordered_tensor = torch.gather(
            tensor,
            -1,
            permutation.unsqueeze(0).expand(tensor.shape[0], -1, -1),
        )
        ordered_hessian = torch.gather(
            hessian,
            1,
            permutation.unsqueeze(-1).expand(-1, -1, BLOCK_SIZE),
        )
        ordered_hessian = torch.gather(
            ordered_hessian,
            2,
            permutation.unsqueeze(1).expand(-1, BLOCK_SIZE, -1),
        )

    damped = ordered_hessian.clone()
    diagonal = damped.diagonal(dim1=-2, dim2=-1)
    diagonal_mean = diagonal.mean(dim=-1).clamp(min=torch.finfo(damped.dtype).eps)
    diagonal.add_(diagonal_mean.unsqueeze(-1) * (damp_percent / 100.0))
    cholesky = torch.linalg.cholesky(damped)
    inverse = torch.cholesky_inverse(cholesky)
    inverse_factor = torch.linalg.cholesky(inverse, upper=True)

    working = ordered_tensor.clone()
    codes = torch.empty_like(working, dtype=torch.uint8)
    values = _SIGNED_VALUES.to(device=tensor.device)
    for coordinate in range(BLOCK_SIZE):
        coordinate_codes = _nearest_mxfp4_codes(
            (working[..., coordinate] / scales).clamp(-6.0, 6.0)
        )
        codes[..., coordinate] = coordinate_codes
        quantized = values[coordinate_codes.long()] * scales
        divisor = inverse_factor[:, coordinate, coordinate].unsqueeze(0)
        error = (working[..., coordinate] - quantized) / divisor
        working[..., coordinate:] -= (
            error.unsqueeze(-1)
            * inverse_factor[:, coordinate, coordinate:].unsqueeze(0)
        )

    if permutation is None:
        return codes
    return torch.empty_like(codes).scatter(
        -1,
        permutation.unsqueeze(0).expand(tensor.shape[0], -1, -1),
        codes,
    )


def quantize_mxfp4(
    tensor: torch.Tensor,
    scale_percentile: float = 99.5,
    gamma: torch.Tensor | None = None,
    hessian: torch.Tensor | None = None,
    method: QuantizationMethod = "mse",
    mse_clip_depth: int = 1,
    hessian_rounding_sweeps: int = 0,
    hessian_error_feedback: bool = False,
    hessian_feedback_damp_percent: float = 1.0,
    hessian_feedback_activation_order: bool = True,
    hessian_feedback_max_mse_ratio: float | None = None,
    feedback_selection_hessian: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a float tensor to MXFP4 with MSE-optimal scale selection.

    With ``method="mse"``, candidate exponents from ``floor-mse_clip_depth``
    through ``floor+1`` are evaluated for every block. The smallest exponent
    that cannot clip the true maximum is always included as a control. With
    ``method="rtn"``, scale construction matches the reference
    compressed-tensors memoryless-minmax MXFP4 path. When a Hessian is supplied,
    optional coordinate sweeps or GPTQ-style error feedback refine the individual
    codes at the selected scale.

    Scale selection priority:
        1. hessian: Hessian-weighted reconstruction error (AWQ/GPTQ-style).
        2. gamma:   gamma^2-weighted MSE (activation magnitude proxy).
        3. Neither: unweighted MSE.

    Args:
        tensor: [out_features, in_features]. Last dim divisible by BLOCK_SIZE.
        scale_percentile: Percentile anchoring the candidate range (100 = true amax).
        gamma: [in_features] per-channel activation magnitude.
        hessian: [num_blocks, block_size, block_size] block Hessian.
        method: ``"rtn"`` reference scaling or ``"mse"`` candidate search.
        mse_clip_depth: Candidate steps below the percentile-derived exponent.
        hessian_rounding_sweeps: Block-local Hessian coordinate-refinement passes.
        hessian_error_feedback: Apply block-local GPTQ-style error compensation.
        hessian_feedback_damp_percent: Average-Hessian-diagonal damping percentage.
        hessian_feedback_activation_order: Quantize high-activation columns first,
            then restore the original stored order.
        hessian_feedback_max_mse_ratio: Optional per-row/block ordinary-MSE trust
            region. Feedback codes are accepted only when their Hessian loss is
            lower and their MSE is at most this multiple of nearest rounding.
        feedback_selection_hessian: Independent block Hessian used only to select
            between nearest and feedback codes. Both the training and selection
            objectives must improve; it never participates in candidate creation.

    Returns:
        packed: [out, in//2] uint8 — two 4-bit codes per byte.
        scales: [out, in//BLOCK_SIZE] uint8 — e8m0 biased exponents.
    """
    if hessian is not None and gamma is not None:
        raise ValueError("hessian and gamma are mutually exclusive")
    if method not in ("rtn", "mse"):
        raise ValueError(f"Unsupported quantization method: {method}")
    if method == "rtn" and (gamma is not None or hessian is not None):
        raise ValueError("rtn does not use gamma or hessian weighting")
    if not 0.0 < scale_percentile <= 100.0:
        raise ValueError("scale_percentile must be in (0, 100]")
    if not isinstance(mse_clip_depth, int) or not 0 <= mse_clip_depth <= 8:
        raise ValueError("mse_clip_depth must be an integer in [0, 8]")
    if not isinstance(hessian_rounding_sweeps, int) or not 0 <= hessian_rounding_sweeps <= 4:
        raise ValueError("hessian_rounding_sweeps must be an integer in [0, 4]")
    if hessian_rounding_sweeps and (method != "mse" or hessian is None):
        raise ValueError("Hessian rounding requires method='mse' and a block Hessian")
    if not isinstance(hessian_error_feedback, bool):
        raise TypeError("hessian_error_feedback must be a boolean")
    if not isinstance(hessian_feedback_activation_order, bool):
        raise TypeError("hessian_feedback_activation_order must be a boolean")
    if not 0.0 < hessian_feedback_damp_percent <= 100.0:
        raise ValueError("hessian_feedback_damp_percent must be in (0, 100]")
    if hessian_feedback_max_mse_ratio is not None and not (
        1.0 <= hessian_feedback_max_mse_ratio <= 4.0
    ):
        raise ValueError("hessian_feedback_max_mse_ratio must be in [1, 4] or None")
    if feedback_selection_hessian is not None and not hessian_error_feedback:
        raise ValueError("feedback_selection_hessian requires Hessian error feedback")
    if hessian_error_feedback and (method != "mse" or hessian is None):
        raise ValueError("Hessian error feedback requires method='mse' and a block Hessian")
    if hessian_error_feedback and hessian_rounding_sweeps:
        raise ValueError("Hessian coordinate rounding and error feedback are mutually exclusive")
    if tensor.ndim != 2:
        raise ValueError(f"Expected a 2D weight tensor, got shape {tuple(tensor.shape)}")
    assert tensor.shape[-1] % BLOCK_SIZE == 0, (
        f"Last dim {tensor.shape[-1]} not divisible by BLOCK_SIZE={BLOCK_SIZE}"
    )

    t = tensor.to(torch.float32)
    *leading, K = t.shape
    num_blocks = K // BLOCK_SIZE

    t_blocked = t.reshape(*leading, num_blocks, BLOCK_SIZE)  # [..., B, 32]
    hessian_f32: torch.Tensor | None = None
    if hessian is not None:
        if hessian.shape != (num_blocks, BLOCK_SIZE, BLOCK_SIZE):
            raise ValueError(
                "hessian must have shape "
                f"{(num_blocks, BLOCK_SIZE, BLOCK_SIZE)}, got {tuple(hessian.shape)}"
            )
        if not torch.isfinite(hessian).all():
            raise ValueError("hessian contains NaN or infinity")
        hessian_f32 = hessian.to(device=t.device, dtype=torch.float32)
    selection_hessian_f32: torch.Tensor | None = None
    if feedback_selection_hessian is not None:
        if feedback_selection_hessian.shape != (num_blocks, BLOCK_SIZE, BLOCK_SIZE):
            raise ValueError(
                "feedback_selection_hessian must have shape "
                f"{(num_blocks, BLOCK_SIZE, BLOCK_SIZE)}, got "
                f"{tuple(feedback_selection_hessian.shape)}"
            )
        if not torch.isfinite(feedback_selection_hessian).all():
            raise ValueError("feedback_selection_hessian contains NaN or infinity")
        selection_hessian_f32 = feedback_selection_hessian.to(
            device=t.device,
            dtype=torch.float32,
        )

    # RTN always uses amax. Percentile anchoring is an mxstream MSE option.
    if method == "rtn" or scale_percentile >= 100.0:
        block_max = t_blocked.abs().amax(dim=-1)
    else:
        block_max = torch.quantile(t_blocked.abs(), scale_percentile / 100.0, dim=-1)

    if method == "rtn":
        raw_exp = _rtn_scale_exponent(block_max)
    else:
        # Suppress numerically empty blocks only in the quality-oriented path;
        # reference RTN preserves even subnormal-scale nonzero values.
        actual_max = t_blocked.abs().amax(dim=-1)
        near_zero = actual_max < 1e-6
        t_blocked = torch.where(near_zero.unsqueeze(-1), torch.zeros_like(t_blocked), t_blocked)
        block_max = block_max.clamp(min=1e-6)
        # Include controlled clipping candidates plus the true no-clipping
        # exponent.  Evaluating the safe exponent in the same objective avoids
        # an after-the-fact overflow rule overriding the calibrated decision.
        exp_floor = torch.floor(torch.log2(block_max / 6.0))
        offsets = torch.arange(
            -mse_clip_depth,
            2,
            dtype=exp_floor.dtype,
            device=exp_floor.device,
        )
        local_candidates = exp_floor.unsqueeze(-1) + offsets
        safe_exp = torch.ceil(torch.log2(actual_max.clamp(min=1e-12) / 6.0)).unsqueeze(-1)
        candidates = torch.cat([local_candidates, safe_exp], dim=-1).clamp(-127, 127)

        t_cand = t_blocked.unsqueeze(-2)  # [out, B, 1, 32]
        scales = torch.pow(2.0, candidates).unsqueeze(-1)
        scaled = (t_cand / scales).clamp(-6.0, 6.0)
        dequant_orig = _round_to_mxfp4(scaled) * scales
        dW = dequant_orig - t_cand

        if hessian_f32 is not None:
            weighted = torch.einsum("obcs,bsd->obcd", dW, hessian_f32)
            err = (weighted * dW).sum(dim=-1)
        elif gamma is not None:
            if gamma.shape != (K,):
                raise ValueError(f"gamma must have shape {(K,)}, got {tuple(gamma.shape)}")
            if not torch.isfinite(gamma).all() or (gamma < 0).any():
                raise ValueError("gamma must contain finite, non-negative magnitudes")
            gamma_b = gamma.to(device=t.device, dtype=torch.float32).reshape(
                num_blocks, BLOCK_SIZE
            )
            err = (dW**2 * (gamma_b**2)[None, :, None, :]).mean(dim=-1)
        else:
            err = (dW**2).mean(dim=-1)

        best_idx = err.argmin(dim=-1, keepdim=True)
        raw_exp = candidates.gather(-1, best_idx).squeeze(-1)

    # E8M0 reserves 255; clamp the exponent before both storage and quantization
    # so the encoded scale and the scale used for rounding can never disagree.
    raw_exp = raw_exp.clamp(-127, 127)
    biased_exp = (raw_exp + 127).to(torch.uint8)
    final_scale = torch.pow(2.0, raw_exp)

    scaled_final = (t_blocked / final_scale.unsqueeze(-1)).clamp(-6.0, 6.0)
    codes = _nearest_mxfp4_codes(scaled_final)
    if hessian_error_feedback:
        assert hessian_f32 is not None
        feedback_codes = _codes_with_hessian_error_feedback(
            t_blocked,
            final_scale,
            hessian_f32,
            damp_percent=hessian_feedback_damp_percent,
            activation_order=hessian_feedback_activation_order,
        )
        if hessian_feedback_max_mse_ratio is None and selection_hessian_f32 is None:
            codes = feedback_codes
        else:
            values = _SIGNED_VALUES.to(device=t.device)
            nearest_error = values[codes.long()] * final_scale.unsqueeze(-1) - t_blocked
            feedback_error = (
                values[feedback_codes.long()] * final_scale.unsqueeze(-1) - t_blocked
            )
            nearest_hessian_loss = (
                torch.einsum("obs,bsd->obd", nearest_error, hessian_f32) * nearest_error
            ).sum(dim=-1)
            feedback_hessian_loss = (
                torch.einsum("obs,bsd->obd", feedback_error, hessian_f32) * feedback_error
            ).sum(dim=-1)
            accept_feedback = feedback_hessian_loss < nearest_hessian_loss
            if selection_hessian_f32 is not None:
                nearest_selection_loss = (
                    torch.einsum(
                        "obs,bsd->obd",
                        nearest_error,
                        selection_hessian_f32,
                    )
                    * nearest_error
                ).sum(dim=-1)
                feedback_selection_loss = (
                    torch.einsum(
                        "obs,bsd->obd",
                        feedback_error,
                        selection_hessian_f32,
                    )
                    * feedback_error
                ).sum(dim=-1)
                accept_feedback &= feedback_selection_loss < nearest_selection_loss
            if hessian_feedback_max_mse_ratio is not None:
                nearest_mse = nearest_error.square().mean(dim=-1)
                feedback_mse = feedback_error.square().mean(dim=-1)
                accept_feedback &= (
                    feedback_mse <= nearest_mse * hessian_feedback_max_mse_ratio
                )
            codes = torch.where(accept_feedback.unsqueeze(-1), feedback_codes, codes)
    elif hessian_rounding_sweeps:
        assert hessian_f32 is not None
        codes = _refine_codes_with_hessian(
            t_blocked,
            final_scale,
            codes,
            hessian_f32,
            hessian_rounding_sweeps,
        )

    codes_flat = codes.reshape(*leading, K)
    packed = codes_flat[..., 0::2] | (codes_flat[..., 1::2] << 4)

    return packed.to(torch.uint8), biased_exp


def dequant_mxfp4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    shape: tuple[int, ...],
) -> torch.Tensor:
    """Dequantize MXFP4 packed weights back to float32.

    Args:
        packed: [..., in//2] uint8 — two 4-bit codes per byte.
        scales: [..., in//BLOCK_SIZE] uint8 — e8m0 biased exponents.
        shape: Original weight shape (e.g., [out, in]).

    Returns:
        Reconstructed float32 tensor with the given shape.
    """
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    codes = torch.stack([lo, hi], dim=-1).reshape(shape)
    sign = ((codes >> 3) & 1).float() * -2 + 1
    mag_idx = (codes & 0x07).long().reshape(-1)
    magnitude = _POS_VALUES.to(packed.device)[mag_idx].reshape(codes.shape)
    raw_exp = scales.float() - 127
    scale_vals = torch.pow(2.0, raw_exp)
    scale_expanded = scale_vals.repeat_interleave(BLOCK_SIZE, dim=-1)
    return sign * magnitude * scale_expanded
