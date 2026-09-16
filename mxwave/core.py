"""MXFP4 E2M1 quantization core.

MXFP4 (OCP MX spec): 4 bits per value, E2M1, 8 positive magnitudes
{0, 0.5, 1, 1.5, 2, 3, 4, 6}. Every 32 consecutive input channels share one
e8m0 scale (2**k, integer k). Two 4-bit codes are packed per byte.

Scale selection is MSE-optimal over an explicit candidate set, optionally
weighted by per-channel activation statistics or a block Hessian. This is a
clean-room implementation against the public MX spec and published
quantization methods.
"""

from __future__ import annotations

from typing import Literal

import torch

BLOCK_SIZE = 32
QuantizationMethod = Literal["rtn", "mse"]

# MXFP4 E2M1 representable positive values and their rounding boundaries.
_POS_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
_BOUNDARIES = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32)


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
    MxWave independent from that package while producing the same scale.
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


def quantize_mxfp4(
    tensor: torch.Tensor,
    scale_percentile: float = 99.5,
    gamma: torch.Tensor | None = None,
    hessian: torch.Tensor | None = None,
    method: QuantizationMethod = "mse",
    mse_clip_depth: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a float tensor to MXFP4 with MSE-optimal scale selection.

    With ``method="mse"``, candidate exponents from ``floor-mse_clip_depth``
    through ``floor+1`` are evaluated for every block. The smallest exponent
    that cannot clip the true maximum is always included as a control. With
    ``method="rtn"``, scale construction matches the reference
    compressed-tensors memoryless-minmax MXFP4 path.

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
    # RTN always uses amax. Percentile anchoring is an MxWave MSE option.
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
