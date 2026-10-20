"""Diagnostic MXFP8 weight rounding; no checkpoint emission or inference kernel changes.

The declared recipe uses E4M3 RNE values and an upward-rounded power-of-two
E8M0 scale per 32 consecutive input columns. It follows the public format and
the scale-selection description in NVIDIA Transformer Engine's MXFP8 docs.
This independently written implementation does not copy another quantizer.
"""

from __future__ import annotations

import numpy as np
import torch


def roundtrip(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return decoded FP32 values, E4M3 codes and biased E8M0 scale bytes."""
    if weight.ndim != 2 or weight.shape[1] % 32 or not weight.is_floating_point():
        raise ValueError("Require a floating-point matrix with 32-aligned input columns")
    values = weight.float().reshape(weight.shape[0], -1, 32)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Require finite source weights")
    maxima = values.abs().amax(dim=-1, keepdim=True)
    exponent = torch.ceil(torch.log2((maxima / 448.0).clamp_min(2.0**-127)))
    exponent = torch.where(maxima == 0, torch.zeros_like(exponent), exponent)
    exponent = exponent.clamp(-127, 127).to(torch.int32)
    scaled = torch.ldexp(values, -exponent)
    if bool((scaled.abs() > 448).any()):
        raise ValueError("Declared upward scale policy must not clip finite BF16 weights")
    codes = scaled.to(torch.float8_e4m3fn)
    decoded = torch.ldexp(codes.float(), exponent).reshape(weight.shape)
    if not bool(torch.isfinite(decoded).all()):
        raise ValueError("MXFP8 roundtrip produced non-finite values")
    return decoded, codes.reshape(weight.shape).view(torch.uint8), (exponent + 127).squeeze(-1).byte()


def numpy_reference(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Independently choose nearest E4M3 values using the analytic finite codebook."""
    if weight.ndim != 2 or weight.shape[1] % 32 or not np.isfinite(weight).all():
        raise ValueError("Require a finite, 32-aligned reference matrix")
    values = weight.astype(np.float64).reshape(weight.shape[0], -1, 32)
    maxima = np.max(np.abs(values), axis=-1, keepdims=True)
    exponent = np.ceil(np.log2(np.maximum(maxima / 448.0, 2.0**-127)))
    exponent = np.where(maxima == 0, 0, exponent).clip(-127, 127).astype(np.int32)
    scaled = np.ldexp(values, -exponent)
    positive = np.arange(127, dtype=np.int32)
    e, m = positive >> 3, positive & 7
    levels = np.where(e == 0, m * 2.0**-9, (1 + m / 8) * np.exp2(e - 7))
    magnitude = np.abs(scaled)
    if np.any(magnitude > 448):
        raise ValueError("Scale policy overflow")
    hi = np.searchsorted(levels, magnitude).clip(0, 126)
    lo = np.maximum(hi - 1, 0)
    left, right = magnitude - levels[lo], levels[hi] - magnitude
    selected = np.where((left < right) | ((left == right) & ((lo & 1) == 0)), lo, hi)
    codes = (selected | (np.signbit(scaled).astype(np.int32) << 7)).astype(np.uint8)
    decoded = np.ldexp(np.copysign(levels[selected], scaled), exponent)
    return (
        decoded.reshape(weight.shape).astype(np.float32),
        codes.reshape(weight.shape),
        (exponent + 127).squeeze(-1).astype(np.uint8),
    )
