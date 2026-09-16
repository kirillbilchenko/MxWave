"""Verification-first contract: the output is recomputed and validated, never
trusted-by-inheritance from the source.

Provides:
  - ``sqnr`` — signal-to-quantization-noise ratio of a quantized tensor
  - ``checkpoint_sqnr`` — per-tensor SQNR across a set of (packed, scale, orig)
  - ``verify_config_coverage`` — diff quantization targets/ignore against the
    real module list so no Linear is silently left unquantized.
"""

from __future__ import annotations

import torch

__all__ = [
    "block_hessian_weighted_sqnr",
    "channel_weighted_sqnr",
    "checkpoint_sqnr",
    "sqnr",
    "verify_config_coverage",
]


def block_hessian_weighted_sqnr(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    hessian: torch.Tensor,
) -> float:
    """Return SQNR in the block-quadratic activation reconstruction objective."""
    if original.shape != reconstructed.shape or original.ndim != 2:
        raise ValueError("original and reconstructed tensors must be same-shaped matrices")
    width = original.shape[1]
    if width % 32 != 0 or hessian.shape != (width // 32, 32, 32):
        raise ValueError(
            "hessian must have shape "
            f"{(width // 32, 32, 32)} for input width {width}, got {tuple(hessian.shape)}"
        )
    orig = original.float().reshape(original.shape[0], width // 32, 32)
    noise_values = (original - reconstructed).float().reshape(
        original.shape[0], width // 32, 32
    )
    matrix = hessian.to(device=orig.device, dtype=torch.float32)
    signal = torch.einsum("rbi,bij,rbj->", orig, matrix, orig).clamp(min=1e-12)
    noise = torch.einsum("rbi,bij,rbj->", noise_values, matrix, noise_values).clamp(
        min=1e-12
    )
    return float(10.0 * torch.log10(signal / noise))


def channel_weighted_sqnr(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    channel_weights: torch.Tensor,
) -> float:
    """Return SQNR after weighting each input channel by ``channel_weights²``.

    This reports the objective used by gamma-proxy scale selection. It complements
    ordinary SQNR; it must not be presented as an activation-calibrated model metric.
    """
    if original.shape != reconstructed.shape:
        raise ValueError("original and reconstructed tensors must have the same shape")
    if original.ndim < 1 or channel_weights.shape != (original.shape[-1],):
        raise ValueError(
            "channel_weights must match the last tensor dimension, got "
            f"{tuple(channel_weights.shape)} for {tuple(original.shape)}"
        )
    orig = original.float()
    recon = reconstructed.float()
    weights = channel_weights.to(orig.device).float().square()
    weights = weights.reshape(*([1] * (orig.ndim - 1)), orig.shape[-1])
    signal = (orig.square() * weights).sum().clamp(min=1e-12)
    noise = ((orig - recon).square() * weights).sum().clamp(min=1e-12)
    return float(10.0 * torch.log10(signal / noise))


def sqnr(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """Signal-to-quantization-noise ratio (dB) between two tensors.

    SQNR = 10 * log10( ||x||^2 / ||x - x_hat||^2 ), with the noise floor
    clamped to avoid div-by-zero.
    """
    orig = original.float().reshape(-1)
    recon = reconstructed.float().reshape(-1)
    signal = (orig**2).sum().clamp(min=1e-12)
    noise = ((orig - recon) ** 2).sum().clamp(min=1e-12)
    return float(10.0 * torch.log10(signal / noise))


def checkpoint_sqnr(
    pairs: list[tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, float]:
    """Per-tensor SQNR for a list of (original, reconstructed) pairs.

    Returns a dict keyed by index with the SQNR in dB. A catastrophically low
    value (e.g. < 5 dB) signals a scale/format bug.
    """
    return {str(i): sqnr(o, r) for i, (o, r) in enumerate(pairs)}


def verify_config_coverage(
    targets: list[str],
    ignore: list[str],
    real_modules: list[str],
) -> list[str]:
    """Return modules that are neither targeted nor ignored (coverage gaps).

    A Linear that is neither targeted nor ignored loads unquantized and can
    silently produce wrong output. Fail on any non-empty result.
    """
    import re

    def matches(pattern: str, module: str) -> bool:
        if pattern.startswith("re:"):
            return re.match(pattern[3:], module) is not None
        return pattern in module

    uncovered: list[str] = []
    for module in real_modules:
        if any(matches(t, module) for t in targets):
            continue
        if any(matches(i, module) for i in ignore):
            continue
        uncovered.append(module)
    return uncovered
