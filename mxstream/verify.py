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
    "checkpoint_sqnr",
    "sqnr",
    "verify_config_coverage",
]


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
