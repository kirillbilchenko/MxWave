"""Bounded materialization of legal MXFP4 weights for research probes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

from .core import BLOCK_SIZE, QuantizationMethod, dequant_mxfp4, quantize_mxfp4

CandidateWeighting = Literal[
    "none",
    "channel",
    "diagonal-hessian",
    "block-hessian",
]

__all__ = [
    "CandidateWeighting",
    "Mxfp4CandidateSpec",
    "diagonal_hessian_channel_weights",
    "materialize_mxfp4_candidate",
    "standard_mxfp4_probe_candidates",
]


@dataclass(frozen=True)
class Mxfp4CandidateSpec:
    """One reproducible MXFP4 quantize/dequant candidate for replay."""

    name: str
    method: QuantizationMethod = "mse"
    weighting: CandidateWeighting = "none"
    scale_percentile: float = 99.5
    mse_clip_depth: int = 1

    def __post_init__(self) -> None:
        """Validate candidate settings before any tensor allocation."""
        if not self.name:
            raise ValueError("MXFP4 candidate name must be non-empty")
        if self.method not in ("rtn", "mse"):
            raise ValueError(f"Unsupported MXFP4 candidate method: {self.method}")
        if self.weighting not in (
            "none",
            "channel",
            "diagonal-hessian",
            "block-hessian",
        ):
            raise ValueError(f"Unsupported MXFP4 candidate weighting: {self.weighting}")
        if self.method == "rtn" and self.weighting != "none":
            raise ValueError("RTN candidates cannot use activation weighting")
        if not math.isfinite(self.scale_percentile) or not 0.0 < self.scale_percentile <= 100.0:
            raise ValueError("MXFP4 candidate scale_percentile must be in (0, 100]")
        if not isinstance(self.mse_clip_depth, int) or not 0 <= self.mse_clip_depth <= 8:
            raise ValueError("MXFP4 candidate mse_clip_depth must be an integer in [0, 8]")


def standard_mxfp4_probe_candidates(
    *,
    scale_percentile: float = 99.5,
    mse_clip_depth: int = 4,
) -> tuple[Mxfp4CandidateSpec, ...]:
    """Return the fixed RTN, MSE, diagonal, and full-Hessian controls."""
    return (
        Mxfp4CandidateSpec("rtn", method="rtn"),
        Mxfp4CandidateSpec(
            "unweighted-mse",
            scale_percentile=scale_percentile,
            mse_clip_depth=mse_clip_depth,
        ),
        Mxfp4CandidateSpec(
            "diagonal-hessian",
            weighting="diagonal-hessian",
            scale_percentile=scale_percentile,
            mse_clip_depth=mse_clip_depth,
        ),
        Mxfp4CandidateSpec(
            "block-hessian",
            weighting="block-hessian",
            scale_percentile=scale_percentile,
            mse_clip_depth=mse_clip_depth,
        ),
    )


def diagonal_hessian_channel_weights(block_hessian: torch.Tensor) -> torch.Tensor:
    """Return channel weights whose squares equal a block-Hessian diagonal."""
    if block_hessian.ndim != 3 or block_hessian.shape[-2:] != (BLOCK_SIZE, BLOCK_SIZE):
        raise ValueError(
            "MXFP4 block Hessian must have shape (num_blocks, 32, 32), got "
            f"{tuple(block_hessian.shape)}"
        )
    diagonal = block_hessian.diagonal(dim1=-2, dim2=-1)
    if not bool(torch.isfinite(diagonal).all().item()) or bool((diagonal < 0).any().item()):
        raise ValueError("MXFP4 block-Hessian diagonal must be finite and non-negative")
    return diagonal.sqrt().reshape(-1).contiguous()


def materialize_mxfp4_candidate(
    weight: torch.Tensor,
    spec: Mxfp4CandidateSpec,
    *,
    channel_weights: torch.Tensor | None = None,
    block_hessian: torch.Tensor | None = None,
    row_chunk_size: int = 1024,
) -> torch.Tensor:
    """Return one dequantized MXFP4 candidate with bounded temporary memory."""
    if weight.ndim != 2:
        raise ValueError(f"MXFP4 candidate weight must be 2D, got {tuple(weight.shape)}")
    if weight.shape[-1] % BLOCK_SIZE != 0:
        raise ValueError(
            f"MXFP4 candidate input dimension {weight.shape[-1]} is not divisible by {BLOCK_SIZE}"
        )
    if not torch.is_floating_point(weight):
        raise TypeError("MXFP4 candidate source weight must be floating point")
    if not bool(torch.isfinite(weight).all().item()):
        raise ValueError("MXFP4 candidate source weight contains NaN or infinity")
    if not isinstance(row_chunk_size, int) or row_chunk_size <= 0:
        raise ValueError("MXFP4 candidate row_chunk_size must be a positive integer")

    gamma: torch.Tensor | None = None
    hessian: torch.Tensor | None = None
    if spec.weighting == "channel":
        if channel_weights is None:
            raise ValueError(f"MXFP4 candidate {spec.name!r} requires channel weights")
        gamma = channel_weights.to(device=weight.device)
    elif spec.weighting == "diagonal-hessian":
        if block_hessian is None:
            raise ValueError(f"MXFP4 candidate {spec.name!r} requires a block Hessian")
        gamma = diagonal_hessian_channel_weights(block_hessian).to(device=weight.device)
    elif spec.weighting == "block-hessian":
        if block_hessian is None:
            raise ValueError(f"MXFP4 candidate {spec.name!r} requires a block Hessian")
        hessian = block_hessian.to(device=weight.device)

    candidate = torch.empty_like(weight, memory_format=torch.contiguous_format)
    with torch.inference_mode():
        for start in range(0, weight.shape[0], row_chunk_size):
            stop = min(start + row_chunk_size, weight.shape[0])
            source = weight[start:stop]
            packed, scales = quantize_mxfp4(
                source,
                scale_percentile=spec.scale_percentile,
                gamma=gamma,
                hessian=hessian,
                method=spec.method,
                mse_clip_depth=spec.mse_clip_depth,
            )
            reconstructed = dequant_mxfp4(
                packed,
                scales,
                (stop - start, weight.shape[1]),
            )
            candidate[start:stop].copy_(reconstructed)
            del packed, scales, reconstructed
    return candidate
