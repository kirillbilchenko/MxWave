"""Routing-weighted RMS input statistics for fused mixture-of-experts layers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def soften_routed_rms(
    rms: torch.Tensor,
    *,
    strength: float = 0.25,
    observations: int,
    pseudo_count: float = 128.0,
    max_variance_ratio: float = 4.0,
) -> torch.Tensor:
    """Blend normalized RMS variances toward uniform MSE with route-count shrinkage.

    A target's activation strength is ``strength * n / (n + pseudo_count)``.
    Variance ratios are clipped to ``[1 / max_variance_ratio, max_variance_ratio]``
    before blending. Returned gamma is float32 on the input device; squaring it
    gives the blended objective up to an objective-preserving scalar factor.
    Route-count shrinkage is a heuristic, not an effective sample-size estimate.
    """
    if (
        not math.isfinite(strength)
        or not 0 <= strength <= 1
        or not math.isfinite(pseudo_count)
        or pseudo_count < 0
        or not math.isfinite(max_variance_ratio)
        or max_variance_ratio < 1
        or observations < 0
    ):
        raise ValueError("Invalid activation strength, route count, or variance bound")
    if (
        rms.ndim != 1
        or not rms.numel()
        or not rms.is_floating_point()
        or not bool(torch.isfinite(rms).all())
        or bool((rms < 0).any())
        or not bool((rms > 0).any())
    ):
        raise ValueError("RMS must be a finite nonnegative channel vector with positive energy")
    values = (rms / rms.amax()).float()
    variance = values.square()
    ratio = (variance / variance.mean()).clamp(1 / max_variance_ratio, max_variance_ratio)
    confidence = observations / (observations + pseudo_count) if observations else 0.0
    alpha = strength * confidence
    gamma = torch.sqrt((1 - alpha) + alpha * ratio)
    return (gamma / gamma.amax()).contiguous()


@dataclass
class RoutedLayerMoments:
    """Bounded per-expert statistics, with no retained token activation history."""

    input_squares: torch.Tensor
    down_squares: torch.Tensor
    route_mass: torch.Tensor
    observations: torch.Tensor
    down_observations: torch.Tensor

    @classmethod
    def create(
        cls,
        experts: int,
        hidden: int,
        intermediate: int,
        device: torch.device,
    ) -> RoutedLayerMoments:
        """Allocate device-local accumulators for one complete unsharded MoE layer."""
        if min(experts, hidden, intermediate) <= 0:
            raise ValueError("Expert count and channel widths must be positive")
        return cls(
            torch.zeros((experts, hidden), dtype=torch.float32, device=device),
            torch.zeros((experts, intermediate), dtype=torch.float32, device=device),
            torch.zeros(experts, dtype=torch.float32, device=device),
            torch.zeros(experts, dtype=torch.int64, device=device),
            torch.zeros(experts, dtype=torch.int64, device=device),
        )

    def _routes(
        self,
        ids: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            ids.ndim != 2
            or weights.shape != ids.shape
            or ids.dtype not in (torch.int32, torch.int64)
            or ids.device != self.input_squares.device
        ):
            raise ValueError("Expected matching device-local [tokens, top_k] route IDs/weights")
        if (
            ids.numel() == 0
            or bool((ids < 0).any())
            or bool((ids >= len(self.route_mass)).any())
            or not bool(torch.isfinite(weights).all())
            or bool((weights < 0).any())
        ):
            raise ValueError("Routing IDs or weights are empty, invalid, or non-finite")
        return ids.reshape(-1).long(), weights.detach().float().reshape(-1).square()

    @torch.no_grad()
    def update_inputs(
        self,
        inputs: torch.Tensor,
        ids: torch.Tensor,
        weights: torch.Tensor,
    ) -> None:
        """Accumulate gate/up inputs for exactly the expert routes selected at runtime."""
        routed_ids, mass = self._routes(ids, weights)
        if inputs.shape != (ids.shape[0], self.input_squares.shape[1]):
            raise ValueError("Gate/up activation shape does not match the actual token routes")
        values = inputs.detach().float().square().repeat_interleave(ids.shape[1], dim=0)
        self.input_squares.index_add_(0, routed_ids, values * mass[:, None])
        self.route_mass.index_add_(0, routed_ids, mass)
        self.observations.add_(torch.bincount(routed_ids, minlength=len(self.route_mass)))

    @torch.no_grad()
    def update_down(
        self,
        inputs: torch.Tensor,
        ids: torch.Tensor,
        weights: torch.Tensor,
    ) -> None:
        """Accumulate token-major down-projection inputs, weighted by router weights squared."""
        routed_ids, mass = self._routes(ids, weights)
        if inputs.shape != (ids.numel(), self.down_squares.shape[1]):
            raise ValueError("Down inputs must have one row per token-major expert route")
        self.down_squares.index_add_(
            0, routed_ids, inputs.detach().float().square() * mass[:, None]
        )
        self.down_observations.add_(torch.bincount(routed_ids, minlength=len(self.route_mass)))

    def finalize(self, layer: int) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
        """Return observed, nonzero RMS targets; unseen targets remain explicitly uncalibrated."""
        if not torch.equal(self.observations, self.down_observations):
            raise ValueError("Gate/up and down captures do not cover the same routed observations")
        if not bool(torch.isfinite(self.input_squares).all()) or not bool(
            torch.isfinite(self.down_squares).all()
        ):
            raise ValueError("Captured activation moments contain NaN or infinity")
        statistics: dict[str, torch.Tensor] = {}
        retained: list[str] = []
        masses = self.route_mass.cpu()
        counts = self.observations.cpu()
        gate = self.input_squares.cpu()
        down = self.down_squares.cpu()
        for expert in range(len(masses)):
            for projection, values in (("gate_proj", gate), ("up_proj", gate), ("down_proj", down)):
                name = f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"
                mass = float(masses[expert])
                if mass <= 0 or not bool((values[expert] > 0).any()):
                    retained.append(name)
                    continue
                gamma = torch.sqrt(values[expert] / mass)
                # A target-wide normalization preserves the scale-selection objective
                # while preventing very small routed weights from causing underflow.
                gamma = (gamma / gamma.amax()).float().contiguous()
                statistics[name] = gamma
        return statistics, {
            "layer": layer,
            "routed_observations": counts.tolist(),
            "routing_weight_squared_mass": masses.tolist(),
            "calibrated_targets": len(statistics),
            "retained_unweighted_targets": retained,
        }


def dequantize_fp8_activation_blocks(
    quantized: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """Decode float32-scaled FP8 activation groups; never reinterpret E8M0 scale bytes."""
    if (
        quantized.ndim != 2
        or quantized.dtype not in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
        or group_size <= 0
        or quantized.shape[1] % group_size
        or scales.shape != (quantized.shape[0], quantized.shape[1] // group_size)
        or scales.dtype != torch.float32
    ):
        raise ValueError(
            "Expected FP8 activations and same-row float32 block dequantization scales"
        )
    if not bool(torch.isfinite(scales).all()) or bool((scales <= 0).any()):
        raise ValueError("Activation dequantization scales must be finite and positive")
    result = quantized.float() * scales.repeat_interleave(group_size, dim=-1)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("Decoded FP8 activations contain NaN or infinity")
    return result
