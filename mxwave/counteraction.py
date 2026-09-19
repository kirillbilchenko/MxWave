"""Architecture-independent residual-error counteraction measurements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

__all__ = ["CounteractionMetrics", "measure_counteraction"]


@dataclass(frozen=True)
class CounteractionMetrics:
    """Energy decomposition across one residual block.

    Let ``e`` be inherited hidden error and ``d`` the current block's update
    error. The resulting error is ``e + d`` and therefore
    ``||e + d||² = ||e||² + ||d||² + 2<e, d>``. A negative interaction is
    counteraction; a positive interaction amplifies inherited error.
    """

    inherited_error_energy: float
    update_error_energy: float
    interaction_energy: float
    resulting_error_energy: float
    reference_output_energy: float
    inherited_hidden_nmse: float
    update_error_nmse: float
    interaction_nmse: float
    resulting_hidden_nmse: float
    error_growth_nmse: float
    counteraction_fraction: float
    recurrence_relative_residual: float

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable metric record."""
        return {
            "inherited_error_energy": self.inherited_error_energy,
            "update_error_energy": self.update_error_energy,
            "interaction_energy": self.interaction_energy,
            "resulting_error_energy": self.resulting_error_energy,
            "reference_output_energy": self.reference_output_energy,
            "inherited_hidden_nmse": self.inherited_hidden_nmse,
            "update_error_nmse": self.update_error_nmse,
            "interaction_nmse": self.interaction_nmse,
            "resulting_hidden_nmse": self.resulting_hidden_nmse,
            "error_growth_nmse": self.error_growth_nmse,
            "counteraction_fraction": self.counteraction_fraction,
            "recurrence_relative_residual": self.recurrence_relative_residual,
        }


def _energy(value: torch.Tensor) -> torch.Tensor:
    return value.to(torch.float64).square().mean()


def measure_counteraction(
    reference_input: torch.Tensor,
    execution_input: torch.Tensor,
    reference_output: torch.Tensor,
    candidate_output: torch.Tensor,
    *,
    epsilon: float = 1e-30,
) -> CounteractionMetrics:
    """Decompose inherited and newly introduced hidden error for one block.

    Inputs may represent any residual architecture. ``execution_input`` is the
    actual quantized-prefix state, while ``candidate_output`` is the output of
    the candidate block evaluated on that state. The function performs no
    model-specific operations and does not assume a particular quantization
    format.
    """
    tensors = (reference_input, execution_input, reference_output, candidate_output)
    shapes = {tuple(value.shape) for value in tensors}
    if len(shapes) != 1:
        raise ValueError(f"Counteraction tensors must share one shape, got {sorted(shapes)}")
    if reference_input.numel() == 0:
        raise ValueError("Counteraction tensors must be non-empty")
    if any(not torch.is_floating_point(value) for value in tensors):
        raise TypeError("Counteraction tensors must be floating point")
    if any(not bool(torch.isfinite(value).all().item()) for value in tensors):
        raise ValueError("Counteraction tensors contain NaN or infinity")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")

    reference_input_f64 = reference_input.detach().to(torch.float64)
    execution_input_f64 = execution_input.detach().to(torch.float64)
    reference_output_f64 = reference_output.detach().to(torch.float64)
    candidate_output_f64 = candidate_output.detach().to(torch.float64)
    inherited = execution_input_f64 - reference_input_f64
    reference_update = reference_output_f64 - reference_input_f64
    candidate_update = candidate_output_f64 - execution_input_f64
    update_error = candidate_update - reference_update
    resulting = candidate_output_f64 - reference_output_f64

    inherited_energy = _energy(inherited)
    update_energy = _energy(update_error)
    interaction = 2.0 * (inherited * update_error).mean()
    resulting_energy = _energy(resulting)
    reference_energy = _energy(reference_output_f64)
    recurrence_rhs = inherited_energy + update_energy + interaction
    recurrence_scale = torch.maximum(
        resulting_energy.abs(),
        inherited_energy.abs() + update_energy.abs() + interaction.abs(),
    ).clamp_min(epsilon)
    recurrence_residual = (resulting_energy - recurrence_rhs).abs() / recurrence_scale
    normalization = reference_energy.clamp_min(epsilon)
    update_denominator = update_energy.clamp_min(epsilon)

    return CounteractionMetrics(
        inherited_error_energy=float(inherited_energy.item()),
        update_error_energy=float(update_energy.item()),
        interaction_energy=float(interaction.item()),
        resulting_error_energy=float(resulting_energy.item()),
        reference_output_energy=float(reference_energy.item()),
        inherited_hidden_nmse=float((inherited_energy / normalization).item()),
        update_error_nmse=float((update_energy / normalization).item()),
        interaction_nmse=float((interaction / normalization).item()),
        resulting_hidden_nmse=float((resulting_energy / normalization).item()),
        error_growth_nmse=float(((resulting_energy - inherited_energy) / normalization).item()),
        counteraction_fraction=float((-interaction / update_denominator).item()),
        recurrence_relative_residual=float(recurrence_residual.item()),
    )
