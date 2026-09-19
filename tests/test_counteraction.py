"""Tests for architecture-independent residual counteraction metrics."""

from __future__ import annotations

import math

import pytest
import torch

from mxwave.counteraction import measure_counteraction


def test_measure_counteraction_recovers_exact_cancellation() -> None:
    reference_input = torch.zeros(2, 3, dtype=torch.float32)
    execution_input = torch.ones(2, 3, dtype=torch.float32)
    reference_output = torch.full((2, 3), 2.0, dtype=torch.float32)
    candidate_output = reference_output.clone()

    metrics = measure_counteraction(
        reference_input,
        execution_input,
        reference_output,
        candidate_output,
    )

    assert metrics.inherited_error_energy == pytest.approx(1.0)
    assert metrics.update_error_energy == pytest.approx(1.0)
    assert metrics.interaction_energy == pytest.approx(-2.0)
    assert metrics.resulting_error_energy == pytest.approx(0.0)
    assert metrics.counteraction_fraction == pytest.approx(2.0)
    assert metrics.recurrence_relative_residual < 1e-12


def test_measure_counteraction_detects_amplification() -> None:
    reference_input = torch.zeros(4, dtype=torch.float64)
    execution_input = torch.ones(4, dtype=torch.float64)
    reference_output = torch.ones(4, dtype=torch.float64)
    candidate_output = torch.full((4,), 3.0, dtype=torch.float64)

    metrics = measure_counteraction(
        reference_input,
        execution_input,
        reference_output,
        candidate_output,
    )

    assert metrics.interaction_energy > 0.0
    assert metrics.error_growth_nmse > 0.0
    assert metrics.resulting_hidden_nmse == pytest.approx(4.0)
    assert metrics.recurrence_relative_residual < 1e-12


def test_measure_counteraction_is_finite_when_update_error_is_zero() -> None:
    reference_input = torch.randn(2, 4)
    execution_input = reference_input + 0.1
    reference_output = reference_input + 0.5
    candidate_output = execution_input + 0.5

    metrics = measure_counteraction(
        reference_input,
        execution_input,
        reference_output,
        candidate_output,
    )

    assert metrics.update_error_energy < 1e-12
    assert math.isfinite(metrics.counteraction_fraction)
    assert metrics.resulting_error_energy == pytest.approx(metrics.inherited_error_energy)


def test_measure_counteraction_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="share one shape"):
        measure_counteraction(
            torch.zeros(2),
            torch.zeros(3),
            torch.zeros(2),
            torch.zeros(2),
        )
