"""Tests for the frozen layer-62 confirmation gate."""

from __future__ import annotations

from typing import Any

from mxwave.layer_confirmation_eval import evaluate_layer_confirmation


def _report(candidate_delta: float = 0.01) -> dict[str, Any]:
    baseline_values = [0.2 + index * 0.001 for index in range(16)]
    candidate_values = [value - candidate_delta for value in baseline_values]
    return {
        "format": "mxwave-counteraction-probe-v1",
        "status": "complete",
        "suffix_sensitivity": "none",
        "requested_layers": [62],
        "execution": {"attention_implementation": "eager", "dtype": "bfloat16"},
        "heldout": {
            "sequence_offset": 120,
            "num_sequences": 16,
            "sequence_length": 512,
            "logit_positions_per_sequence": 8,
            "token_ids_sha256": (
                "49ac6fcdd0f473d37d5258cd387c8e7a60d3ef5934810f66d901d2e9ed3aee81"
            ),
        },
        "layers": [
            {
                "layer_index": 62,
                "baseline": {"sample_teacher_kl": baseline_values},
                "candidates": [
                    {
                        "candidate": "unweighted-mse",
                        "sample_teacher_kl": candidate_values,
                        "baseline_weight_nmse": 0.1,
                        "max_recurrence_relative_residual": 1e-15,
                    },
                    {
                        "candidate": "block-hessian",
                        "sample_teacher_kl": baseline_values,
                        "baseline_weight_nmse": 0.0,
                        "max_recurrence_relative_residual": 1e-15,
                    },
                ],
            }
        ],
    }


def test_confirmation_passes_consistent_paired_improvement() -> None:
    result = evaluate_layer_confirmation(_report())

    assert result["passed"] is True
    assert result["paired_wins"] == 16
    assert result["paired_bootstrap_95_interval"][0] > 0.0


def test_confirmation_rejects_regression() -> None:
    result = evaluate_layer_confirmation(_report(candidate_delta=-0.01))

    assert result["passed"] is False
    assert result["gates"]["candidate_mean_teacher_kl_lower"] is False
