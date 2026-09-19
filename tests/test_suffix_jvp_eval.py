"""Tests for frozen two-split suffix-JVP gates."""

from __future__ import annotations

from typing import Any

from mxwave.suffix_jvp_eval import evaluate_suffix_jvp_reports


def _report(token_hash: str) -> dict[str, Any]:
    names = ("rtn", "unweighted-mse", "diagonal-hessian", "block-hessian")
    primary = dict(zip(names, (0.1, 0.2, 0.3, 0.4), strict=True))
    control = dict(zip(names, (0.4, 0.3, 0.2, 0.1), strict=True))
    layers: list[dict[str, Any]] = []
    for layer_index in (16, 42, 62):
        layers.append(
            {
                "layer_index": layer_index,
                "suffix_sensitivity": "forward-ad",
                "baseline": {"mean_teacher_kl": primary["block-hessian"]},
                "candidates": [
                    {
                        "candidate": name,
                        "weight_nmse": control[name],
                        "baseline_weight_nmse": 0.0 if name == "block-hessian" else 0.1,
                        "mean_operator_nmse": control[name],
                        "mean_update_error_nmse": control[name],
                        "mean_resulting_hidden_nmse": control[name],
                        "max_recurrence_relative_residual": 1e-15,
                        "mean_teacher_kl": primary[name],
                        "mean_suffix_jvp_teacher_kl": primary[name],
                    }
                    for name in names
                ],
            }
        )
    return {
        "format": "mxwave-suffix-jvp-probe-v1",
        "status": "complete",
        "suffix_sensitivity": "forward-ad",
        "source": {"config_sha256": "source-config", "index_sha256": "source-index"},
        "execution": {
            "baseline_config_sha256": "baseline-config",
            "baseline_index_sha256": "baseline-index",
        },
        "calibration": {"file_sha256": "stats"},
        "heldout": {"token_ids_sha256": token_hash},
        "candidate_specs": [{"name": name} for name in names],
        "layers": layers,
    }


def test_evaluation_passes_a_stable_predictive_suffix_signal() -> None:
    result = evaluate_suffix_jvp_reports(_report("split-a"), _report("split-b"))

    assert result["passed"] is True
    assert result["mean_spearman"]["mean_suffix_jvp_teacher_kl"] == 1.0
    assert result["strongest_control_spearman"] == -1.0
    assert result["maximum_zero_tangent_teacher_kl_error"] == 0.0
    assert result["cross_split_improvement_rate"] == 1.0
    assert result["stable_layers"] == 3


def test_evaluation_rejects_reused_tokens() -> None:
    result = evaluate_suffix_jvp_reports(_report("same"), _report("same"))

    assert result["passed"] is False
    assert result["gates"]["identity_matches_and_tokens_are_disjoint"] is False
