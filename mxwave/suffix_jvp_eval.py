"""Frozen two-split evaluation gates for suffix-JVP probe reports."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .counteraction_eval import (
    _candidate_map,
    _float,
    _identity,
    _layer_map,
    _mapping,
    _spearman,
)

_PRIMARY = "mean_suffix_jvp_teacher_kl"
_CONTROLS = (
    "weight_nmse",
    "mean_operator_nmse",
    "mean_update_error_nmse",
    "mean_resulting_hidden_nmse",
)
_BASELINE_CANDIDATE = "block-hessian"
_CANDIDATES = {
    "rtn",
    "unweighted-mse",
    "diagonal-hessian",
    "block-hessian",
}

__all__ = ["evaluate_suffix_jvp_reports", "main"]


def _select_candidate(candidates: Mapping[str, Mapping[str, Any]]) -> str:
    values = {
        name: _float(candidate.get(_PRIMARY), f"{name}.{_PRIMARY}")
        for name, candidate in candidates.items()
    }
    minimum = min(values.values())
    tied = {
        name
        for name, value in values.items()
        if math.isclose(value, minimum, rel_tol=1e-9, abs_tol=1e-12)
    }
    if _BASELINE_CANDIDATE in tied:
        return _BASELINE_CANDIDATE
    return min(tied)


def evaluate_suffix_jvp_reports(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate two disjoint suffix-JVP reports against pre-registered gates."""
    reports = (first, second)
    complete = all(report.get("status") == "complete" for report in reports)
    format_valid = all(report.get("format") == "mxwave-suffix-jvp-probe-v1" for report in reports)
    sensitivity_valid = all(
        report.get("suffix_sensitivity") == "forward-ad" for report in reports
    )
    identity_matches = _identity(first) == _identity(second)
    first_execution = _mapping(first.get("execution"), "execution")
    second_execution = _mapping(second.get("execution"), "execution")
    runtime_matches = (
        first_execution.get("attention_implementation")
        == second_execution.get("attention_implementation")
        == "eager"
        and first_execution.get("dtype") == second_execution.get("dtype") == "bfloat16"
    )
    first_heldout = _mapping(first.get("heldout"), "heldout")
    second_heldout = _mapping(second.get("heldout"), "heldout")
    disjoint_tokens = first_heldout.get("token_ids_sha256") != second_heldout.get(
        "token_ids_sha256"
    )

    layer_maps = (_layer_map(first), _layer_map(second))
    same_layers = set(layer_maps[0]) == set(layer_maps[1]) and len(layer_maps[0]) == 3
    layer_indices = sorted(set(layer_maps[0]).intersection(layer_maps[1]))
    metric_correlations: dict[str, list[float]] = {
        metric: [] for metric in (_PRIMARY, *_CONTROLS)
    }
    reconstruction_values: list[float] = []
    recurrence_values: list[float] = []
    zero_tangent_errors: list[float] = []
    selections: list[dict[int, str]] = []

    for layer_map in layer_maps:
        split_selections: dict[int, str] = {}
        for layer_index in layer_indices:
            layer = layer_map[layer_index]
            if layer.get("suffix_sensitivity") != "forward-ad":
                sensitivity_valid = False
            candidates = _candidate_map(layer)
            if set(candidates) != _CANDIDATES:
                raise ValueError(f"Layer {layer_index} has a non-registered candidate set")
            candidate_names = sorted(candidates)
            teacher_values = [
                _float(candidates[name].get("mean_teacher_kl"), f"{name}.mean_teacher_kl")
                for name in candidate_names
            ]
            for metric, correlations in metric_correlations.items():
                metric_values = [
                    _float(candidates[name].get(metric), f"{name}.{metric}")
                    for name in candidate_names
                ]
                correlations.append(_spearman(metric_values, teacher_values))

            baseline = candidates[_BASELINE_CANDIDATE]
            baseline_result = _mapping(layer.get("baseline"), "baseline")
            baseline_kl = _float(
                baseline_result.get("mean_teacher_kl"),
                f"layer {layer_index} baseline.mean_teacher_kl",
            )
            exact_baseline_candidate_kl = _float(
                baseline.get("mean_teacher_kl"),
                f"layer {layer_index} block-hessian.mean_teacher_kl",
            )
            predicted_baseline_candidate_kl = _float(
                baseline.get(_PRIMARY),
                f"layer {layer_index} block-hessian.{_PRIMARY}",
            )
            zero_tangent_errors.extend(
                (
                    abs(exact_baseline_candidate_kl - baseline_kl),
                    abs(predicted_baseline_candidate_kl - baseline_kl),
                )
            )
            reconstruction_values.append(
                _float(
                    baseline.get("baseline_weight_nmse"),
                    f"layer {layer_index} baseline reconstruction",
                )
            )
            recurrence_values.extend(
                _float(
                    candidate.get("max_recurrence_relative_residual"),
                    f"layer {layer_index} recurrence",
                )
                for candidate in candidates.values()
            )
            split_selections[layer_index] = _select_candidate(candidates)
        selections.append(split_selections)

    mean_correlations = {
        metric: math.fsum(values) / len(values) if values else 0.0
        for metric, values in metric_correlations.items()
    }
    strongest_control = max(mean_correlations[metric] for metric in _CONTROLS)
    correlation_advantage = mean_correlations[_PRIMARY] - strongest_control

    cross_split_records: list[dict[str, Any]] = []
    improvements = 0
    for source_split, target_split in ((0, 1), (1, 0)):
        for layer_index in layer_indices:
            selected = selections[source_split][layer_index]
            target_candidates = _candidate_map(layer_maps[target_split][layer_index])
            selected_kl = _float(
                target_candidates[selected].get("mean_teacher_kl"),
                f"{selected}.mean_teacher_kl",
            )
            baseline_kl = _float(
                target_candidates[_BASELINE_CANDIDATE].get("mean_teacher_kl"),
                "block-hessian.mean_teacher_kl",
            )
            improved = selected_kl < baseline_kl - 1e-12
            improvements += int(improved)
            cross_split_records.append(
                {
                    "selected_on_split": source_split,
                    "evaluated_on_split": target_split,
                    "layer_index": layer_index,
                    "candidate": selected,
                    "selected_teacher_kl": selected_kl,
                    "baseline_teacher_kl": baseline_kl,
                    "improved": improved,
                }
            )

    decision_count = len(cross_split_records)
    improvement_rate = improvements / decision_count if decision_count else 0.0
    stable_layers = sum(
        selections[0].get(layer_index) == selections[1].get(layer_index)
        for layer_index in layer_indices
    )
    stability_rate = stable_layers / len(layer_indices) if layer_indices else 0.0
    maximum_reconstruction = max(reconstruction_values, default=math.inf)
    maximum_recurrence = max(recurrence_values, default=math.inf)
    maximum_zero_tangent_error = max(zero_tangent_errors, default=math.inf)

    gates = {
        "complete_format_and_sensitivity_valid": complete and format_valid and sensitivity_valid,
        "identity_runtime_match_and_tokens_are_disjoint": (
            identity_matches and runtime_matches and disjoint_tokens
        ),
        "three_matching_layers": same_layers,
        "baseline_reconstruction_at_most_1e-12": maximum_reconstruction <= 1e-12,
        "recurrence_residual_at_most_1e-10": maximum_recurrence <= 1e-10,
        "zero_tangent_teacher_kl_error_at_most_1e-8": maximum_zero_tangent_error <= 1e-8,
        "primary_spearman_at_least_0.50": mean_correlations[_PRIMARY] >= 0.50,
        "primary_advantage_at_least_0.20": correlation_advantage >= 0.20,
        "cross_split_improvement_at_least_0.70": improvement_rate >= 0.70,
        "selector_stable_on_two_of_three_layers": stable_layers >= 2,
    }
    return {
        "format": "mxwave-suffix-jvp-evaluation-v1",
        "passed": all(gates.values()),
        "gates": gates,
        "mean_spearman": mean_correlations,
        "strongest_control_spearman": strongest_control,
        "primary_correlation_advantage": correlation_advantage,
        "maximum_baseline_reconstruction_nmse": maximum_reconstruction,
        "maximum_recurrence_relative_residual": maximum_recurrence,
        "maximum_zero_tangent_teacher_kl_error": maximum_zero_tangent_error,
        "cross_split_improvement_rate": improvement_rate,
        "stable_layers": stable_layers,
        "selector_stability_rate": stability_rate,
        "selections": [
            {str(layer): candidate for layer, candidate in sorted(split.items())}
            for split in selections
        ],
        "cross_split_decisions": cross_split_records,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the frozen suffix-JVP report evaluator parser."""
    parser = argparse.ArgumentParser(prog="mxwave-suffix-jvp-evaluate")
    parser.add_argument("report_a")
    parser.add_argument("report_b")
    parser.add_argument("--output", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Evaluate two reports, returning zero only when every gate passes."""
    args = build_parser().parse_args(argv)
    try:
        first = _mapping(json.loads(Path(args.report_a).read_text()), "report A")
        second = _mapping(json.loads(Path(args.report_b).read_text()), "report B")
        result = evaluate_suffix_jvp_reports(first, second)
        rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.output is None:
            print(rendered, end="")
        else:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_name(f".{output.name}.incomplete")
            temporary.write_text(rendered)
            temporary.replace(output)
        return 0 if result["passed"] else 2
    except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
        print(f"mxwave: suffix-JVP evaluation error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
