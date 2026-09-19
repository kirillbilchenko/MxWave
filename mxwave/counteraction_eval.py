"""Frozen two-split evaluation gates for counteraction probe reports."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

_PRIMARY = "mean_resulting_hidden_nmse"
_CONTROLS = ("weight_nmse", "mean_operator_nmse", "mean_update_error_nmse")
_BASELINE_CANDIDATE = "block-hessian"

__all__ = ["evaluate_counteraction_reports", "main"]


def _float(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _ranks(values: Sequence[float]) -> list[float]:
    """Return ascending average ranks with deterministic handling of ties."""
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    result = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        stop = cursor + 1
        while stop < len(order) and values[order[stop]] == values[order[cursor]]:
            stop += 1
        average_rank = (cursor + 1 + stop) / 2.0
        for position in range(cursor, stop):
            result[order[position]] = average_rank
        cursor = stop
    return result


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Spearman inputs must have equal length of at least two")
    left_ranks = _ranks(left)
    right_ranks = _ranks(right)
    left_mean = math.fsum(left_ranks) / len(left_ranks)
    right_mean = math.fsum(right_ranks) / len(right_ranks)
    covariance = math.fsum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left_ranks, right_ranks, strict=True)
    )
    left_energy = math.fsum((value - left_mean) ** 2 for value in left_ranks)
    right_energy = math.fsum((value - right_mean) ** 2 for value in right_ranks)
    denominator = math.sqrt(left_energy * right_energy)
    return covariance / denominator if denominator > 0.0 else 0.0


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, label: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a list")
    return cast(Sequence[Any], value)


def _layer_map(report: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    result: dict[int, Mapping[str, Any]] = {}
    for raw_layer in _sequence(report.get("layers"), "layers"):
        layer = _mapping(raw_layer, "layer")
        raw_index = layer.get("layer_index")
        if not isinstance(raw_index, int) or isinstance(raw_index, bool):
            raise TypeError("layer_index must be an integer")
        if raw_index in result:
            raise ValueError(f"Duplicate layer result: {raw_index}")
        result[raw_index] = layer
    return result


def _candidate_map(layer: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for raw_candidate in _sequence(layer.get("candidates"), "candidates"):
        candidate = _mapping(raw_candidate, "candidate")
        name = candidate.get("candidate")
        if not isinstance(name, str) or not name:
            raise TypeError("candidate name must be a non-empty string")
        if name in result:
            raise ValueError(f"Duplicate candidate result: {name}")
        result[name] = candidate
    return result


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


def _identity(report: Mapping[str, Any]) -> tuple[object, ...]:
    source = _mapping(report.get("source"), "source")
    execution = _mapping(report.get("execution"), "execution")
    calibration = _mapping(report.get("calibration"), "calibration")
    specs = _sequence(report.get("candidate_specs"), "candidate_specs")
    names = tuple(_mapping(item, "candidate spec").get("name") for item in specs)
    return (
        source.get("config_sha256"),
        source.get("index_sha256"),
        execution.get("baseline_config_sha256"),
        execution.get("baseline_index_sha256"),
        calibration.get("file_sha256"),
        names,
    )


def evaluate_counteraction_reports(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate two disjoint reports against the frozen pre-registered gates."""
    reports = (first, second)
    complete = all(report.get("status") == "complete" for report in reports)
    format_valid = all(
        report.get("format") == "mxwave-counteraction-probe-v1" for report in reports
    )
    identity_matches = _identity(first) == _identity(second)
    first_heldout = _mapping(first.get("heldout"), "heldout")
    second_heldout = _mapping(second.get("heldout"), "heldout")
    disjoint_tokens = first_heldout.get("token_ids_sha256") != second_heldout.get(
        "token_ids_sha256"
    )

    layer_maps = (_layer_map(first), _layer_map(second))
    same_layers = set(layer_maps[0]) == set(layer_maps[1]) and len(layer_maps[0]) == 6
    layer_indices = sorted(set(layer_maps[0]).intersection(layer_maps[1]))
    metric_correlations: dict[str, list[float]] = {
        metric: [] for metric in (_PRIMARY, *_CONTROLS)
    }
    reconstruction_values: list[float] = []
    recurrence_values: list[float] = []
    selections: list[dict[int, str]] = []

    for layer_map in layer_maps:
        split_selections: dict[int, str] = {}
        for layer_index in layer_indices:
            candidates = _candidate_map(layer_map[layer_index])
            if set(candidates) != {
                "rtn",
                "unweighted-mse",
                "diagonal-hessian",
                "block-hessian",
            }:
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

    gates = {
        "complete_and_format_valid": complete and format_valid,
        "identity_matches_and_tokens_are_disjoint": identity_matches and disjoint_tokens,
        "six_matching_layers": same_layers,
        "baseline_reconstruction_at_most_1e-12": maximum_reconstruction <= 1e-12,
        "recurrence_residual_at_most_1e-10": maximum_recurrence <= 1e-10,
        "primary_spearman_at_least_0.50": mean_correlations[_PRIMARY] >= 0.50,
        "primary_advantage_at_least_0.20": correlation_advantage >= 0.20,
        "cross_split_improvement_at_least_0.70": improvement_rate >= 0.70,
        "selector_stable_on_four_of_six_layers": stable_layers >= 4,
    }
    return {
        "format": "mxwave-counteraction-evaluation-v1",
        "passed": all(gates.values()),
        "gates": gates,
        "mean_spearman": mean_correlations,
        "strongest_control_spearman": strongest_control,
        "primary_correlation_advantage": correlation_advantage,
        "maximum_baseline_reconstruction_nmse": maximum_reconstruction,
        "maximum_recurrence_relative_residual": maximum_recurrence,
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
    """Build the frozen report evaluator parser."""
    parser = argparse.ArgumentParser(prog="mxwave-counteraction-evaluate")
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
        result = evaluate_counteraction_reports(first, second)
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
        print(f"mxwave: counteraction evaluation error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
