"""Frozen evaluation for the layer-62 unweighted-MSE confirmation."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .counteraction_eval import _candidate_map, _float, _layer_map, _mapping, _sequence

_LAYER = 62
_CANDIDATE = "unweighted-mse"
_BASELINE = "block-hessian"
_BOOTSTRAP_SAMPLES = 100_000
_BOOTSTRAP_SEED = 20_260_919
_TOKEN_HASH = "49ac6fcdd0f473d37d5258cd387c8e7a60d3ef5934810f66d901d2e9ed3aee81"

__all__ = ["evaluate_layer_confirmation", "main"]


def _float_sequence(value: object, label: str) -> tuple[float, ...]:
    return tuple(_float(item, f"{label}[{index}]") for index, item in enumerate(_sequence(value, label)))


def _bootstrap_interval(
    differences: Sequence[float],
    *,
    samples: int = _BOOTSTRAP_SAMPLES,
    seed: int = _BOOTSTRAP_SEED,
) -> tuple[float, float]:
    if not differences:
        raise ValueError("Bootstrap differences must not be empty")
    generator = random.Random(seed)
    count = len(differences)
    means = [
        math.fsum(differences[generator.randrange(count)] for _ in range(count)) / count
        for _ in range(samples)
    ]
    means.sort()
    lower = means[math.floor(0.025 * (samples - 1))]
    upper = means[math.ceil(0.975 * (samples - 1))]
    return lower, upper


def evaluate_layer_confirmation(report: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate the one registered fresh-split confirmation report."""
    heldout = _mapping(report.get("heldout"), "heldout")
    execution = _mapping(report.get("execution"), "execution")
    layers = _layer_map(report)
    registered_scope = (
        report.get("format") == "mxwave-counteraction-probe-v1"
        and report.get("status") == "complete"
        and report.get("suffix_sensitivity") == "none"
        and report.get("requested_layers") == [_LAYER]
        and heldout.get("sequence_offset") == 120
        and heldout.get("num_sequences") == 16
        and heldout.get("sequence_length") == 512
        and heldout.get("logit_positions_per_sequence") == 8
        and heldout.get("token_ids_sha256") == _TOKEN_HASH
        and execution.get("attention_implementation") == "eager"
        and execution.get("dtype") == "bfloat16"
        and set(layers) == {_LAYER}
    )
    layer = layers.get(_LAYER)
    if layer is None:
        raise ValueError(f"Report has no registered layer {_LAYER}")
    candidates = _candidate_map(layer)
    if set(candidates) != {_CANDIDATE, _BASELINE}:
        raise ValueError("Layer confirmation has a non-registered candidate set")

    candidate = candidates[_CANDIDATE]
    baseline = candidates[_BASELINE]
    candidate_values = _float_sequence(
        candidate.get("sample_teacher_kl"), f"{_CANDIDATE}.sample_teacher_kl"
    )
    baseline_values = _float_sequence(
        baseline.get("sample_teacher_kl"), f"{_BASELINE}.sample_teacher_kl"
    )
    recorded_baseline = _mapping(layer.get("baseline"), "baseline")
    recorded_values = _float_sequence(
        recorded_baseline.get("sample_teacher_kl"), "baseline.sample_teacher_kl"
    )
    if not len(candidate_values) == len(baseline_values) == len(recorded_values) == 16:
        raise ValueError("Layer confirmation must contain exactly 16 paired samples")

    differences = tuple(
        baseline_value - candidate_value
        for baseline_value, candidate_value in zip(
            baseline_values, candidate_values, strict=True
        )
    )
    interval = _bootstrap_interval(differences)
    mean_candidate = math.fsum(candidate_values) / len(candidate_values)
    mean_baseline = math.fsum(baseline_values) / len(baseline_values)
    wins = sum(value > 0.0 for value in differences)
    maximum_baseline_error = max(
        abs(left - right)
        for left, right in zip(baseline_values, recorded_values, strict=True)
    )
    reconstruction = _float(
        baseline.get("baseline_weight_nmse"), "block-hessian.baseline_weight_nmse"
    )
    recurrence = max(
        _float(
            value.get("max_recurrence_relative_residual"),
            f"{name}.max_recurrence_relative_residual",
        )
        for name, value in candidates.items()
    )

    gates = {
        "registered_scope_complete": registered_scope,
        "baseline_reconstruction_at_most_1e-12": reconstruction <= 1e-12,
        "recurrence_residual_at_most_1e-10": recurrence <= 1e-10,
        "baseline_teacher_kl_error_at_most_1e-8": maximum_baseline_error <= 1e-8,
        "candidate_mean_teacher_kl_lower": mean_candidate < mean_baseline,
        "candidate_wins_at_least_12_of_16": wins >= 12,
        "paired_bootstrap_lower_bound_positive": interval[0] > 0.0,
    }
    return {
        "format": "mxwave-layer-confirmation-evaluation-v1",
        "passed": all(gates.values()),
        "gates": gates,
        "layer_index": _LAYER,
        "candidate": _CANDIDATE,
        "baseline": _BASELINE,
        "mean_candidate_teacher_kl": mean_candidate,
        "mean_baseline_teacher_kl": mean_baseline,
        "mean_relative_improvement": (
            (mean_baseline - mean_candidate) / mean_baseline
            if mean_baseline > 0.0
            else 0.0
        ),
        "paired_wins": wins,
        "paired_count": len(differences),
        "paired_bootstrap_95_interval": list(interval),
        "maximum_baseline_teacher_kl_error": maximum_baseline_error,
        "baseline_reconstruction_nmse": reconstruction,
        "maximum_recurrence_relative_residual": recurrence,
        "bootstrap_samples": _BOOTSTRAP_SAMPLES,
        "bootstrap_seed": _BOOTSTRAP_SEED,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the frozen confirmation evaluator parser."""
    parser = argparse.ArgumentParser(prog="mxwave-layer-confirmation-evaluate")
    parser.add_argument("report")
    parser.add_argument("--output", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Evaluate one report, returning zero only when every gate passes."""
    args = build_parser().parse_args(argv)
    try:
        report = _mapping(json.loads(Path(args.report).read_text()), "report")
        result = evaluate_layer_confirmation(report)
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
        print(f"mxwave: layer confirmation evaluation error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
