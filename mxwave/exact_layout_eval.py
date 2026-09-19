"""Frozen evaluator for the two-split exact-layout experiment."""

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
from .exact_layout import EXACT_LAYOUT_CANDIDATES

_LAYERS = (16, 42, 62)
_FAMILIES = ("weight-norm-sort", "activation-weighted-sort")
_H64 = "block-hessian"
_IDENTITY = "identity-unweighted-down"
_OFFSETS = (136, 144)
_TOKEN_HASHES = (
    "d3873dbab129e1131e77beb000ee9ad6b021fe41b0746541f5687a3a347a08b9",
    "08715cfe3c0e33cb62cb017332f7f511d6b3779661f23bd535bed6735200ebb4",
)
_SOURCE_CONFIG_HASH = "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab"
_SOURCE_INDEX_HASH = "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df"
_BASELINE_CONFIG_HASH = "d19a06baa5074b9f1f7b38296d846852fa2c071d2cebdd833e409206fc3feff0"
_BASELINE_INDEX_HASH = "be4a2cd4e130058b23080c6f71172d1e0e37d703e27040deb60fe7f431a7331b"
_CALIBRATION_HASH = "4f9e726a77d2f9f9e304ef18fafc8ef103e12caccf00d3c9950369809824483f"
_CORPUS_HASH = "07ffd944ea4fc09f12e5d019633a77309199fdced749e3ddf13989e53586f36a"
_BOOTSTRAP_SAMPLES = 100_000
_BOOTSTRAP_SEED = 20_260_919

__all__ = ["evaluate_exact_layout", "main"]


def _float_sequence(value: object, label: str) -> tuple[float, ...]:
    return tuple(
        _float(item, f"{label}[{index}]")
        for index, item in enumerate(_sequence(value, label))
    )


def _stratified_interval(
    split_differences: Sequence[Sequence[float]],
    *,
    samples: int = _BOOTSTRAP_SAMPLES,
    seed: int = _BOOTSTRAP_SEED,
) -> tuple[float, float]:
    if len(split_differences) != 2 or any(len(split) != 8 for split in split_differences):
        raise ValueError("Exact-layout bootstrap requires two strata of eight differences")
    generator = random.Random(seed)
    means: list[float] = []
    for _ in range(samples):
        total = 0.0
        for split in split_differences:
            total += math.fsum(split[generator.randrange(8)] for _ in range(8))
        means.append(total / 16.0)
    means.sort()
    return (
        means[math.floor(0.025 * (samples - 1))],
        means[math.ceil(0.975 * (samples - 1))],
    )


def _registered_report(report: Mapping[str, Any], offset: int, token_hash: str) -> bool:
    heldout = _mapping(report.get("heldout"), "heldout")
    source = _mapping(report.get("source"), "source")
    execution = _mapping(report.get("execution"), "execution")
    calibration = _mapping(report.get("calibration"), "calibration")
    specs = _mapping(report.get("candidate_specs"), "candidate_specs")
    return (
        report.get("format") == "mxwave-exact-layout-probe-v1"
        and report.get("status") == "complete"
        and report.get("requested_layers") == list(_LAYERS)
        and heldout.get("sequence_offset") == offset
        and heldout.get("num_sequences") == 8
        and heldout.get("sequence_length") == 512
        and heldout.get("logit_positions_per_sequence") == 8
        and heldout.get("token_ids_sha256") == token_hash
        and heldout.get("corpus_sha256") == _CORPUS_HASH
        and source.get("config_sha256") == _SOURCE_CONFIG_HASH
        and source.get("index_sha256") == _SOURCE_INDEX_HASH
        and execution.get("baseline_config_sha256") == _BASELINE_CONFIG_HASH
        and execution.get("baseline_index_sha256") == _BASELINE_INDEX_HASH
        and execution.get("attention_implementation") == "sdpa"
        and execution.get("dtype") == "bfloat16"
        and execution.get("expected_baseline_candidate") == _H64
        and calibration.get("objective") == "block-hessian"
        and calibration.get("file_sha256") == _CALIBRATION_HASH
        and specs.get("names") == list(EXACT_LAYOUT_CANDIDATES)
        and specs.get("scale_percentile") == 99.5
        and specs.get("mse_clip_depth") == 4
        and specs.get("sort") == "stable-ascending"
    )


def evaluate_exact_layout(
    reports: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int = _BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    """Apply the frozen validity, paired-context, and bootstrap gates."""
    if len(reports) != 2:
        raise ValueError("Exact-layout evaluation requires exactly two reports")
    layer_maps = tuple(_layer_map(report) for report in reports)
    registered = all(
        _registered_report(report, offset, token_hash)
        and set(layers) == set(_LAYERS)
        for report, layers, offset, token_hash in zip(
            reports, layer_maps, _OFFSETS, _TOKEN_HASHES, strict=True
        )
    )
    observed_hashes = [
        _mapping(report.get("heldout"), "heldout").get("token_ids_sha256")
        for report in reports
    ]
    disjoint_hashes = len(set(observed_hashes)) == 2

    maximum_reconstruction = 0.0
    maximum_recurrence = 0.0
    invariants_valid = True
    permutation_hashes_stable = True
    for layer_index in _LAYERS:
        first_metadata: Mapping[str, Any] | None = None
        for split_index, layers in enumerate(layer_maps):
            layer = layers[layer_index]
            candidates = _candidate_map(layer)
            if set(candidates) != set(EXACT_LAYOUT_CANDIDATES):
                raise ValueError("Exact-layout report has a non-registered candidate set")
            baseline = candidates[_H64]
            maximum_reconstruction = max(
                maximum_reconstruction,
                _float(baseline.get("baseline_weight_nmse"), "baseline_weight_nmse"),
            )
            maximum_recurrence = max(
                maximum_recurrence,
                *(
                    _float(
                        candidate.get("max_recurrence_relative_residual"),
                        "max_recurrence_relative_residual",
                    )
                    for candidate in candidates.values()
                ),
            )
            metadata = _mapping(layer.get("candidate_metadata"), "candidate_metadata")
            if set(metadata) != set(EXACT_LAYOUT_CANDIDATES):
                invariants_valid = False
                continue
            for name in EXACT_LAYOUT_CANDIDATES:
                values = _mapping(metadata.get(name), f"candidate_metadata.{name}")
                invariants_valid &= all(
                    values.get(key) is True
                    for key in (
                        "permutation_bijective",
                        "inverse_layout_exact",
                        "unquantized_output_equivalent",
                    )
                )
            if split_index == 0:
                first_metadata = metadata
            elif first_metadata is not None:
                permutation_hashes_stable &= all(
                    _mapping(first_metadata.get(name), name).get("permutation_sha256")
                    == _mapping(metadata.get(name), name).get("permutation_sha256")
                    for name in EXACT_LAYOUT_CANDIDATES
                )

    family_results: dict[str, dict[str, Any]] = {}
    for family in _FAMILIES:
        layer_results: dict[str, Any] = {}
        passing_layers: list[int] = []
        for layer_index in _LAYERS:
            split_results: list[dict[str, Any]] = []
            h64_differences: list[tuple[float, ...]] = []
            identity_differences: list[tuple[float, ...]] = []
            for layers in layer_maps:
                candidates = _candidate_map(layers[layer_index])
                family_values = _float_sequence(
                    candidates[family].get("sample_teacher_kl"),
                    f"{family}.sample_teacher_kl",
                )
                h64_values = _float_sequence(
                    candidates[_H64].get("sample_teacher_kl"),
                    f"{_H64}.sample_teacher_kl",
                )
                identity_values = _float_sequence(
                    candidates[_IDENTITY].get("sample_teacher_kl"),
                    f"{_IDENTITY}.sample_teacher_kl",
                )
                if not len(family_values) == len(h64_values) == len(identity_values) == 8:
                    raise ValueError("Each exact-layout split must have eight paired contexts")
                h64_delta = tuple(
                    base - candidate
                    for base, candidate in zip(h64_values, family_values, strict=True)
                )
                identity_delta = tuple(
                    base - candidate
                    for base, candidate in zip(identity_values, family_values, strict=True)
                )
                h64_differences.append(h64_delta)
                identity_differences.append(identity_delta)
                split_results.append(
                    {
                        "mean_family_teacher_kl": math.fsum(family_values) / 8,
                        "mean_h64_teacher_kl": math.fsum(h64_values) / 8,
                        "mean_identity_teacher_kl": math.fsum(identity_values) / 8,
                        "wins_vs_h64": sum(value > 0.0 for value in h64_delta),
                    }
                )
            h64_interval = _stratified_interval(
                h64_differences, samples=bootstrap_samples
            )
            identity_interval = _stratified_interval(
                identity_differences, samples=bootstrap_samples
            )
            lower_means = all(
                split["mean_family_teacher_kl"] < split["mean_h64_teacher_kl"]
                and split["mean_family_teacher_kl"] < split["mean_identity_teacher_kl"]
                for split in split_results
            )
            paired_wins = all(split["wins_vs_h64"] >= 5 for split in split_results)
            passed = (
                lower_means
                and paired_wins
                and h64_interval[0] > 0.0
                and identity_interval[0] > 0.0
            )
            if passed:
                passing_layers.append(layer_index)
            layer_results[str(layer_index)] = {
                "passed": passed,
                "split_results": split_results,
                "h64_minus_family_bootstrap_95_interval": list(h64_interval),
                "identity_minus_family_bootstrap_95_interval": list(identity_interval),
            }
        family_results[family] = {
            "passing_layers": passing_layers,
            "passing_layer_count": len(passing_layers),
            "layers": layer_results,
        }

    counts = {family: family_results[family]["passing_layer_count"] for family in _FAMILIES}
    winner = max(_FAMILIES, key=lambda family: (counts[family], family == "weight-norm-sort"))
    global_gates = {
        "registered_scope_complete": registered,
        "token_hashes_disjoint": disjoint_hashes,
        "baseline_reconstruction_at_most_1e-12": maximum_reconstruction <= 1e-12,
        "recurrence_residual_at_most_1e-10": maximum_recurrence <= 1e-10,
        "layout_invariants_valid": invariants_valid,
        "permutation_hashes_stable_across_splits": permutation_hashes_stable,
        "winning_family_passes_at_least_2_of_3_layers": counts[winner] >= 2,
    }
    return {
        "format": "mxwave-exact-layout-evaluation-v1",
        "passed": all(global_gates.values()),
        "gates": global_gates,
        "winning_family": winner,
        "winning_layers": family_results[winner]["passing_layers"],
        "families": family_results,
        "maximum_baseline_reconstruction_nmse": maximum_reconstruction,
        "maximum_recurrence_relative_residual": maximum_recurrence,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": _BOOTSTRAP_SEED,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the frozen exact-layout evaluator parser."""
    parser = argparse.ArgumentParser(prog="mxwave-exact-layout-evaluate")
    parser.add_argument("reports", nargs=2)
    parser.add_argument("--output", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Evaluate two reports, returning zero only when every gate passes."""
    args = build_parser().parse_args(argv)
    try:
        reports = [
            _mapping(json.loads(Path(path).read_text()), f"report[{index}]")
            for index, path in enumerate(args.reports)
        ]
        result = evaluate_exact_layout(reports)
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
        print(f"mxwave: exact-layout evaluation error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
