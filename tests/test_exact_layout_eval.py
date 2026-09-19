"""Tests for the frozen two-split exact-layout gate."""

from __future__ import annotations

from typing import Any

from mxwave.exact_layout import EXACT_LAYOUT_CANDIDATES
from mxwave.exact_layout_eval import evaluate_exact_layout


def _report(offset: int, token_hash: str, *, improvement: float = 0.01) -> dict[str, Any]:
    layers: list[dict[str, Any]] = []
    for layer_index in (16, 42, 62):
        h64 = [0.2 + sample * 0.001 for sample in range(8)]
        identity = [value - 0.002 for value in h64]
        permutation = [value - improvement for value in identity]
        metadata = {
            name: {
                "permutation_sha256": f"{layer_index}-{name}",
                "permutation_bijective": True,
                "inverse_layout_exact": True,
                "unquantized_output_equivalent": True,
            }
            for name in EXACT_LAYOUT_CANDIDATES
        }
        layers.append(
            {
                "layer_index": layer_index,
                "candidate_metadata": metadata,
                "candidates": [
                    {
                        "candidate": "block-hessian",
                        "sample_teacher_kl": h64,
                        "baseline_weight_nmse": 0.0,
                        "max_recurrence_relative_residual": 1e-15,
                    },
                    {
                        "candidate": "identity-unweighted-down",
                        "sample_teacher_kl": identity,
                        "baseline_weight_nmse": 0.1,
                        "max_recurrence_relative_residual": 1e-15,
                    },
                    {
                        "candidate": "weight-norm-sort",
                        "sample_teacher_kl": permutation,
                        "baseline_weight_nmse": 0.1,
                        "max_recurrence_relative_residual": 1e-15,
                    },
                    {
                        "candidate": "activation-weighted-sort",
                        "sample_teacher_kl": permutation,
                        "baseline_weight_nmse": 0.1,
                        "max_recurrence_relative_residual": 1e-15,
                    },
                ],
            }
        )
    return {
        "format": "mxwave-exact-layout-probe-v1",
        "status": "complete",
        "requested_layers": [16, 42, 62],
        "source": {
            "config_sha256": (
                "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab"
            ),
            "index_sha256": (
                "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df"
            ),
        },
        "execution": {
            "baseline_config_sha256": (
                "d19a06baa5074b9f1f7b38296d846852fa2c071d2cebdd833e409206fc3feff0"
            ),
            "baseline_index_sha256": (
                "be4a2cd4e130058b23080c6f71172d1e0e37d703e27040deb60fe7f431a7331b"
            ),
            "attention_implementation": "sdpa",
            "dtype": "bfloat16",
            "expected_baseline_candidate": "block-hessian",
        },
        "calibration": {
            "objective": "block-hessian",
            "file_sha256": (
                "4f9e726a77d2f9f9e304ef18fafc8ef103e12caccf00d3c9950369809824483f"
            ),
        },
        "heldout": {
            "sequence_offset": offset,
            "num_sequences": 8,
            "sequence_length": 512,
            "logit_positions_per_sequence": 8,
            "token_ids_sha256": token_hash,
            "corpus_sha256": (
                "07ffd944ea4fc09f12e5d019633a77309199fdced749e3ddf13989e53586f36a"
            ),
        },
        "candidate_specs": {
            "names": list(EXACT_LAYOUT_CANDIDATES),
            "scale_percentile": 99.5,
            "mse_clip_depth": 4,
            "sort": "stable-ascending",
        },
        "layers": layers,
    }


def _reports(*, improvement: float = 0.01) -> list[dict[str, Any]]:
    return [
        _report(
            136,
            "d3873dbab129e1131e77beb000ee9ad6b021fe41b0746541f5687a3a347a08b9",
            improvement=improvement,
        ),
        _report(
            144,
            "08715cfe3c0e33cb62cb017332f7f511d6b3779661f23bd535bed6735200ebb4",
            improvement=improvement,
        ),
    ]


def test_exact_layout_passes_consistent_two_split_improvement() -> None:
    result = evaluate_exact_layout(_reports(), bootstrap_samples=1_000)

    assert result["passed"] is True
    assert result["winning_family"] == "weight-norm-sort"
    assert result["winning_layers"] == [16, 42, 62]


def test_exact_layout_rejects_regression() -> None:
    result = evaluate_exact_layout(_reports(improvement=-0.01), bootstrap_samples=1_000)

    assert result["passed"] is False
    assert result["gates"]["winning_family_passes_at_least_2_of_3_layers"] is False
