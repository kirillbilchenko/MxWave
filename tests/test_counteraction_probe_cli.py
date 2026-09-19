"""Tests for bounded counteraction probe CLI validation."""

from __future__ import annotations

import pytest

from mxwave.counteraction_probe_cli import _candidate_specs, _parse_layers, build_parser


def _required_args() -> list[str]:
    return [
        "--model-dir",
        "/source",
        "--baseline-model-dir",
        "/baseline",
        "--corpus",
        "/corpus.json",
        "--activation-stats",
        "/stats.safetensors",
        "--output",
        "/report.json",
    ]


def test_parser_freezes_bounded_probe_defaults() -> None:
    args = build_parser().parse_args(_required_args())

    assert _parse_layers(args.layers) == (8, 16, 31, 42, 54, 62)
    assert args.num_sequences == 8
    assert args.sequence_offset == 104
    assert args.sequence_length == 512
    assert args.logit_positions == 8
    assert args.max_elapsed_minutes == 90.0
    assert [spec.name for spec in _candidate_specs(args)] == [
        "rtn",
        "unweighted-mse",
        "diagonal-hessian",
        "block-hessian",
    ]


def test_layer_zero_is_rejected_as_non_counteraction_control() -> None:
    with pytest.raises(ValueError, match="no inherited error"):
        _parse_layers("0,8")
