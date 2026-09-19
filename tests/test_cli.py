"""Public command-name regression tests."""

import tomllib
from pathlib import Path

from mxwave.calibration_cli import build_parser as build_calibration_parser
from mxwave.cli import build_parser as build_quantize_parser
from mxwave.precision_budget_cli import build_parser as build_precision_budget_parser


def test_public_command_names_are_mxwave() -> None:
    assert build_calibration_parser().prog == "mxwave-calibrate"
    assert build_quantize_parser().prog == "mxwave-quantize"
    assert build_precision_budget_parser().prog == "mxwave-precision-budget"


def test_packaged_entry_points_are_mxwave_only() -> None:
    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    assert project["project"]["scripts"] == {
        "mxwave-calibrate": "mxwave.calibration_cli:main",
        "mxwave-counteraction-evaluate": "mxwave.counteraction_eval:main",
        "mxwave-counteraction-probe": "mxwave.counteraction_probe_cli:main",
        "mxwave-exact-layout-evaluate": "mxwave.exact_layout_eval:main",
        "mxwave-exact-layout-probe": "mxwave.exact_layout_probe_cli:main",
        "mxwave-layer-confirmation-evaluate": "mxwave.layer_confirmation_eval:main",
        "mxwave-precision-budget": "mxwave.precision_budget_cli:main",
        "mxwave-quantize": "mxwave.cli:main",
        "mxwave-suffix-jvp-evaluate": "mxwave.suffix_jvp_eval:main",
    }
