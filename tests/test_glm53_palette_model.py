"""Full-model qualification must preserve its reference and reject an unstable repeat."""

from __future__ import annotations

import copy
import importlib
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def source_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Import orchestration without touching a GPU or either Spark."""
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))


def test_unstable_repeat_cannot_be_qualified_by_an_identical_candidate() -> None:
    analyzer = importlib.import_module("analyze_palette_model")
    repeat = {
        "all_decode_conditioning_matches": True,
        "mean_decode_kl": 2.5e-5,
        "mean_decode_nll_difference": 0.0,
    }
    candidate = {**repeat, "mean_decode_kl": 0.0}
    assert analyzer.numerical_gates(repeat, candidate) == (False, False)
    repeat["mean_decode_kl"] = 0.0
    assert analyzer.numerical_gates(repeat, candidate) == (True, True)
    candidate["all_decode_conditioning_matches"] = False
    assert analyzer.numerical_gates(repeat, candidate) == (True, False)


def test_full_model_arms_keep_the_same_command_and_owned_restore_names(monkeypatch) -> None:
    runner = importlib.import_module("run_palette_model")
    command = [
        "serve",
        "/model",
        "--worker-extension-cls",
        "original.Worker",
        "--compilation-config",
        "{}",
    ]
    base = (
        [
            "docker",
            "run",
            "--name",
            "original",
            "-e",
            "MXWAVE_MOE_MODE=stock",
            "-e",
            "MXWAVE_MOE_ALIGNED_LIBRARY=/audit/original.so",
            "--entrypoint",
            "python3",
            *command,
        ],
        {
            "name": "original",
            "command": command,
            "environment": {"MXWAVE_MOE_ALIGNED_LIBRARY": "/audit/original.so"},
            "mounts": [],
        },
    )
    monkeypatch.setattr(runner.c, "engine", lambda *a, **kw: copy.deepcopy(base))
    control = runner.launch(0, "qualified", "control", {}, 154880, "manifest")
    candidate = runner.launch(0, "qualified", "candidate", {}, 154880, "manifest")
    assert control[1]["command"] == candidate[1]["command"]
    assert control[1]["name"] == "mxwave-glm53-full-qualified-raw-r0"
    assert candidate[1]["name"] == "mxwave-glm53-full-qualified-aligned-r0"
    assert control[1]["environment"]["MXWAVE_MOE_ALIGNED_LIBRARY"] == "/audit/original.so"
    assert "palette-full-v1" in candidate[1]["environment"]["MXWAVE_MOE_ALIGNED_LIBRARY"]
    assert "MXWAVE_PALETTE_MODEL=0" in control[0] and "MXWAVE_PALETTE_MODEL=1" in candidate[0]
