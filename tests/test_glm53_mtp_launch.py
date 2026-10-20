"""Confine the pilot comparison to adding MTP while retaining experiment restoration ownership."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pytest


def test_both_arms_use_compressed_target_and_only_second_adds_mtp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))
    pilot = importlib.import_module("run_mtp_pilot")
    labels = []

    def base_launch(rank: int, revision: str, label: str, *args: Any) -> tuple[list[str], Any]:
        labels.append(label)
        command = ["serve", "/model", "--worker-extension-cls", "old.Worker",
            "--compilation-config", json.dumps({"cudagraph_capture_sizes": [1, 4]})]
        source = str(tmp_path / f"run-{revision}/kda/cache")
        return (["docker", "run", "--name", "old", "-v", source + ":/cache:rw",
                 "--entrypoint", "python3", "image", *command],
                {"name": "old", "command": command, "environment": {"MXWAVE_MOE_MODE": "aligned"},
                 "mounts": [(source, "/cache", "rw")]})

    monkeypatch.setattr(pilot.graph, "launch", base_launch)
    monkeypatch.setattr(pilot.graph.quality.c, "ROOT", tmp_path)
    monkeypatch.setattr(pilot.graph.quality.c, "run", lambda *args, **kwargs: "")
    control, cm = pilot.launch(0, "mtp1", "control", {}, 100, "sha")
    candidate, dm = pilot.launch(0, "mtp1", "candidate", {}, 100, "sha")
    assert labels == ["candidate", "candidate"]
    assert "--speculative-config" not in control
    assert json.loads(candidate[candidate.index("--speculative-config") + 1]) == {
        "method": "mtp", "num_speculative_tokens": 2, "use_local_argmax_reduction": True}
    assert cm["environment"]["MXWAVE_MOE_MODE"] == dm["environment"]["MXWAVE_MOE_MODE"] == "aligned"
    assert cm["name"] == "mxwave-glm53-full-mtp1-raw-r0"
    assert dm["name"] == "mxwave-glm53-full-mtp1-aligned-r0"
    assert cm["mounts"] == [(str(tmp_path / "run-mtp1/raw/cache"), "/cache", "rw")]
    assert dm["mounts"] == [(str(tmp_path / "run-mtp1/aligned/cache"), "/cache", "rw")]
