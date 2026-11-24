"""Measure input-only FP32 split two under the unchanged warmed full-model protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import run_warmed_model as warmed

LIBRARY = "/audit/full-model-v1/fp32-split-v1-native/build/libmxwave_split_fp32.so"


def launch(
    rank: int,
    revision: str,
    label: str,
    original: dict[str, Any],
    vocabulary: int,
    manifest_sha: str,
) -> tuple[list[str], dict[str, Any]]:
    """Keep both arms' serving settings and enable only the candidate's input adapter."""
    if label not in ("control", "candidate"):
        raise ValueError("Require a declared stock or compressed comparison arm")
    argv, meta = warmed.launch(rank, revision, label, original, vocabulary, manifest_sha)
    for command in (argv, meta["command"]):
        command[command.index("--worker-extension-cls") + 1] = "fp32_input_worker.FP32InputWorker"
    extra = {"MXWAVE_GLM53_KDA_INPUT_FP32_SPLIT2": "1" if label == "candidate" else "0"}
    if label == "candidate":
        extra["MXWAVE_GLM53_FP32_LIBRARY"] = LIBRARY
    meta["environment"].update(extra)
    for key, value in extra.items():
        at = argv.index("--entrypoint")
        argv[at:at] = ["-e", key + "=" + value]
    return argv, meta


def configure() -> None:
    """Preserve the idle guard, fixed workload, finite outage and exact restoration."""
    warmed.configure()
    warmed.controller.launch = launch
    warmed.controller.PREPARED_NAME = "fp32-input-model-v1-prepared.json"
    warmed.controller.ANALYZE_SCRIPT = "analyze_fp32_input_model.py"
    warmed.controller.ENTRY_SCRIPT = Path(__file__).resolve()


if __name__ == "__main__":
    configure()
    warmed.controller.main()
