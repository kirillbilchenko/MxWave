"""Run fresh C1/C8/C16 quality, speed and memory checks after identical startup preparation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import run_high_concurrency as high
import run_quality_model as controller


def launch(
    rank: int,
    revision: str,
    label: str,
    original: dict[str, Any],
    vocabulary: int,
    manifest_sha: str,
) -> tuple[list[str], dict[str, Any]]:
    """Preserve arithmetic, transport and capacity; use the same observation worker."""
    argv, meta = high.launch(rank, revision, label, original, vocabulary, manifest_sha)
    for command in (argv, meta["command"]):
        command[command.index("--worker-extension-cls") + 1] = (
            "warmed_model_worker.WarmedModelWorker"
        )
        at = command.index("--profiler-config") + 1
        config = json.loads(command[at])
        config.update(max_iterations=64, ignore_frontend=True)
        command[at] = json.dumps(config)
    if label == "candidate":
        extra = {"MXWAVE_GLM53_KDA_OUTPUT_FUSED_16": "1", "MXWAVE_GLM53_KDA_INPUT_FUSED_16": "1"}
        meta["environment"].update(extra)
        at = argv.index("--entrypoint")
        for key, value in extra.items():
            argv[at:at] = ["-e", key + "=" + value]
    return argv, meta


def configure() -> None:
    """Keep both-arm source checks, idle guard and independent exact restoration."""
    controller.launch = launch
    controller.PREPARED_NAME = "warmed-model-v1-prepared.json"
    controller.CLIENT_SCRIPT = "measure_warmed_model.py"
    controller.ANALYZE_SCRIPT = "analyze_warmed_model.py"
    controller.CLIENT_MEMORY = "6g"
    controller.CLIENT_WAIT_SECONDS = 3600
    controller.OUTAGE_SECONDS = 9000
    controller.ENTRY_SCRIPT = Path(__file__).resolve()
    controller.SANDBOX_TASKS = False


if __name__ == "__main__":
    configure()
    controller.main()
