"""Run a fresh stock/compressed C1/C8 pair with native mHC and matched quality cohorts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import run_high_concurrency as high
import run_quality_model as controller


def launch(
    rank: int, revision: str, label: str, original: dict[str, Any],
    vocabulary: int, manifest_sha: str,
) -> tuple[list[str], dict[str, Any]]:
    """Preserve arithmetic, transport and capacity; use the same observation worker."""
    argv, meta = high.launch(rank, revision, label, original, vocabulary, manifest_sha)
    for command in (argv, meta["command"]):
        command[command.index("--worker-extension-cls") + 1] = "matched_c8_worker.MatchedC8Worker"
        at = command.index("--profiler-config") + 1
        config = json.loads(command[at])
        config.update(max_iterations=64, ignore_frontend=True)
        command[at] = json.dumps(config)
    return argv, meta


def configure() -> None:
    """Keep both-arm source checks, idle guard and independent exact restoration."""
    controller.launch = launch
    controller.PREPARED_NAME = "matched-c8-v1-prepared.json"
    controller.CLIENT_SCRIPT = "measure_matched_c8.py"
    controller.ANALYZE_SCRIPT = "analyze_matched_c8.py"
    controller.CLIENT_MEMORY = "6g"
    controller.CLIENT_WAIT_SECONDS = 2400
    controller.OUTAGE_SECONDS = 9000
    controller.ENTRY_SCRIPT = Path(__file__).resolve()
    controller.SANDBOX_TASKS = False


if __name__ == "__main__":
    configure()
    controller.main()
