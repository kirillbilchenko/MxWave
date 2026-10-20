"""Compare the current model arms at C1/C4/C8 without changing their numerical gates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import run_graph_quality as graph


def launch(*args: Any, **kwargs: Any) -> tuple[list[str], dict[str, Any]]:
    """Retain graphs [1,4], fixed KV, canonical dispatch and all storage flags in both arms."""
    argv, meta = graph.launch(*args, **kwargs)
    for command in (argv, meta["command"]):
        command[command.index("--worker-extension-cls") + 1] = "speed_worker.SpeedWorker"
    return argv, meta


def configure() -> None:
    """Reuse the tested paired controller, idle guard and exact restoration watchdog."""
    graph.configure()
    quality = graph.quality
    quality.launch = launch
    quality.PREPARED_NAME = "paired-speed-v1-prepared.json"
    quality.CLIENT_SCRIPT = "measure_paired_speed.py"
    quality.ANALYZE_SCRIPT = "analyze_paired_speed.py"
    quality.CLIENT_WAIT_SECONDS = 1800
    quality.OUTAGE_SECONDS = 5400
    quality.CLIENT_MEMORY = "4g"
    quality.SANDBOX_TASKS = False
    quality.ENTRY_SCRIPT = Path(__file__).resolve()


if __name__ == "__main__":
    configure()
    graph.quality.main()
