"""Run the bounded both-arm latency diagnostic with inherited exact service restoration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import run_matched_c8 as parent
import run_quality_model as controller


def launch(*args: Any, **kwargs: Any) -> tuple[list[str], dict[str, Any]]:
    """Retain matchedc8v1 arithmetic, adding only the diagnostic observer extension."""
    argv, meta = parent.launch(*args, **kwargs)
    for command in (argv, meta["command"]):
        command[command.index("--worker-extension-cls") + 1] = (
            "latency_probe_worker.LatencyProbeWorker"
        )
        index = command.index("--profiler-config") + 1
        config = json.loads(command[index])
        config.update(max_iterations=192, ignore_frontend=True)
        command[index] = json.dumps(config)
    return argv, meta


def configure() -> None:
    """Keep the shared idle guard, source checks and independent restoration watchdog."""
    parent.configure()
    controller.launch = launch
    controller.PREPARED_NAME = "latency-probe-v1-prepared.json"
    controller.CLIENT_SCRIPT = "measure_latency_probe.py"
    controller.ANALYZE_SCRIPT = "analyze_latency_probe.py"
    controller.CLIENT_MEMORY = "3g"
    controller.CLIENT_WAIT_SECONDS = 1800
    controller.OUTAGE_SECONDS = 5400
    controller.ENTRY_SCRIPT = Path(__file__).resolve()
    controller.SANDBOX_TASKS = False


if __name__ == "__main__":
    configure()
    controller.main()
