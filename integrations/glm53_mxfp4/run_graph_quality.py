"""Run a matched graph-four raw-runtime/reference and compressed-candidate quality experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import run_quality_model as quality
from run_stable_quality import launch as stable_launch


def launch(*args: Any, **kwargs: Any) -> tuple[list[str], dict[str, Any]]:
    """Use identical graphs and read-only coverage reporting in both model arms."""
    argv, meta = stable_launch(*args, **kwargs)
    for command in (argv, meta["command"]):
        command[command.index("--worker-extension-cls") + 1] = (
            "graph_quality_worker.GraphQualityWorker"
        )
        command[command.index("--compilation-config") + 1] = json.dumps({
            "mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1, 4], "max_cudagraph_capture_size": 4,
        })
    return argv, meta


def configure() -> None:
    """Keep the independent watchdog bound to this exact paired-run configuration."""
    quality.launch = launch
    quality.PREPARED_NAME = "graph-quality-v1-prepared.json"
    quality.CLIENT_SCRIPT = "measure_graph_quality.py"
    quality.ANALYZE_SCRIPT = "analyze_graph_quality.py"
    quality.CLIENT_MEMORY = "4g"
    quality.CLIENT_WAIT_SECONDS = 5400
    quality.OUTAGE_SECONDS = 14400
    quality.ENTRY_SCRIPT = Path(__file__).resolve()


if __name__ == "__main__":
    configure()
    quality.main()
