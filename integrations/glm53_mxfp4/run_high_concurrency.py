"""Matched stock/compressed C8/C16 timing with exact restoration and no speculative drafter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import run_graph_quality as graph


def launch(*args: Any, **kwargs: Any) -> tuple[list[str], dict[str, Any]]:
    """Give both targets identical captures through sixteen, queue hook and sequence capacity."""
    argv, metadata = graph.launch(*args, **kwargs)
    for command in (argv, metadata["command"]):
        command[command.index("--worker-extension-cls") + 1] = (
            "high_concurrency_worker.HighConcurrencyWorker")
        command[command.index("--max-num-seqs") + 1] = "16"
        command[command.index("--compilation-config") + 1] = json.dumps({
            "mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1, 2, 4, 8, 16], "max_cudagraph_capture_size": 16})
    metadata["environment"]["MXWAVE_PREFILL_ADMISSION"] = "1"
    at = argv.index("--entrypoint")
    argv[at:at] = ["-e", "MXWAVE_PREFILL_ADMISSION=1"]
    return argv, metadata


def configure() -> None:
    """Reuse the paired idle guard, native qualification binding and independent watchdog."""
    graph.configure()
    quality = graph.quality
    quality.launch = launch
    quality.PREPARED_NAME = "high-concurrency-v1-prepared.json"
    quality.CLIENT_SCRIPT = "measure_high_concurrency.py"
    quality.ANALYZE_SCRIPT = "analyze_high_concurrency.py"
    quality.CLIENT_WAIT_SECONDS = 2700
    quality.OUTAGE_SECONDS = 9000
    quality.CLIENT_MEMORY = "6g"
    quality.SANDBOX_TASKS = False
    quality.ENTRY_SCRIPT = Path(__file__).resolve()


if __name__ == "__main__":
    configure()
    graph.quality.main()
