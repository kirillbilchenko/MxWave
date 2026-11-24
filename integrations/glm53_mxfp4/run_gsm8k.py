"""Run the full paired GSM8K test set without retrying the failed reference KL qualification."""

from __future__ import annotations

from pathlib import Path

import run_graph_quality as graph


def configure() -> None:
    """Use unchanged graph-four model arms with a benchmark-only CPU client and watchdog."""
    graph.configure()
    graph.quality.PREPARED_NAME = "gsm8k-full-v1-prepared.json"
    graph.quality.CLIENT_SCRIPT = "measure_gsm8k.py"
    graph.quality.ANALYZE_SCRIPT = "analyze_gsm8k.py"
    graph.quality.CLIENT_WAIT_SECONDS = 10800
    graph.quality.OUTAGE_SECONDS = 28800
    graph.quality.ENTRY_SCRIPT = Path(__file__).resolve()
    graph.quality.SANDBOX_TASKS = False


if __name__ == "__main__":
    configure()
    graph.quality.main()
