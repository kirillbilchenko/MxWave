"""Run one source-bound model with controlled admission and retained batch-four graphs."""

from __future__ import annotations

from pathlib import Path

import run_eager_model_dispatch as runner


def configure() -> None:
    """Bind controller and its independent watchdog to this immutable experiment revision."""
    runner.REVISION = "cohortdispatchv2"
    runner.PREPARED_NAME = "cohort-dispatch-model-v2-prepared.json"
    runner.WORKER_CLASS = "cohort_dispatch_worker.CohortDispatchWorker"
    runner.CAPTURE_SIZES = [1, 4]
    runner.CLIENT_SCRIPT = "measure_cohort_model_dispatch.py"
    runner.REPORT_NAME = "cohort-dispatch-model-review.json"
    runner.ENTRY_SCRIPT = Path(__file__).resolve()
    runner.SCOPE = ("Controlled four-request quality cohorts plus real API eager/graph-four timing "
                    "on one resident v9 candidate; all previous failed gates preserved")


if __name__ == "__main__":
    configure()
    runner.main()
