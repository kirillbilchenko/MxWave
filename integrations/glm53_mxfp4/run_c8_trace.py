"""Profile the failing C8 stock workload and controlled repeats without changing kernels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import run_eager_model_dispatch as runner
import run_high_concurrency as high


def launch(rank: int, original: dict[str, Any], vocabulary: int, manifest_sha: str):
    """Preserve every prior C8 arithmetic/transport/capacity flag and add attribution only."""
    argv, meta = high.launch(rank, runner.REVISION, "control", original, vocabulary, manifest_sha)
    for command in (argv, meta["command"]):
        command[command.index("--worker-extension-cls") + 1] = runner.WORKER_CLASS
        at = command.index("--profiler-config") + 1
        config = json.loads(command[at])
        config.update(max_iterations=24, ignore_frontend=True)
        command[at] = json.dumps(config)
    return argv, meta


def configure() -> None:
    """Keep the prior idle guard, pinned native source proofs and restoration watchdog."""
    runner.REVISION = "c8tracev2"
    runner.PREPARED_NAME = "c8-trace-v2-prepared.json"
    runner.WORKER_CLASS = "c8_trace_worker.C8TraceWorker"
    runner.CAPTURE_SIZES = [1, 2, 4, 8, 16]
    runner.CLIENT_SCRIPT = "measure_c8_trace.py"
    runner.REPORT_NAME = "c8-trace-review.json"
    runner.ENTRY_SCRIPT = Path(__file__).resolve()
    runner.ARM_LABEL = "control"
    runner.CONTROL_REVISION = "gsm8kfullv1"
    runner.GPU_FIXTURE_SCRIPT = None
    runner.CLIENT_MEMORY = "6g"
    runner.CLIENT_WAIT_SECONDS = 1800
    runner.SCOPE = (
        "Profile failing stock C8 repeats with exact host batch attribution; retain separate "
        "unprofiled repeats and fixed-admission controls. Same kernels/2048-token budget; "
        "no compression qualification, automatic promotion or MTP."
    )
    runner.launch = launch


if __name__ == "__main__":
    configure()
    runner.main()
