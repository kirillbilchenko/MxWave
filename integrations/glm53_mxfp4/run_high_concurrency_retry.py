"""Retry only compressed C8/C16 after a bounded scratch-lifetime CUDA regression fixture."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import run_eager_model_dispatch as controller
from run_high_concurrency import launch as high_concurrency_launch


def launch(rank: int, original: dict[str, Any], vocabulary: int,
           manifest_sha: str) -> tuple[list[str], dict[str, Any]]:
    """Retain both-arm C16 settings; install startup scratch retirement on the compressed target."""
    argv, metadata = high_concurrency_launch(rank, controller.REVISION, "candidate",
                                            original, vocabulary, manifest_sha)
    for command in (argv, metadata["command"]):
        command[command.index("--worker-extension-cls") + 1] = (
            "high_concurrency_pool_worker.HighConcurrencyPoolWorker")
    return argv, metadata


def configure() -> None:
    """Keep the independent idle guard/watchdog and label the completed stock arm as reused."""
    controller.REVISION = "c816retryv2"
    controller.PREPARED_NAME = "high-concurrency-retry-v2-prepared.json"
    controller.CLIENT_SCRIPT = "measure_high_concurrency_retry.py"
    controller.REPORT_NAME = "quality-model-review.json"
    controller.ENTRY_SCRIPT = Path(__file__).resolve()
    controller.SCOPE = (
        "Compressed-only retry with startup scratch retirement; reuses the complete c816v1 stock "
        "arm. Matched settings with a time gap; no fresh paired stock load. MTP remains deferred.")
    controller.CONTROL_REVISION = "c816v1"
    controller.GPU_FIXTURE_SCRIPT = "kda_capture_pool_gpu_fixture.py"
    controller.CLIENT_MEMORY = "6g"
    controller.CLIENT_WAIT_SECONDS = 2700
    controller.launch = launch


if __name__ == "__main__":
    configure()
    controller.main()
