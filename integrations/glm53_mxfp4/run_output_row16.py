"""Measure the passing output-only row16 component in the unchanged full C8/C16 protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import run_eager_model_dispatch as controller
from run_high_concurrency_retry import launch as pool_launch


def launch(rank: int, original: dict[str, Any], vocabulary: int,
           manifest_sha: str) -> tuple[list[str], dict[str, Any]]:
    """Opt in only the square output projection; keep input/global routing unchanged."""
    argv, metadata = pool_launch(rank, original, vocabulary, manifest_sha)
    for command in (argv, metadata["command"]):
        command[command.index("--worker-extension-cls") + 1] = (
            "output_row16_worker.OutputRow16Worker")
    metadata["environment"]["MXWAVE_GLM53_KDA_OUTPUT_FUSED_16"] = "1"
    at = argv.index("--entrypoint")
    argv[at:at] = ["-e", "MXWAVE_GLM53_KDA_OUTPUT_FUSED_16=1"]
    return argv, metadata


def configure() -> None:
    """Preserve idle guards, original restoration, numerical floors and matched workloads."""
    controller.REVISION = "output16v1"
    controller.PREPARED_NAME = "output-row16-v1-prepared.json"
    controller.CLIENT_SCRIPT = "measure_output_row16.py"
    controller.REPORT_NAME = "quality-model-review.json"
    controller.ENTRY_SCRIPT = Path(__file__).resolve()
    controller.SCOPE = (
        "Output-only exact16 route; unchanged input reconstruction and <=8 route. "
        "Reuse c816v1 stock and c816retryv2 compressed observations with explicit time gaps. "
        "No fresh paired model load, resident BF16 cache, new kernel or MTP.")
    controller.CONTROL_REVISION = "c816v1"
    controller.GPU_FIXTURE_SCRIPT = "kda_capture_pool_gpu_fixture.py"
    controller.CLIENT_MEMORY = "6g"
    controller.CLIENT_WAIT_SECONDS = 2700
    controller.launch = launch


if __name__ == "__main__":
    configure()
    controller.main()
