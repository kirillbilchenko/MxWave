"""Retry only native MTP after fixing draft CT policy, reusing the completed no-drafter control."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import run_eager_model_dispatch as single
import run_mtp_pilot as pilot


def launch(rank: int, original: dict[str, Any], vocabulary: int,
           manifest_sha: str) -> tuple[list[str], dict[str, Any]]:
    """Keep the frozen compressed target and transport; repair only the MTP load/observation path."""
    return pilot.launch(rank, single.REVISION, "candidate", original, vocabulary, manifest_sha)


def configure() -> None:
    """Reuse one-load idle admission and exact independent restoration, without replaying control."""
    pilot.configure()
    single.REVISION = "mtpretryv1"
    single.PREPARED_NAME = "mtp-retry-v1-prepared.json"
    single.WORKER_CLASS = "mtp_pilot_worker.MtpPilotWorker"
    single.CAPTURE_SIZES = [1, 4]
    single.CLIENT_SCRIPT = "measure_mtp_retry.py"
    single.REPORT_NAME = "quality-model-review.json"
    single.ENTRY_SCRIPT = Path(__file__).resolve()
    single.SCOPE = ("Native depth-two MTP loader repair; completed mtppilotv2 no-drafter control "
                    "reused; target kernels/weights/precision unchanged; numerical failures retained")
    single.ARM_LABEL = "candidate"
    single.CONTROL_REVISION = "mtppilotv2"
    single.GPU_FIXTURE_SCRIPT = None
    single.launch = launch


if __name__ == "__main__":
    configure()
    single.main()
