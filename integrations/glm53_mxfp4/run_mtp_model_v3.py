"""Run a fresh no-drafter/depth-two MTP comparison after actual zero-rope CPU proofs."""

from __future__ import annotations

from pathlib import Path

import run_mtp_pilot as pilot


def configure() -> None:
    """Bind this revision's frozen preparation and the existing paired restoration controller."""
    pilot.configure()
    quality = pilot.graph.quality
    quality.PREPARED_NAME = "mtp-pilot-v3-prepared.json"
    quality.CLIENT_WAIT_SECONDS = 1800
    quality.OUTAGE_SECONDS = 5400
    quality.ENTRY_SCRIPT = Path(__file__).resolve()


if __name__ == "__main__":
    configure()
    pilot.graph.quality.main()
