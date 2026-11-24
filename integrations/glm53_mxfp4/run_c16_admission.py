"""Run the final short-C16 admission comparison with the qualified warmed kernels."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import run_warmed_model as warmed


def launch(
    rank: int,
    revision: str,
    label: str,
    original: dict[str, Any],
    vocabulary: int,
    manifest_sha: str,
) -> tuple[list[str], dict[str, Any]]:
    """Preserve the verified parent's kernels, worker, graphs and capacity exactly."""
    if label not in ("control", "candidate"):
        raise ValueError("Undeclared admission arm")
    return warmed.launch(rank, revision, label, original, vocabulary, manifest_sha)


def configure() -> None:
    """Change the experiment client/reviewer while retaining exact service restoration."""
    warmed.configure()
    warmed.controller.launch = launch
    warmed.controller.PREPARED_NAME = "c16-admission-v1-prepared.json"
    warmed.controller.CLIENT_SCRIPT = "measure_c16_admission.py"
    warmed.controller.ANALYZE_SCRIPT = "analyze_c16_admission.py"
    warmed.controller.ENTRY_SCRIPT = Path(__file__).resolve()


if __name__ == "__main__":
    configure()
    warmed.controller.main()
