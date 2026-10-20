"""Finish output16 numerical/profiling stages on a fresh load while retaining all original timings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import run_eager_model_dispatch as controller
from run_output_row16 import configure as configure_parent
from run_output_row16 import launch as parent_launch


def launch(
    rank: int, original: dict[str, Any], vocabulary: int, manifest_sha: str
) -> tuple[list[str], dict[str, Any]]:
    """Require space for numerical rows, profiles and retention before an idle service pause."""
    check = (
        "import json,shutil,sys;free=shutil.disk_usage(sys.argv[1]).free;"
        "assert free>=6*1024**3,('Require 6 GiB before model launch',free);"
        "print(json.dumps({'free_bytes':free,'required_bytes':6*1024**3}))"
    )
    controller.c.run(rank, ["python3", "-S", "-c", check, str(controller.c.ROOT)])
    return parent_launch(rank, original, vocabulary, manifest_sha)


def configure() -> None:
    """Keep the exact output-only decoder and distinguish this validation-only model load."""
    configure_parent()
    controller.REVISION = "output16qualv1"
    controller.PREPARED_NAME = "output16-validation-v1-prepared.json"
    controller.CLIENT_SCRIPT = "measure_output_validation.py"
    controller.ENTRY_SCRIPT = Path(__file__).resolve()
    controller.launch = launch
    controller.SCOPE = (
        "Validation/profiles only, unchanged output16 decoder. Reuses all72 timing groups from "
        "disk-full output16v1; fresh numerical repeats and profiles have a model-load/time gap. "
        "Original failed run and missing original source-after check stay explicit; no new "
        "timed groups, native kernel, BF16 cache, global threshold increase or MTP."
    )


if __name__ == "__main__":
    configure()
    controller.main()
