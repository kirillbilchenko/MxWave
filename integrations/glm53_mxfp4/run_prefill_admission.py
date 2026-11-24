"""One resident compressed target, bounded input-window variants, and exact original restoration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import run_eager_model_dispatch as single

_BASE_LAUNCH = single.launch


def launch(*args: Any, **kwargs: Any) -> tuple[list[str], dict[str, Any]]:
    """Keep target kernels/precision/graphs and enable only the experiment queue hook."""
    argv, metadata = _BASE_LAUNCH(*args, **kwargs)
    metadata["environment"]["MXWAVE_PREFILL_ADMISSION"] = "1"
    at = argv.index("--entrypoint")
    argv[at:at] = ["-e", "MXWAVE_PREFILL_ADMISSION=1"]
    return argv, metadata


def configure() -> None:
    """Reuse the proven one-load idle guard, component bindings and independent watchdog."""
    single.REVISION = "prefilladmissionv1"
    single.PREPARED_NAME = "prefill-admission-model-v1-prepared.json"
    single.WORKER_CLASS = "admission_worker.AdmissionWorker"
    single.CAPTURE_SIZES = [1, 4]
    single.CLIENT_SCRIPT = "measure_prefill_admission.py"
    single.REPORT_NAME = "prefill-admission-model-review.json"
    single.ENTRY_SCRIPT = Path(__file__).resolve()
    single.SCOPE = "Same compressed target; idle text-prefill input windows 0/3/10 ms; no MTP"
    single.ARM_LABEL = "candidate"
    single.CONTROL_REVISION = "pairedspeedv1"
    single.GPU_FIXTURE_SCRIPT = "prefill_admission_gpu_fixture.py"
    single.launch = launch


if __name__ == "__main__":
    configure()
    single.main()
