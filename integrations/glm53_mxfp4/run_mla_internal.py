"""Run one raw-reference model with bounded actual-MLA observation after compression timing ends."""

from __future__ import annotations

from pathlib import Path

import run_eager_model_dispatch as runner


def configure() -> None:
    """Keep original reference kernels, idle guard, tiny fixture and independent restoration."""
    runner.REVISION = "mlainternalv1"
    runner.PREPARED_NAME = "mla-internal-v1-prepared.json"
    runner.WORKER_CLASS = "mla_internal_extension.MLAInternalExtension"
    runner.CAPTURE_SIZES = [1, 4]
    runner.CLIENT_SCRIPT = "measure_mla_internal.py"
    runner.REPORT_NAME = "decode-state-model-review.json"
    runner.ENTRY_SCRIPT = Path(__file__).resolve()
    runner.ARM_LABEL = "control"
    runner.CONTROL_REVISION = "gsm8kfullv1"
    runner.GPU_FIXTURE_SCRIPT = "mla_internal_gpu_fixture.py"
    runner.CLIENT_MEMORY = "6g"
    runner.CLIENT_WAIT_SECONDS = 2700
    runner.SCOPE = (
        "One original raw reference; actual layer3 query/value BMMs, cache writes, sparse slots, "
        "selected KV bytes and FlashInfer outputs. Five fixed full-graph replays, serve fifth; "
        "balanced tracked/private-workspace controls. Intrusive diagnostic, no quality or speed "
        "qualification. Broad boundary copies replaced; MTP deferred.")


if __name__ == "__main__":
    configure()
    runner.main()
