"""Run one uncompressed reference load for an intrusive identical-state graph diagnostic."""

from __future__ import annotations

from pathlib import Path

import run_eager_model_dispatch as runner


def configure() -> None:
    """Keep one reference arm, all source guards and the independent restoration watchdog."""
    runner.REVISION = "decodestatev3"
    runner.PREPARED_NAME = "decode-state-v3-prepared.json"
    runner.WORKER_CLASS = "decode_state_worker.DecodeStateWorker"
    runner.CAPTURE_SIZES = [1, 4]
    runner.CLIENT_SCRIPT = "measure_decode_state.py"
    runner.REPORT_NAME = "decode-state-model-review.json"
    runner.ENTRY_SCRIPT = Path(__file__).resolve()
    runner.ARM_LABEL = "control"
    runner.CONTROL_REVISION = "gsm8kfullv1"
    runner.GPU_FIXTURE_SCRIPT = "decode_state_gpu_fixture.py"
    runner.SCOPE = ("One raw-runtime reference; fixed input/cache graph replays and KDA state hashes; "
                    "intrusive diagnostic, no new accuracy or speed claim; old failures retained")


if __name__ == "__main__":
    configure()
    runner.main()
