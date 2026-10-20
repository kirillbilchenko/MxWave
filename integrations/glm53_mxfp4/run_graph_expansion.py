"""One compressed target with four physical captures and two alternating graph dispatch policies."""

from __future__ import annotations

from pathlib import Path

import run_eager_model_dispatch as single


def configure() -> None:
    """Keep the sequence/KV limits fixed while testing exact C2 and graph C8 dispatch."""
    single.REVISION = "graphexpandv1"
    single.PREPARED_NAME = "graph-expansion-model-v1-prepared.json"
    single.WORKER_CLASS = "graph_expansion_worker.GraphExpansionWorker"
    single.CAPTURE_SIZES = [1, 2, 4, 8]
    single.CLIENT_SCRIPT = "measure_graph_expansion.py"
    single.REPORT_NAME = "graph-expansion-model-review.json"
    single.ENTRY_SCRIPT = Path(__file__).resolve()
    single.SCOPE = "Same target/allocations: baseline [1,4] versus [1,2,4,8]; max sequences eight"
    single.ARM_LABEL = "candidate"
    single.CONTROL_REVISION = "pairedspeedv1"
    single.GPU_FIXTURE_SCRIPT = None


if __name__ == "__main__":
    configure()
    single.main()
