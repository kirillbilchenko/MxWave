"""Add read-only batch attribution to the unchanged stock deterministic-compaction worker."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from typing import Any

from c8_forward_trace import ForwardTrace
from stable_sparse_quality_worker import StableSparseQualityWorker

RUNNER_SHA256 = "f84255d75435e84f44972d3fd25e53447f9d4d2edd8bff4f8c19dfb793448415"


class C8TraceWorker(StableSparseQualityWorker):
    """Observe exact scheduler membership and prepared row order on both TP ranks."""

    def mxwave_c8_trace(self, enabled: bool, label: str = "") -> dict[str, Any]:
        """Arm after startup/warmup and retain host metadata with profiler annotations."""
        import torch
        from vllm.distributed import get_tensor_model_parallel_rank

        runner = self.model_runner
        if not hasattr(self, "_mxwave_c8_forward_trace"):
            path = Path(inspect.getfile(type(runner)))
            if hashlib.sha256(path.read_bytes()).hexdigest() != RUNNER_SHA256:
                raise ValueError("C8 trace requires the audited pinned native runner")
            self._mxwave_c8_forward_trace = ForwardTrace(runner, torch.profiler.record_function)
        result = self._mxwave_c8_forward_trace.configure(enabled, label)
        return {
            **result,
            "rank": get_tensor_model_parallel_rank(),
            "native_runner_sha256": RUNNER_SHA256,
        }
