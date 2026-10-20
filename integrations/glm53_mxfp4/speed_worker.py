"""Read-only dispatch and allocation observations for paired serving measurements."""

from __future__ import annotations

from typing import Any

import torch
from graph_quality_worker import GraphQualityWorker


class SpeedWorker(GraphQualityWorker):
    """Keep the qualified model operations and expose the unchanged eight-request limit."""

    def mxwave_speed_snapshot(self) -> dict[str, Any]:
        """Observe actual graph eligibility, counters and allocation between timing groups."""
        from vllm.distributed import get_tensor_model_parallel_rank

        result = self.mxwave_batch_snapshot()
        manager = self.model_runner.cudagraph_manager
        descriptor = manager.dispatch(num_reqs=8, num_tokens=8,
                                      uniform_token_count=1, num_active_loras=0)
        result["graph_modes"]["8"] = str(descriptor.cg_mode)
        result.update(rank=get_tensor_model_parallel_rank(),
                      cuda_allocated_bytes=torch.cuda.memory_allocated(),
                      cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated())
        return result
