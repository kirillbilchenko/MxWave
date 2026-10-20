"""Observe native C16 graph eligibility and capacity without replacing model operations."""

from __future__ import annotations

from typing import Any

import torch
from graph_quality_worker import GraphQualityWorker


class HighConcurrencyWorker(GraphQualityWorker):
    """Keep the qualified stock/compressed arithmetic and report actual native state."""

    def mxwave_high_concurrency_snapshot(self) -> dict[str, Any]:
        """Read descriptors and bounded capacity metadata between completed request groups."""
        from stable_dispatch import report
        from vllm.distributed import get_tensor_model_parallel_rank

        runner = self.model_runner
        config = runner.vllm_config
        manager = runner.cudagraph_manager
        descriptors = {}
        for size in (1, 2, 4, 8, 16):
            value = manager.dispatch(num_reqs=size, num_tokens=size,
                                      uniform_token_count=1, num_active_loras=0)
            descriptors[str(size)] = {"mode": str(value.cg_mode), "num_tokens": value.num_tokens,
                                      "num_requests": value.num_reqs}
        kv_config = getattr(runner, "kv_cache_config", None)
        return {"rank": get_tensor_model_parallel_rank(),
            "physical_capture_sizes": sorted({d.num_tokens for d in manager.graphs}),
            "graphs_enabled": bool(manager._graphs_captured), "descriptors": descriptors,
            "max_num_seqs": manager.max_num_reqs,
            "max_batched_tokens": config.scheduler_config.max_num_batched_tokens,
            "max_model_len": config.model_config.max_model_len,
            "KV_bytes_per_rank": config.cache_config.kv_cache_memory_bytes,
            "native_cache_num_blocks": getattr(kv_config, "num_blocks", None),
            "native_cache_group_count": len(getattr(kv_config, "kv_cache_groups", [])),
            "speculative_config_absent": config.speculative_config is None,
            "stable_dispatch": report(), "cuda_allocated_bytes": torch.cuda.memory_allocated(),
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated()}
