"""Compare baseline and expanded graph dispatch with all graph allocations held resident."""

from __future__ import annotations

from typing import Any

import torch
from graph_quality_worker import GraphQualityWorker


class GraphExpansionWorker(GraphQualityWorker):
    """Select descriptors between closed groups; preserve native graph replay and model arithmetic."""

    def mxwave_graph_policy_select(self, policy: str) -> dict[str, Any]:
        """Require the four physical captures before selecting baseline [1,4] or [1,2,4,8]."""
        from graph_capture_policy import candidates_for_sizes

        if policy not in ("baseline", "expanded"):
            raise ValueError("Unknown frozen graph dispatch policy")
        manager = self.model_runner.cudagraph_manager
        if not manager._graphs_captured or not manager.graphs:
            raise RuntimeError("Graph captures are unavailable")
        sizes = sorted({d.num_tokens for d in manager.graphs})
        if sizes != [1, 2, 4, 8] or manager.max_num_reqs != 8:
            raise RuntimeError("Physical capture/sequence scope differs from the frozen comparison")
        torch.cuda.synchronize()
        allowed = {1, 4} if policy == "baseline" else {1, 2, 4, 8}
        candidates = candidates_for_sizes(manager, allowed)
        if any(d not in manager.graphs for descriptors in candidates.values() for d in descriptors):
            raise RuntimeError("Dispatch policy includes a descriptor that was never captured")
        manager._candidates = candidates
        self._mxwave_graph_policy = policy
        return self.mxwave_graph_policy_snapshot()

    def mxwave_graph_policy_snapshot(self) -> dict[str, Any]:
        """Report actual selected and padded token counts on both ranks without inferring execution."""
        from stable_dispatch import report
        from vllm.distributed import get_tensor_model_parallel_rank

        manager = self.model_runner.cudagraph_manager
        descriptors = {}
        for size in (1, 2, 3, 4, 8, 16):
            value = manager.dispatch(num_reqs=size, num_tokens=size,
                                     uniform_token_count=1, num_active_loras=0)
            descriptors[str(size)] = {"mode": str(value.cg_mode), "num_tokens": value.num_tokens,
                                      "num_requests": value.num_reqs}
        return {"rank": get_tensor_model_parallel_rank(),
            "policy": getattr(self, "_mxwave_graph_policy", "expanded"),
            "physical_capture_sizes": sorted({d.num_tokens for d in manager.graphs}),
            "graphs_enabled": bool(manager._graphs_captured), "descriptors": descriptors,
            "max_num_seqs": manager.max_num_reqs, "stable_dispatch": report(),
            "speculative_config_absent": self.model_runner.vllm_config.speculative_config is None,
            "cuda_allocated_bytes": torch.cuda.memory_allocated(),
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated()}
