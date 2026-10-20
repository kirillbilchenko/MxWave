"""Read-only batch coverage and graph eligibility for the bounded concurrent diagnostic."""

from __future__ import annotations

from typing import Any

from eager_dispatch_worker import EagerDispatchWorker


class CohortDispatchWorker(EagerDispatchWorker):
    """Keep all inherited source/quality checks; expose no new inference mutation."""

    def mxwave_batch_snapshot(self) -> dict[str, Any]:
        """Observe Python dispatch counters and the retained graph manager between requests."""
        from stable_dispatch import report

        manager = getattr(self.model_runner, "cudagraph_manager", None)
        if manager is None or not manager.graphs:
            raise RuntimeError("Require the pinned runner and its retained CUDA graphs")
        modes = {}
        for size in (1, 2, 3, 4):
            descriptor = manager.dispatch(num_reqs=size, num_tokens=size,
                                          uniform_token_count=1, num_active_loras=0)
            modes[str(size)] = str(descriptor.cg_mode)
        return {"stable_dispatch": report(), "graph_modes": modes,
                "retained_graphs": len(manager.graphs),
                "graphs_enabled": bool(manager._graphs_captured)}
