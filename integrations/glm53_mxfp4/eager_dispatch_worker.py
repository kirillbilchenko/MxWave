"""Idle-only RPC selector for a single resident v9 model's eager KDA host comparison."""

from __future__ import annotations

from typing import Any

from stable_dispatch import StableWorker


class EagerDispatchWorker(StableWorker):
    """Retain stable source/quality checks and switch no model parameters or graph caches."""

    def mxwave_kda_dispatch(self, mode: str) -> dict[str, Any]:
        """Select only after the client verifies there are no requests in flight."""
        from kda_eager_dispatch import select

        return select(mode)
