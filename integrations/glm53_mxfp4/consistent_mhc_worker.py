"""Install the source-pinned mHC path before constructing the stock diagnostic model."""

from __future__ import annotations

from typing import Any

from c8_trace_worker import C8TraceWorker
from consistent_mhc import install_consistent_mhc

POLICY_SNAPSHOT = install_consistent_mhc()


class ConsistentMHCWorker(C8TraceWorker):
    """Expose component-tested policy metadata and retain exact forward attribution."""

    def mxwave_consistent_mhc_policy(self) -> dict[str, Any]:
        """Report host capture/eager counters; graph replays are counted by device traces."""
        return POLICY_SNAPSHOT()
