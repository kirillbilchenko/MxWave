"""Keep C8/C16 observations while pinning captured KDA scratch and retiring startup leases."""

from __future__ import annotations

from typing import Any

from high_concurrency_worker import HighConcurrencyWorker
from kda_capture_pool import install

install()


class HighConcurrencyPoolWorker(HighConcurrencyWorker):
    """Preserve native operations and attest to the startup-only two-slot lifetime repair."""

    def mxwave_high_concurrency_snapshot(self) -> dict[str, Any]:
        """Add lease/capture ownership to the existing read-only capacity observation."""
        import kda_capture_pool
        import kda_runtime

        result = super().mxwave_high_concurrency_snapshot()
        result["capture_pool_lifecycle"] = kda_capture_pool.report()
        storage = kda_runtime.report()
        result["kda_pool"] = {key: storage[key] for key in
            ("pool_bytes", "pool_streams", "captured_pool_streams", "fused_max_tokens")}
        return result
