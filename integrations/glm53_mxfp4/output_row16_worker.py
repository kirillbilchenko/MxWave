"""Attest to a single measured output-projection exception without changing graph capacity."""

from __future__ import annotations

from typing import Any

from high_concurrency_pool_worker import HighConcurrencyPoolWorker


class OutputRow16Worker(HighConcurrencyPoolWorker):
    """Expose the exact sixteen-row route separately from the global eight-row limit."""

    def mxwave_high_concurrency_snapshot(self) -> dict[str, Any]:
        """Require all34 output-only exceptions and no input/intermediate-row exceptions."""
        import kda_runtime

        result = super().mxwave_high_concurrency_snapshot()
        storage = kda_runtime.report()
        layers = storage["layers"]
        inputs = [row for row in layers if row["kind"] == "input"]
        outputs = [row for row in layers if row["kind"] == "output"]
        if (not storage["output_fused_16"] or storage["fused_max_tokens"] != 8
                or len(inputs) != 34 or len(outputs) != 34
                or any(row["fused_exact_rows"] for row in inputs)
                or any(row["fused_exact_rows"] != [16]
                       or row["logical_shape"] != [4096, 4096] for row in outputs)):
            raise ValueError("Output-only sixteen-row scope differs from the frozen policy")
        result["output_row16_policy"] = {
            "enabled": True, "global_fused_limit": 8, "output_exact_rows": [16],
            "input_exception_weights": 0, "output_exception_weights": len(outputs),
            "intermediate_rows_qualified": False, "resident_BF16_cache_bytes": 0,
        }
        return result
