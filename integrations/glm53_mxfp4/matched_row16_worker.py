"""Opt in exact16 KDA routes while preserving native mHC, tracing and scratch lifetime."""

from __future__ import annotations

import os
from typing import Any

from input_row16_dispatch import install, policy
from matched_c8_worker import MatchedC8Worker

INPUT_ENABLED = os.environ.get("MXWAVE_GLM53_KDA_INPUT_FUSED_16", "0") == "1"
if INPUT_ENABLED:
    install()


class MatchedRow16Worker(MatchedC8Worker):
    """Attest to the component-qualified exact16 input/output exception."""

    def mxwave_high_concurrency_snapshot(self) -> dict[str, Any]:
        """Retain inherited guards and prove the bounded geometry on the candidate."""
        import kda_runtime

        result = super().mxwave_high_concurrency_snapshot()
        state = kda_runtime.report()
        if INPUT_ENABLED:
            inputs = [x for x in state["layers"] if x["kind"] == "input"]
            outputs = [x for x in state["layers"] if x["kind"] == "output"]
            assert len(inputs) == len(outputs) == 34
            assert state["fused_max_tokens"] == 8 and state["output_fused_16"]
            assert getattr(kda_runtime.use_fused, "_mxwave_input_sixteen", False)
            assert all(
                x["fused_exact_rows"] == [16]
                and x["logical_shape"] == [12576, 4096]
                and x["native_shape"] == [12608, 4096]
                for x in inputs
            )
            assert all(
                x["fused_exact_rows"] == [16] and x["logical_shape"] == [4096, 4096]
                for x in outputs
            )
            assert state["pool_bytes"] == 206569472
        else:
            assert not state["layers"] and not state["output_fused_16"]
        result["focused_row16_policy"] = {
            **policy(),
            "compressed": INPUT_ENABLED,
            "input_exact_rows": [16] if INPUT_ENABLED else [],
            "output_exact_rows": [16] if INPUT_ENABLED else [],
            "resident_BF16_weight_cache_bytes": 0,
        }
        return result
