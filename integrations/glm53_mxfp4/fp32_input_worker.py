"""Attest to an exploratory FP32 input policy while preserving matched model qualification."""

from __future__ import annotations

import os
from typing import Any

from fp32_input_policy import policy
from warmed_model_worker import WarmedModelWorker

ENABLED = os.environ.get("MXWAVE_GLM53_KDA_INPUT_FP32_SPLIT2", "0") == "1"
if ENABLED:
    from fp32_input_runtime import install

    install()


class FP32InputWorker(WarmedModelWorker):
    """Preserve the qualified warmup/trace behavior and expose the exact candidate routing."""

    def mxwave_high_concurrency_snapshot(self) -> dict[str, Any]:
        """Require the existing two-slot allocation and the explicit input-only wrapper."""
        import kda_runtime

        result = super().mxwave_high_concurrency_snapshot()
        if ENABLED:
            assert getattr(kda_runtime.apply, "_mxwave_input_fp32", False)
            state = kda_runtime.report()
            assert state["pool_bytes"] == 206569472 and state["resident_layers"] == 68
        else:
            assert not getattr(kda_runtime.apply, "_mxwave_input_fp32", False)
        result["fp32_input_policy"] = {**policy(), "enabled": ENABLED}
        return result
