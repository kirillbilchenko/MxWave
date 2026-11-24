"""Actual-target-input observer for the disposable stable-alignment diagnostic."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from nv_worker import NvWorker


class InputWorker(NvWorker):
    """Capture speculative input IDs and target outputs with normal graph execution."""

    def nv_inputs(self, enabled: bool, label: str = "") -> dict[str, Any]:
        """Arm or finish a bounded target-input observation between request groups."""
        from input_capture import InputCapture
        from vllm.distributed import get_tensor_model_parallel_rank

        if not hasattr(self, "_nv_inputs"):
            self._nv_inputs = InputCapture(self.model_runner)
        result = (self._nv_inputs.arm(label) if enabled else
                  self._nv_inputs.finish(Path("/traces") / self._nv_inputs.label))
        return {"rank": get_tensor_model_parallel_rank(), **result}
