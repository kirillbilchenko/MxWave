"""Opt-in original-checkpoint numerical diagnostic RPCs."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from nv_worker import NvWorker


class RepeatWorker(NvWorker):
    """Keep original arithmetic and expose bounded prefill snapshots."""

    def nv_capture(self, enabled: bool, label: str = "", focus: int = -1) -> dict[str, Any]:
        """Arm between requests, or retain exact snapshot bytes after completion."""
        from repeat_capture import PrefillCapture
        from vllm.distributed import get_tensor_model_parallel_rank

        rank = get_tensor_model_parallel_rank()
        if not hasattr(self, "_nv_capture"):
            self._nv_capture = PrefillCapture(self.model_runner)
        if enabled:
            result = self._nv_capture.arm(label, focus)
        else:
            directory = Path("/traces") / self._nv_capture.label
            result = self._nv_capture.finish(directory)
        return {"rank": rank, **result}
