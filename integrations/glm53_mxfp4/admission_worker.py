"""Expose ordinary speed checks and untimed, removable per-weight call observation."""

from __future__ import annotations

from typing import Any

import torch
from speed_worker import SpeedWorker


class AdmissionWorker(SpeedWorker):
    """Keep target arithmetic and captured graphs unchanged throughout the admission test."""

    def mxwave_prefill_weight_audit(self, enabled: bool) -> dict[str, Any]:
        """Observe only subsequent diagnostic requests; the client must finish all timing first."""
        import kda_runtime
        from prefill_weight_audit import WeightAudit
        from vllm.distributed import get_tensor_model_parallel_rank

        if enabled:
            if getattr(self, "_mxwave_weight_audit", None) is not None:
                raise RuntimeError("An earlier weight observer has not been removed")
            observer = WeightAudit(self.model_runner.model, kda_runtime,
                capture=torch.cuda.is_current_stream_capturing,
                compiling=torch.compiler.is_compiling)
            observer.start()
            self._mxwave_weight_audit = observer
            prefixes = sorted(module.mxwave_kda_qualification["prefix"]
                for module in self.model_runner.model.modules()
                if hasattr(module, "mxwave_kda_qualification"))
            if len(prefixes) != 68 or len(set(prefixes)) != 68:
                observer.stop()
                self._mxwave_weight_audit = None
                raise ValueError("Require all 68 distinct base-model KDA projections")
            return {"status": "armed", "rank": get_tensor_model_parallel_rank(),
                    "expected_weight_prefixes": prefixes, "during_unprofiled_timing": False}
        observer = getattr(self, "_mxwave_weight_audit", None)
        if observer is None:
            raise RuntimeError("No weight observer is active")
        result = observer.stop()
        self._mxwave_weight_audit = None
        return {**result, "rank": get_tensor_model_parallel_rank()}
