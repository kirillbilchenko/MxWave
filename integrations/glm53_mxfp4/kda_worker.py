"""Reuse qualified diagnostics while exposing one decoded KDA projection at a time."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from attention_diagnostic import AttentionDiagnosticWorker, check_attention_loaded
from diagnostic_worker import DiagnosticWorker


class DecodedView:
    """Present transient BF16 input matrices to the existing source-row checker."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def named_modules(self) -> Any:
        """Decode only adapted fused inputs, releasing each temporary before the next."""
        from mxwave_zipserv import runtime

        for name, module in self.model.named_modules():
            metadata = getattr(module, "mxwave_kda_metadata", None)
            if metadata is not None and name.endswith(".in_proj_qkvbfg_a"):
                packed = runtime.PackedWeight(
                    tuple(getattr(module, "mxwave_kda_" + n) for n in runtime.BUFFER_NAMES),
                    *metadata[:5])
                decoded = runtime.decompress(packed)
                yield name, SimpleNamespace(weight=decoded[:metadata[5]],
                                            quant_method=module.quant_method)
                del decoded
            else:
                yield name, module


class KdaWorker(AttentionDiagnosticWorker):
    """Keep inherited graph/warmup RPCs unchanged and report selective storage ownership."""

    def mxwave_checkpoint_check(self) -> dict[str, Any]:
        """Run original expert checks and the same attention checks through decoded views."""
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        result = DiagnosticWorker.mxwave_checkpoint_check(self)
        try:
            model = self.model_runner.model
            if os.environ.get("MXWAVE_GLM53_KDA_MODE") == "packed":
                model = DecodedView(model)
            attention = check_attention_loaded(model, Path("/model"),
                get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size())
        except Exception as exc:  # noqa: BLE001 -- controller retains evidence and restores.
            attention = {"status": "failed", "error": repr(exc)}
        result["attention"] = attention
        result["variant"] = os.environ.get("MXWAVE_GLM53_KDA_MODE", "stock")
        if attention["status"] != "passed":
            result["status"] = "failed"
        self._mxwave_checkpoint_report = result
        return result

    def mxwave_scale_report(self) -> dict[str, Any]:
        """Report actual allocation plus live selective layers and shared scratch once."""
        import kda_runtime
        import model_plugin

        result = model_plugin.report()
        result["kda"] = kda_runtime.report()
        adapted = [module for module in self.model_runner.model.modules()
                   if hasattr(module, "mxwave_kda_metadata")]
        result["kda"]["dense_parameter_references_alive"] = sum(
            getattr(module, "weight", None) is not None
            or module.mxwave_kda_original_parameter() is not None for module in adapted)
        return result
