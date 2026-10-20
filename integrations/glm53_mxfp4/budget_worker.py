"""Prepare deterministic expert dispatch before a fresh matched full-model comparison."""

from __future__ import annotations

import ctypes
import gc
import os
import time
from pathlib import Path
from typing import Any

from benchmark_stable_warmup import cache_inventory
from budget_dispatch_warmup import warmup
from budget_startup_preparation import validate_startup_report
from warmed_model_worker import WarmedModelWorker


class BudgetWorker(WarmedModelWorker):
    """Preserve the qualified native-mHC/exact16 runtime and add startup-only preparation."""

    def mxwave_moe_operand_capture(self, enabled: bool) -> dict[str, Any]:
        """Capture stock layer-23 operands only after measurements, using separate eager calls."""
        try:
            return self._mxwave_moe_capture_toggle(enabled)
        except Exception as exc:  # noqa: BLE001 -- diagnostics must not kill a loaded worker.
            capture = getattr(self, "_mxwave_moe_capture", None)
            if capture is not None:
                capture.restore()
            return {"status": "failed", "error": repr(exc)}

    def _mxwave_moe_capture_toggle(self, enabled: bool) -> dict[str, Any]:
        import re

        from moe_capture import LayerCapture
        from vllm.distributed import get_tensor_model_parallel_rank
        from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel

        rank = get_tensor_model_parallel_rank()
        if enabled:
            if getattr(self, "_mxwave_moe_capture", None) is not None:
                raise RuntimeError("Preserve one bounded capture attempt")
            if self.model_runner.cudagraph_manager._graphs_captured:
                raise RuntimeError("Disable graph dispatch after measurements before capture")
            pointers = {module.w13_weight.data_ptr()
                        for name, module in self.model_runner.model.named_modules()
                        if hasattr(module, "w13_weight") and re.search(
                            r"layers\.23\.mlp\.experts$", getattr(module, "layer_name", name))}
            if len(pointers) != 1:
                raise ValueError("Require exactly one layer-23 expert weight storage")
            self._mxwave_moe_capture = LayerCapture(FusedMoEKernel, pointers.pop())
            return {"status": "complete", "rank": rank, "armed": True,
                    "after_measured_work": True}
        capture = getattr(self, "_mxwave_moe_capture", None)
        if capture is None:
            raise RuntimeError("No owned MoE capture")
        try:
            return {"rank": rank, **capture.retain(Path("/traces/moe-operands") / f"rank{rank}")}
        finally:
            capture.restore()
            self._mxwave_moe_capture = None

    def mxwave_prepare_dispatch_startup(self) -> dict[str, Any]:
        """Warm supported variants, trim free CPU heap once and verify reuse before serving."""
        import torch
        from vllm.distributed import get_tensor_model_parallel_rank

        if getattr(self, "_mxwave_dispatch_startup_attempted", False):
            raise RuntimeError("Preserve the single declared startup-preparation attempt")
        self._mxwave_dispatch_startup_attempted = True
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Startup preparation must run outside graph capture")
        cache = Path(os.environ["TRITON_CACHE_DIR"])
        torch.cuda.synchronize()
        report: dict[str, Any] = {
            "status": "running", "rank": get_tensor_model_parallel_rank(),
            "started_epoch": time.time(), "CUDA_allocated_before": torch.cuda.memory_allocated(),
            "CUDA_reserved_before": torch.cuda.memory_reserved(),
            "telemetry_before": self.mxwave_gpu_telemetry(),
            "cache_before": cache_inventory(cache),
            "model_quality_or_serving_speed_qualified": False,
        }
        report["warmup"] = warmup(4096)
        report["cache_after_warmup"] = cache_inventory(cache)
        report["telemetry_after_warmup"] = self.mxwave_gpu_telemetry()
        libc = ctypes.CDLL(None)
        libc.gnu_get_libc_version.argtypes = []
        libc.gnu_get_libc_version.restype = ctypes.c_char_p
        libc.malloc_trim.argtypes = [ctypes.c_size_t]
        libc.malloc_trim.restype = ctypes.c_int
        gc.collect()
        before = self.mxwave_gpu_telemetry()
        started = time.perf_counter()
        returned = libc.malloc_trim(0)
        report["trim"] = {"returned": returned, "seconds": time.perf_counter() - started,
                          "glibc": libc.gnu_get_libc_version().decode(),
                          "telemetry_before": before,
                          "telemetry_after": self.mxwave_gpu_telemetry()}
        report["warmup_after_trim"] = warmup(4096)
        report["cache_after_trim_probe"] = cache_inventory(cache)
        report["telemetry_after"] = self.mxwave_gpu_telemetry()
        torch.cuda.synchronize()
        report.update(CUDA_allocated_after=torch.cuda.memory_allocated(),
                      CUDA_reserved_after=torch.cuda.memory_reserved(), finished_epoch=time.time())
        report["checks"] = validate_startup_report(report)
        report["status"] = "complete"
        self._mxwave_dispatch_startup_report = report
        return report

    def mxwave_dispatch_startup_snapshot(self) -> dict[str, Any]:
        """Detect any later canonicalization compilation between benchmark groups."""
        from vllm.distributed import get_tensor_model_parallel_rank

        report = getattr(self, "_mxwave_dispatch_startup_report", None)
        if report is None or report["status"] != "complete":
            raise RuntimeError("Complete startup preparation before checking cache reuse")
        current = cache_inventory(Path(os.environ["TRITON_CACHE_DIR"]))
        unchanged = current == report["cache_after_trim_probe"]
        if not unchanged:
            raise RuntimeError("A model request created or rewrote a dispatch specialization")
        return {"rank": get_tensor_model_parallel_rank(), "cache_unchanged": unchanged,
                "cache": current, "telemetry": self.mxwave_gpu_telemetry()}

    def mxwave_budget_memory(self, reset: bool = False) -> dict[str, Any]:
        """Observe per-trial allocation peaks outside request timing, without clearing caches."""
        import torch
        from vllm.distributed import get_tensor_model_parallel_rank

        torch.cuda.synchronize()
        if reset:
            torch.cuda.reset_peak_memory_stats()
        return {"rank": get_tensor_model_parallel_rank(), "reset": reset,
                "allocated_bytes": torch.cuda.memory_allocated(),
                "reserved_bytes": torch.cuda.memory_reserved(),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
