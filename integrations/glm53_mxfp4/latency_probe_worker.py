"""Observe a focused latency replay with the unchanged matched-C8 model policy."""

from __future__ import annotations

import hashlib
import inspect
import time
from pathlib import Path
from typing import Any

from c8_trace_worker import RUNNER_SHA256
from latency_forward_trace import LatencyForwardTrace, TelemetrySampler
from matched_c8_worker import MatchedC8Worker

_NVML_HANDLE: Any = None


def nvml_snapshot() -> dict[str, Any]:
    """Read native NVML state without a subprocess fallback or settings changes."""
    import pynvml as nvml

    global _NVML_HANDLE
    if _NVML_HANDLE is None:
        nvml.nvmlInit()
        _NVML_HANDLE = nvml.nvmlDeviceGetHandleByIndex(0)
    handle = _NVML_HANDLE
    value: dict[str, Any] = {
        "epoch": time.time(),
        "method": "NVML",
        "sm_MHz": nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_SM),
        "temperature_C": nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU),
        "pstate": nvml.nvmlDeviceGetPowerState(handle),
    }
    optional = {
        "memory_MHz": lambda: nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_MEM),
        "throttle_reasons": lambda: nvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle),
        "GPU_utilization_percent": lambda: nvml.nvmlDeviceGetUtilizationRates(handle).gpu,
        "power_mW": lambda: nvml.nvmlDeviceGetPowerUsage(handle),
    }
    for name, read in optional.items():
        try:
            value[name] = read()
        except (AttributeError, nvml.NVMLError) as exc:
            value[name] = None
            value[name + "_unavailable"] = repr(exc)
    return value


class LatencyProbeWorker(MatchedC8Worker):
    """Keep model arithmetic unchanged while observing both TP workers."""

    def mxwave_latency_trace(self, enabled: bool, label: str = "") -> dict[str, Any]:
        """Arm between groups; retain actual grouping, host timings and bounded clock samples."""
        import torch
        from vllm.distributed import get_tensor_model_parallel_rank

        runner = self.model_runner
        if hasattr(self, "_mxwave_c8_forward_trace"):
            raise RuntimeError("Use only one forward observer in this diagnostic")
        if not hasattr(self, "_mxwave_latency_trace"):
            path = Path(inspect.getfile(type(runner)))
            if hashlib.sha256(path.read_bytes()).hexdigest() != RUNNER_SHA256:
                raise ValueError("Latency probe requires the pinned native runner")
            self._mxwave_latency_trace = LatencyForwardTrace(runner, torch.profiler.record_function)
        trace = self._mxwave_latency_trace
        telemetry = None
        if enabled:
            if trace.enabled:
                raise RuntimeError("Disarm the previous latency group first")
            sampler = TelemetrySampler(nvml_snapshot)
            sampler.start()
            try:
                result = trace.configure(True, label)
            except BaseException:
                sampler.stop()
                raise
            self._mxwave_latency_sampler = sampler
        else:
            result = trace.configure(False, label)
            sampler = getattr(self, "_mxwave_latency_sampler", None)
            if sampler is None:
                raise RuntimeError("No active telemetry sampler")
            telemetry = sampler.stop()
            self._mxwave_latency_sampler = None
        return {
            **result,
            "rank": get_tensor_model_parallel_rank(),
            "native_runner_sha256": RUNNER_SHA256,
            "telemetry": telemetry,
        }
