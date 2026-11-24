"""Diagnostic observations around the unchanged, qualified combined model adapter."""

from __future__ import annotations

import gc
import threading
import time
from pathlib import Path
from typing import Any

import torch
from combined_worker import CombinedWorker


class NsysWorker(CombinedWorker):
    """Observe worker scheduling and GC without changing model computation or storage."""

    def mxwave_nsys_observe(self, enabled: bool) -> dict[str, Any]:
        """Annotate GC only during capture and retain bounded, per-thread observations."""
        callback = getattr(self, "_mxwave_gc_callback", None)
        if enabled and callback is None:
            self._mxwave_gc_events: list[dict[str, Any]] = []

            def observe(phase: str, info: dict[str, Any]) -> None:
                if phase == "start":
                    torch.cuda.nvtx.range_push(f"mxwave::gc::generation{info['generation']}")
                else:
                    torch.cuda.nvtx.range_pop()
                if len(self._mxwave_gc_events) < 512:
                    self._mxwave_gc_events.append({"phase": phase, **info,
                        "monotonic_ns": time.monotonic_ns(), "tid": threading.get_native_id()})

            self._mxwave_gc_callback = observe
            gc.callbacks.append(observe)
        elif not enabled and callback is not None:
            gc.callbacks.remove(callback)
            self._mxwave_gc_callback = None
        threads = []
        for task in sorted(Path("/proc/self/task").iterdir()):
            try:
                values = [int(v) for v in (task / "schedstat").read_text().split()]
                threads.append({"tid": int(task.name), "name": (task / "comm").read_text().strip(),
                    "cpu_runtime_ns": values[0], "runqueue_wait_ns": values[1],
                    "timeslices": values[2]})
            except FileNotFoundError:
                continue
        reports = [{"name": p.name, "bytes": p.stat().st_size}
                   for p in sorted(Path("/traces").glob("nsys-r*.nsys-rep"))]
        return {"status": "complete", "observation_enabled": enabled, "capture_reports": reports,
            "kineto_profiler_enabled": torch.autograd._profiler_enabled(),
            "gc_stats": gc.get_stats(), "gc_count": list(gc.get_count()),
            "gc_events": list(getattr(self, "_mxwave_gc_events", [])), "threads": threads,
            "monotonic_ns": time.monotonic_ns(), "epoch": time.time(),
            "limitation": "schedstat counters are interval totals, not a scheduling timeline"}
