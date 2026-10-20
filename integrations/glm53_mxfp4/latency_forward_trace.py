"""Add host elapsed/CPU timing and bounded telemetry to the existing batch observer."""

from __future__ import annotations

import functools
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from typing import Any

from c8_forward_trace import ForwardTrace


class LatencyForwardTrace(ForwardTrace):
    """Time native calls without synchronizing CUDA or changing their arguments/results."""

    def __init__(self, runner: Any, annotation: Callable[[str], Any] = nullcontext) -> None:
        self.native_annotation = annotation
        super().__init__(runner, self._timed_annotation)
        original_prepare = runner.prepare_inputs

        @functools.wraps(original_prepare)
        def prepare(*args: Any, **kwargs: Any) -> Any:
            if not self.enabled or self.current is None:
                return original_prepare(*args, **kwargs)
            event = self.current
            started, cpu = time.perf_counter_ns(), time.thread_time_ns()
            try:
                return original_prepare(*args, **kwargs)
            finally:
                event["host_prepare_elapsed_ns"] = time.perf_counter_ns() - started
                event["host_prepare_thread_cpu_ns"] = time.thread_time_ns() - cpu

        runner.prepare_inputs = prepare

    @contextmanager
    def _timed_annotation(self, label: str) -> Iterator[None]:
        event = self.current
        if event is None:
            raise RuntimeError("Timed annotation requires an active forward")
        event["host_start_epoch_ns"] = time.time_ns()
        started, cpu = time.perf_counter_ns(), time.thread_time_ns()
        event["host_start_monotonic_ns"] = started
        try:
            with self.native_annotation(label):
                yield
        finally:
            event["host_end_monotonic_ns"] = time.perf_counter_ns()
            event["host_elapsed_ns"] = event["host_end_monotonic_ns"] - started
            event["host_thread_cpu_ns"] = time.thread_time_ns() - cpu
            event["host_end_epoch_ns"] = time.time_ns()

    def report(self) -> dict[str, Any]:
        """Delimit host-call timing from asynchronous GPU completion."""
        return {
            **super().report(),
            "host_timing_not_GPU_completion": True,
            "CUDA_synchronization_added": False,
            "thread_CPU_excludes_other_threads": True,
        }


class TelemetrySampler:
    """Bound read-only samples and stop the owned thread before returning a report."""

    def __init__(
        self,
        snapshot: Callable[[], dict[str, Any]],
        *,
        interval: float = 0.1,
        maximum_samples: int = 1200,
    ) -> None:
        if interval < 0.05 or maximum_samples < 2:
            raise ValueError("Telemetry needs a bounded, non-busy sampling interval")
        self.snapshot = snapshot
        self.interval = interval
        self.maximum_samples = maximum_samples
        self.samples: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.limit_reached = False
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def _sample(self) -> None:
        started = time.perf_counter_ns()
        sample = self.snapshot()
        self.samples.append(
            {
                **sample,
                "sample_start_monotonic_ns": started,
                "sample_end_monotonic_ns": time.perf_counter_ns(),
            }
        )

    def start(self) -> None:
        """Validate the first sample before starting an owned daemon thread."""
        if self.thread is not None or self.samples:
            raise RuntimeError("A telemetry sampler may start only once")
        self._sample()

        def collect() -> None:
            while not self.stop_event.wait(self.interval):
                if len(self.samples) >= self.maximum_samples:
                    self.limit_reached = True
                    return
                try:
                    self._sample()
                except Exception as exc:  # noqa: BLE001 -- retain observer failure, never spin.
                    self.errors.append(repr(exc))
                    return

        self.thread = threading.Thread(target=collect, name="mxwave-latency-telemetry", daemon=True)
        self.thread.start()

    def stop(self) -> dict[str, Any]:
        """Stop only this sampler and report gaps rather than claiming continuous coverage."""
        if self.thread is None:
            raise RuntimeError("Telemetry was not started")
        self.stop_event.set()
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            raise RuntimeError("Owned telemetry thread did not stop")
        return {
            "samples": list(self.samples),
            "errors": list(self.errors),
            "interval_seconds": self.interval,
            "maximum_samples": self.maximum_samples,
            "limit_reached": self.limit_reached,
            "settings_changed": False,
            "thread_stopped": True,
            "observer_uses_CPU": True,
        }
