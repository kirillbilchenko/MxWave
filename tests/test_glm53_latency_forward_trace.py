"""The latency observer must preserve native behavior and bound its telemetry thread."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))

from latency_forward_trace import LatencyForwardTrace, TelemetrySampler
from test_glm53_c8_forward_trace import _Runner, _scheduled


def test_latency_records_native_wait_without_turning_it_into_CPU_time() -> None:
    runner = _Runner()
    native = runner.execute_model

    def waiting(*args: object, **kwargs: object) -> object:
        time.sleep(0.02)
        return native(*args, **kwargs)

    runner.execute_model = waiting
    trace = LatencyForwardTrace(runner)
    scheduled = _scheduled()
    trace.configure(True, "candidate-split-first")
    assert runner.execute_model(scheduled, 11, dummy_run=False) is runner.value
    report = trace.configure(False, "")
    event = report["events"][0]
    assert event["host_elapsed_ns"] >= 20_000_000
    assert event["host_elapsed_ns"] > event["host_thread_cpu_ns"]
    assert 0 <= event["host_prepare_elapsed_ns"] <= event["host_elapsed_ns"]
    assert runner.calls == [(scheduled, (11,), {"dummy_run": False})]
    assert report["host_timing_not_GPU_completion"] and not report["CUDA_synchronization_added"]
    assert runner.execute_model(scheduled) is runner.value
    assert len(trace.events) == 1


def test_native_exception_retains_elapsed_time_and_can_disarm() -> None:
    runner = _Runner()
    trace = LatencyForwardTrace(runner)
    trace.configure(True, "control-split-first")
    runner.fail = True
    with pytest.raises(RuntimeError, match="native failure"):
        runner.execute_model(_scheduled())
    event = trace.configure(False, "")["events"][0]
    assert event["status"] == "failed" and event["host_elapsed_ns"] > 0
    assert event["host_prepare_elapsed_ns"] > 0


def test_telemetry_sample_failure_is_retained_and_thread_stops() -> None:
    attempted = threading.Event()
    count = 0

    def sample() -> dict[str, object]:
        nonlocal count
        count += 1
        if count == 2:
            attempted.set()
            raise RuntimeError("driver read failed")
        return {"sm_MHz": 2496}

    sampler = TelemetrySampler(sample, interval=0.05)
    sampler.start()
    assert attempted.wait(2)
    report = sampler.stop()
    assert report["thread_stopped"] and len(report["samples"]) == 1
    assert "driver read failed" in report["errors"][0]
    with pytest.raises(RuntimeError, match="only once"):
        sampler.start()


def test_sample_cap_is_explicit_and_never_silently_truncates() -> None:
    sampler = TelemetrySampler(lambda: {"sm_MHz": 2496}, interval=0.05, maximum_samples=2)
    sampler.start()
    assert sampler.thread is not None
    sampler.thread.join(timeout=2)
    report = sampler.stop()
    assert report["limit_reached"] and len(report["samples"]) == 2
    assert report["thread_stopped"] and not report["settings_changed"]
