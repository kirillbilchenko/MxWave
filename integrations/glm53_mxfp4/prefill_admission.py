"""Bound an idle prefill's input-collection window without changing the native scheduler."""

from __future__ import annotations

import functools
import hashlib
import inspect
import math
import queue
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CORE_SHA = "86b8f3b3826504549ef8bea2e7fbf7553728339803abcb3e7d051d146963ff9c"
SCHEDULER_SHA = "4c38a32c7405eb95eb9dd3b3d04cbfe5d0cb4ebc0b18efbaa4adc68c7a9bca5a"
MAX_WINDOW_MS = 10.0
_INSTALLED = False


def pending_shape(engine: Any) -> tuple[int, int] | None:
    """Describe only fresh text prefills with no running/in-flight/paused work."""
    scheduler = engine.scheduler
    if (not engine.is_running() or engine.is_scheduler_paused() or engine.batch_queue
            or scheduler.running or scheduler.skipped_waiting
            or scheduler.connector is not None or scheduler.ec_connector is not None
            or scheduler.num_waiting_for_streaming_input):
        return None
    pending = list(scheduler.waiting)
    if not pending or any(r.num_computed_tokens != 0 or r.num_output_tokens != 0
                          or r.num_prompt_tokens < 1 or r.pooling_params is not None
                          for r in pending):
        return None
    return len(pending), sum(r.num_prompt_tokens for r in pending)


@dataclass
class AdmissionWindow:
    """Retain a bounded observation for each idle cohort, including the zero-wait control."""

    window_ms: float = 0.0
    events: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.configure(self.window_ms)

    def configure(self, window_ms: float) -> None:
        """Validate an explicit finite window and reset observations between closed groups."""
        if isinstance(window_ms, bool) or not math.isfinite(window_ms) \
                or not 0 <= window_ms <= MAX_WINDOW_MS:
            raise ValueError("Admission window must be finite and between zero and ten milliseconds")
        self.window_ms = float(window_ms)
        self.events.clear()

    def collect(
        self, engine: Any, *, was_idle: bool, add_type: Any,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Keep native message handling and stop at the original, never-extended deadline."""
        shape = pending_shape(engine) if was_idle else None
        if shape is None:
            return
        if len(self.events) >= 4096:
            raise RuntimeError("Admission observation bound exceeded")
        before = shape
        started = clock()
        deadline = started + self.window_ms / 1000
        messages = 0
        reason = "deadline"
        scheduler = engine.scheduler
        while self.window_ms > 0:
            shape = pending_shape(engine)
            if shape is None:
                reason = "state_changed"
                break
            if shape[0] >= scheduler.max_num_running_reqs:
                reason = "request_budget_full"
                break
            if shape[1] >= scheduler.max_num_scheduled_tokens:
                reason = "token_budget_full"
                break
            remaining = deadline - clock()
            if remaining <= 0:
                break
            try:
                message = engine.input_queue.get(timeout=remaining)
            except queue.Empty:
                break
            engine._handle_client_request(*message)
            messages += 1
            if message[0] != add_type:
                reason = "control_or_abort"
                break
        final = pending_shape(engine)
        self.events.append({"window_ms": self.window_ms, "initial_requests": before[0],
            "initial_prompt_tokens": before[1], "final_requests": None if final is None else final[0],
            "final_prompt_tokens": None if final is None else final[1],
            "additional_messages": messages, "elapsed_seconds": max(0., clock() - started),
            "end_reason": reason, "deadline_extended": False})

    def report(self) -> dict[str, Any]:
        """Expose observations outside timing without inferring a serving speed gain."""
        return {"window_ms": self.window_ms, "events": list(self.events),
                "model_or_scheduler_arithmetic_changed": False,
                "only_idle_fresh_text_prefills": True}


def install() -> None:
    """Wrap the pinned input drain and an experiment-only control RPC in disposable engines."""
    global _INSTALLED
    if _INSTALLED:
        return
    from vllm.v1.core.sched import scheduler
    from vllm.v1.engine import EngineCoreRequestType, core

    for module, expected in ((core, CORE_SHA), (scheduler, SCHEDULER_SHA)):
        if hashlib.sha256(Path(inspect.getfile(module)).read_bytes()).hexdigest() != expected:
            raise ValueError("Admission experiment requires the audited pinned runtime")
    original_drain = core.EngineCoreProc._process_input_queue
    original_rpc = core.EngineCore.collective_rpc

    def state(engine: Any) -> AdmissionWindow:
        if not hasattr(engine, "_mxwave_admission_window"):
            parallel = engine.vllm_config.parallel_config
            if parallel.data_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
                raise ValueError("This input-queue experiment supports only one DP/PP engine")
            engine._mxwave_admission_window = AdmissionWindow()
        return engine._mxwave_admission_window

    @functools.wraps(original_drain)
    def drain(engine: Any) -> None:
        was_idle = not engine.has_work()
        original_drain(engine)
        state(engine).collect(engine, was_idle=was_idle, add_type=EngineCoreRequestType.ADD)

    @functools.wraps(original_rpc)
    def rpc(engine: Any, method: Any, timeout: float | None = None, args: tuple = (),
            kwargs: dict[str, Any] | None = None) -> Any:
        if method not in ("mxwave_admission_configure", "mxwave_admission_report"):
            return original_rpc(engine, method, timeout, args, kwargs)
        value = state(engine)
        if kwargs:
            raise ValueError("Admission control does not accept keyword arguments")
        if method == "mxwave_admission_configure":
            if len(args) != 1 or engine.has_work() or not engine.input_queue.empty():
                raise RuntimeError("Configure admission only between closed, idle request groups")
            value.configure(args[0])
        elif args:
            raise ValueError("Admission report does not accept arguments")
        return [{**value.report(), "engine_idle": not engine.has_work(),
                 "engine_core_sha256": CORE_SHA, "scheduler_sha256": SCHEDULER_SHA}]

    core.EngineCoreProc._process_input_queue = drain
    core.EngineCore.collective_rpc = rpc
    _INSTALLED = True
