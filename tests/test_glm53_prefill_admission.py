"""The idle-only window must preserve native queue order, deadline and decode latency."""

from __future__ import annotations

import queue
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))
from prefill_admission import AdmissionWindow


def request(tokens: int = 256) -> Any:
    """Represent only fields inspected by the independent collection policy."""
    return SimpleNamespace(num_prompt_tokens=tokens, num_computed_tokens=0,
                           num_output_tokens=0, pooling_params=None)


class Engine:
    """Use native-shaped state and an explicitly controlled input queue."""

    def __init__(self, pending: list[Any] | None = None) -> None:
        self.scheduler = SimpleNamespace(waiting=[request()] if pending is None else pending,
            running=[], skipped_waiting=[], connector=None, ec_connector=None,
            num_waiting_for_streaming_input=0, max_num_running_reqs=8,
            max_num_scheduled_tokens=2048)
        self.batch_queue: list[Any] = []
        self.input_queue: Any = queue.Queue()
        self.paused = False
        self.handled: list[Any] = []

    def is_running(self) -> bool:
        return True

    def is_scheduler_paused(self) -> bool:
        return self.paused

    def _handle_client_request(self, kind: str, value: Any) -> None:
        self.handled.append((kind, value))
        if kind == "add":
            self.scheduler.waiting.append(value)


def test_fixed_deadline_collects_in_order_and_never_restarts_for_new_arrivals() -> None:
    engine = Engine()
    now = 0.
    deadlines = []

    class Input:
        def get(self, timeout: float) -> Any:
            nonlocal now
            deadlines.append(timeout)
            if timeout < .004:
                now += timeout
                raise queue.Empty
            now += .004
            return "add", request()

    engine.input_queue = Input()
    window = AdmissionWindow(10.)
    window.collect(engine, was_idle=True, add_type="add", clock=lambda: now)
    assert len(engine.handled) == 2
    assert deadlines == pytest.approx([.010, .006, .002])
    assert window.events[0]["additional_messages"] == 2
    assert not window.events[0]["deadline_extended"]


@pytest.mark.parametrize("condition", ["already_running", "inflight", "paused", "resumed",
                                      "pooling", "no_waiting", "not_idle"])
def test_decode_or_ineligible_work_does_not_wait(condition: str) -> None:
    engine = Engine()
    if condition == "already_running":
        engine.scheduler.running.append(request())
    elif condition == "inflight":
        engine.batch_queue.append(object())
    elif condition == "paused":
        engine.paused = True
    elif condition == "resumed":
        engine.scheduler.waiting[0].num_computed_tokens = 128
    elif condition == "pooling":
        engine.scheduler.waiting[0].pooling_params = object()
    elif condition == "no_waiting":
        engine.scheduler.waiting.clear()
    window = AdmissionWindow(10.)
    window.collect(engine, was_idle=condition != "not_idle", add_type="add",
                   clock=lambda: pytest.fail("No clock or blocking call for ineligible work"))
    assert window.events == []


@pytest.mark.parametrize("pending", [[request(2048)], [request()] * 8])
def test_full_token_or_request_budget_cannot_pay_for_an_unusable_wait(pending: list[Any]) -> None:
    engine = Engine(pending)
    window = AdmissionWindow(10.)
    window.collect(engine, was_idle=True, add_type="add", clock=lambda: 0.)
    assert window.events[0]["additional_messages"] == 0
    assert window.events[0]["end_reason"].endswith("budget_full")


def test_control_message_is_handled_immediately_and_later_input_is_preserved() -> None:
    engine = Engine()
    engine.input_queue.put(("abort", "r0"))
    engine.input_queue.put(("add", request()))
    window = AdmissionWindow(10.)
    window.collect(engine, was_idle=True, add_type="add", clock=lambda: 0.)
    assert engine.handled == [("abort", "r0")]
    assert engine.input_queue.qsize() == 1
    assert window.events[0]["end_reason"] == "control_or_abort"


@pytest.mark.parametrize("invalid", [-1., 10.01, float("nan"), float("inf"), True])
def test_unbounded_windows_are_rejected(invalid: float) -> None:
    with pytest.raises(ValueError, match="between zero and ten"):
        AdmissionWindow().configure(invalid)


def test_zero_window_preserves_queued_input_and_records_control() -> None:
    engine = Engine()
    engine.input_queue.put(("add", request()))
    window = AdmissionWindow()
    window.collect(engine, was_idle=True, add_type="add", clock=lambda: 0.)
    assert engine.handled == [] and engine.input_queue.qsize() == 1
    assert window.events[0]["elapsed_seconds"] == 0.
