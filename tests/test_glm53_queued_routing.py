"""GPU timing must enqueue the whole leg and keep cache pressure outside every event pair."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))
from prefill_routing_queued_blocks import ITERATIONS, leg, measure


@pytest.mark.parametrize("evict", [False, True])
def test_leg_queues_all_samples_before_wait_and_excludes_pressure(
    monkeypatch: pytest.MonkeyPatch, evict: bool,
) -> None:
    events: list[str] = []

    class Event:
        def __init__(self, **kwargs: Any) -> None:
            assert kwargs == {"enable_timing": True}

        def record(self) -> None:
            events.append("record")

        def synchronize(self) -> None:
            events.append("synchronize")

        def elapsed_time(self, other: Event) -> float:
            assert events[-1] == "synchronize"
            return .5

    import prefill_routing_queued_blocks as queued

    monkeypatch.setattr(queued.torch.cuda, "Event", Event)
    operation = SimpleNamespace(replay=lambda: events.append("operation"))
    pressure = SimpleNamespace(add_=lambda value: events.append("pressure"))
    result = leg(operation, pressure, evict=evict)
    expected = (["pressure"] if evict else []) + ["operation"]
    expected += ((["pressure"] if evict else []) + ["record", "operation", "record"]) * ITERATIONS
    assert events == expected + ["synchronize"]
    assert result["times_us"] == [500.] * ITERATIONS
    assert result["untimed_primer_calls"] == 1


def test_distinct_control_graphs_are_rejected() -> None:
    with pytest.raises(ValueError, match="identical reference"):
        measure({"stock": object(), "qualified": object()}, SimpleNamespace(), evict=False)
