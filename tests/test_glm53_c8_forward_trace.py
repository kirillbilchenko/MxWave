"""C8 attribution must preserve native calls and keep true prompt-order differences visible."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))

from c8_forward_trace import ForwardTrace, forward_signature


class _Runner:
    def __init__(self) -> None:
        self.value = object()
        self.calls: list[Any] = []
        self.fail = False
        self.descriptor = SimpleNamespace(
            cg_mode="NONE", num_tokens=4, num_reqs=2, max_query_len=2
        )

    def prepare_inputs(self, scheduled: Any, state: Any, descriptor: Any) -> Any:
        return SimpleNamespace(
            req_ids=list(scheduled.num_scheduled_tokens),
            num_reqs=2,
            num_tokens=3,
            num_tokens_after_padding=4,
            has_prefill=True,
            idx_mapping_np=np.array([3, 7]),
            prefill_len_np=np.array([2, 1]),
        )

    def execute_model(self, scheduled: Any, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((scheduled, args, kwargs))
        self.prepare_inputs(
            scheduled, None, self.descriptor,
        )
        if self.fail:
            raise RuntimeError("native failure")
        return self.value


def _scheduled() -> Any:
    return SimpleNamespace(
        total_num_scheduled_tokens=3,
        num_scheduled_tokens={"a": 2, "b": 1},
        scheduled_new_reqs=[
            SimpleNamespace(req_id="a", prompt_token_ids=[1, 2]),
            SimpleNamespace(req_id="b", prompt_token_ids=[3]),
        ],
    )


def test_native_arguments_result_and_disabled_calls_are_preserved() -> None:
    runner, scheduled = _Runner(), _scheduled()
    trace = ForwardTrace(runner)
    assert runner.execute_model(scheduled, 9, dummy_run=False) is runner.value
    assert not trace.events
    trace.configure(True, "ordinary-unprofiled-r0")
    assert runner.execute_model(scheduled, 9, dummy_run=False) is runner.value
    result = trace.configure(False, "")
    assert runner.calls == [(scheduled, (9,), {"dummy_run": False})] * 2
    assert result["events"][0]["prepared"]["num_tokens_after_padding"] == 4
    assert result["host_metadata_only"] and not result["GPU_tensors_copied"]
    assert runner.execute_model(scheduled) is runner.value
    assert len(trace.events) == 1


def test_signatures_normalize_only_transient_ids_and_state_slots() -> None:
    runner = _Runner()
    trace = ForwardTrace(runner)
    trace.configure(True, "controlled-profiled-r0")
    runner.execute_model(_scheduled())
    event = trace.configure(False, "")["events"][0]
    other = copy.deepcopy(event)
    other["prepared"]["request_state_indices"] = [8, 1]
    for row in other["prepared"]["row_order"]:
        row["request_id"] += "-new"
    assert forward_signature(event) == forward_signature(other)
    other["prepared"]["row_order"].reverse()
    assert forward_signature(event) != forward_signature(other)
    other = copy.deepcopy(event)
    other["prepared"]["num_tokens_after_padding"] += 4
    assert forward_signature(event) != forward_signature(other)


def test_native_failure_is_retained_and_observer_can_be_disarmed() -> None:
    runner = _Runner()
    trace = ForwardTrace(runner)
    trace.configure(True, "ordinary-profiled-r0")
    runner.fail = True
    with pytest.raises(RuntimeError, match="native failure"):
        runner.execute_model(_scheduled())
    report = trace.configure(False, "")
    assert report["events"][0]["status"] == "failed"
    assert "native failure" in report["events"][0]["error"]
    assert not report["enabled"]


def test_native_optional_descriptor_fields_remain_none() -> None:
    runner = _Runner()
    runner.descriptor.max_query_len = None
    runner.descriptor.num_reqs = None
    trace = ForwardTrace(runner)
    trace.configure(True, "ordinary-unprofiled-r0")
    assert runner.execute_model(_scheduled()) is runner.value
    descriptor = trace.configure(False, "")["events"][0]["prepared"]["descriptor"]
    assert descriptor["max_query_len"] is None
    assert descriptor["num_reqs"] is None
