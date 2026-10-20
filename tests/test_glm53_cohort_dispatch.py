"""Protect controlled admission, unchanged numeric floors and actual graph-four selection."""

from __future__ import annotations

import copy
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))
import measure_cohort_model_dispatch as client
import run_cohort_model_dispatch as controller


def snapshots(counts: dict[str, int]) -> list[dict[str, Any]]:
    """Represent the two independent workers without requiring CUDA in controller tests."""
    return [{"stable_dispatch": {"installed": True, "calls_by_tokens": counts.copy()},
             "graph_modes": {"4": "FULL"}, "retained_graphs": 2} for _ in range(2)]


@pytest.mark.parametrize("graph", [False, True])
def test_batch_proof_requires_both_prefill_and_expected_decode_shapes(graph: bool) -> None:
    before = snapshots({"2048": 42, "4": 84})
    after = snapshots({"2048": 42, "1032": 42, "4": 84 if graph else 210})
    assert client.batch_proof(before, after, 1032, 4, graph)["passed"]
    after[1]["stable_dispatch"]["calls_by_tokens"]["1032"] = 0
    after[1]["stable_dispatch"]["calls_by_tokens"]["258"] = 42
    assert not client.batch_proof(before, after, 1032, 4, graph)["passed"]


def test_graph_descriptor_alone_cannot_pass_if_eager_decode_executed() -> None:
    before = snapshots({})
    after = snapshots({"1032": 42, "4": 126})
    assert not client.batch_proof(before, after, 1032, 4, True)["passed"]


@pytest.mark.parametrize("failure", [False, True])
def test_request_is_queued_until_resume_and_resume_occurs_on_http_failure(
    monkeypatch: pytest.MonkeyPatch, failure: bool,
) -> None:
    resumed, entered = threading.Event(), threading.Event()
    paused = False
    calls = []
    observations = iter([snapshots({}), snapshots({"1032": 42, "4": 126})])

    def request(base: str, route: str, body: Any = None, timeout: int = 180) -> Any:
        nonlocal paused
        calls.append(route)
        if route.startswith("/pause?"):
            paused = True
            return {"status": "paused"}
        if route == "/resume":
            paused = False
            resumed.set()
            return {"status": "resumed"}
        if route == "/is_paused":
            return {"is_paused": paused}
        assert route == "/v1/completions" and paused
        entered.set()
        assert resumed.wait(2)
        if failure:
            raise ValueError("retained API error")
        return {"choices": []}

    monkeypatch.setattr(client, "request", request)
    monkeypatch.setattr(client, "snapshot", lambda base: next(observations))
    monkeypatch.setattr(client.time, "sleep", lambda seconds: entered.wait(2))
    body = {"prompt": [[0] * 258] * 4, "max_tokens": 4}
    if failure:
        with pytest.raises(ValueError, match="retained API error"):
            client.cohort_request(client.BASE, body, False, 1.)
    else:
        _, receipt = client.cohort_request(client.BASE, body, False, 1.)
        assert receipt["proof"]["passed"]
    assert resumed.is_set() and not paused and calls.count("/resume") == 1


def test_pause_cannot_target_original_service(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(client, "request", lambda *args, **kwargs: pytest.fail("must not send"))
    with pytest.raises(ValueError, match="isolated"):
        client.cohort_request("http://127.0.0.1:8000", {"prompt": [[0]] * 4}, False, 1.)


def test_controller_binds_capture_sizes_worker_and_independent_watchdog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = controller.runner
    for name in ("REVISION", "PREPARED_NAME", "WORKER_CLASS", "CAPTURE_SIZES", "CLIENT_SCRIPT",
                 "REPORT_NAME", "ENTRY_SCRIPT", "SCOPE"):
        monkeypatch.setattr(runner, name, copy.deepcopy(getattr(runner, name)))
    controller.configure()
    assert runner.REVISION == "cohortdispatchv2"
    assert runner.CAPTURE_SIZES == [1, 4]
    assert runner.WORKER_CLASS == "cohort_dispatch_worker.CohortDispatchWorker"
    assert runner.ENTRY_SCRIPT.name == "run_cohort_model_dispatch.py"
