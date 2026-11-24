"""Check diagnostic stream preservation and exception-safe scheduler resumption."""

from __future__ import annotations

import io
import json
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))

from latency_stream import split_group, stream_one


def test_stream_preserves_real_chunks_and_rejects_incomplete_usage(monkeypatch: Any) -> None:
    events = [
        {"choices": [{"token_ids": [7, 8]}]},
        {"choices": [{"token_ids": [9]}], "usage": {"prompt_tokens": 2, "completion_tokens": 3}},
    ]
    data = b"".join(b"data: " + json.dumps(row).encode() + b"\n" for row in events)
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: io.BytesIO(data))
    result = stream_one("http://unused", [1, 2], 3)
    assert result["token_ids"] == [7, 8, 9]
    assert [a["tokens"] for a in result["arrivals"]] == [2, 1]
    assert result["request_start_monotonic_ns"] <= result["arrivals"][0]["monotonic_ns"]
    assert result["arrivals"][-1]["monotonic_ns"] <= result["request_end_monotonic_ns"]
    with pytest.raises(ValueError, match="Incomplete"):
        stream_one("http://unused", [1, 2], 4)


def test_split_submission_waits_for_resume_and_preserves_request_order() -> None:
    resumed = threading.Event()
    calls: list[str] = []

    def request(base: str, path: str, body: Any = None) -> dict[str, Any]:
        calls.append(path)
        if path.startswith("/pause"):
            resumed.clear()
        if path == "/resume":
            resumed.set()
        return {"is_paused": not resumed.is_set()}

    def complete(base: str, ids: list[int], tokens: int) -> dict[str, Any]:
        assert resumed.wait(3)
        return {"token_ids": ids}

    value = split_group(
        "http://127.0.0.1:18784",
        [[i] for i in range(8)],
        1,
        request,
        grace_seconds=0.1,
        late_delay_seconds=0.02,
        complete=complete,
    )
    assert value["status"] == "complete"
    assert [r["token_ids"] for r in value["requests"]] == [[i] for i in range(8)]
    assert calls.count("/resume") == 1
    assert value["late_submission_monotonic_ns"] > value["resume_return_monotonic_ns"]


def test_early_client_failure_resumes_before_waiting_for_other_threads() -> None:
    resumed = threading.Event()
    calls: list[str] = []

    def request(base: str, path: str, body: Any = None) -> dict[str, Any]:
        calls.append(path)
        if path == "/resume":
            resumed.set()
        return {"is_paused": True}

    def complete(base: str, ids: list[int], tokens: int) -> dict[str, Any]:
        if ids == [0]:
            raise RuntimeError("client failed")
        assert resumed.wait(3)
        return {}

    with pytest.raises(RuntimeError, match="client failed"):
        split_group(
            "http://127.0.0.1:18784",
            [[i] for i in range(8)],
            1,
            request,
            grace_seconds=0.1,
            late_delay_seconds=0.02,
            complete=complete,
        )
    assert calls.count("/resume") == 1 and resumed.is_set()


def test_pause_never_targets_original_service() -> None:
    with pytest.raises(ValueError, match="isolated"):
        split_group("http://127.0.0.1:8000", [[i] for i in range(8)], 1, lambda *a: None)
