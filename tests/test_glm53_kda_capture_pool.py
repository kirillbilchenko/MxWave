"""Warmup leases may retire only while idle; graph-owned storage must never be reassigned."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))


def test_initialization_warmup_capture_and_serving_keep_two_disjoint_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    runtime = importlib.import_module("kda_runtime")
    current = {"stream": 10, "capturing": False}
    synchronized = []

    class Pool:
        device = torch.device("cuda:0")
        shape = (2, 16)

        def data_ptr(self) -> int:
            return 99

        def record_stream(self, stream: Any) -> None:
            pass

    pool = Pool()
    monkeypatch.setattr(runtime, "_STATES", {})
    monkeypatch.setattr(runtime, "_POOLS", {0: pool})
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: SimpleNamespace(
        cuda_stream=current["stream"]))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: current["capturing"])
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: synchronized.append(device))
    assert runtime._slot(pool)[0] == 0
    current["stream"] = 20
    assert runtime._slot(pool)[0] == 1
    current["stream"] = 30
    with pytest.raises(ValueError, match="two inference"):
        runtime._slot(pool)
    assert runtime.retire_idle_eager_slots()["retired_eager_stream_leases"] == 2
    current["stream"] = 40
    assert runtime._slot(pool)[0] == 0  # Native outer capture-context warmup.
    current.update(stream=30, capturing=True)
    captured_slot = runtime._slot(pool)[0]
    assert captured_slot == 1
    with pytest.raises(RuntimeError, match="outside CUDA"):
        runtime.retire_idle_eager_slots()
    current["capturing"] = False
    retired = runtime.retire_idle_eager_slots()
    assert retired["retired_eager_stream_leases"] == 1
    assert retired["retained_captured_stream_leases"] == 1
    current["stream"] = 10
    assert runtime._slot(pool)[0] == 0  # A hole at zero must not alias pinned slot one.
    state = next(iter(runtime._STATES.values()))
    assert state.streams[30][0] == captured_slot and state.captured_streams == {30}
    assert len(synchronized) == 2 and len(state.streams) == runtime.SLOTS == 2


def test_capture_wrapper_preserves_call_and_retires_only_at_startup_boundaries() -> None:
    from kda_capture_pool import patch

    events = []

    class Manager:
        def capture(self, argument: str, *, flag: bool) -> str:
            events.append((argument, flag))
            self.graphs = {}
            return "native-result"

    def retire() -> dict[str, bool]:
        events.append("retire")
        return {"startup_only": True}

    patch(Manager, SimpleNamespace(retire_idle_eager_slots=retire))
    assert Manager().capture("model", flag=True) == "native-result"
    assert events == ["retire", ("model", True), "retire"]
