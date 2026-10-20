"""The focused decode observer must retain native selection without modifying its contract."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from integrations.glm53_mxfp4.drafter_component_probe import ComponentTrace
from integrations.glm53_mxfp4.drafter_long_decode_probe import LongDecodeTrace


def test_decode_selection_observation_preserves_native_return_and_copies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ComponentTrace, "install", lambda *args: None)
    scores = torch.arange(8 * 600, dtype=torch.float32).reshape(8, 600)
    bounds = torch.full((8,), 600, dtype=torch.int32)
    output = torch.full((8, 512), -1, dtype=torch.int32)
    sentinel = object()
    calls = []

    def native(*args: Any) -> object:
        calls.append(args)
        output.copy_(torch.arange(512).expand(8, -1))
        return sentinel

    monkeypatch.setattr(torch.ops._C, "top_k_per_row_decode", native, raising=False)
    trace = LongDecodeTrace(capturing=lambda: True)
    trace.active_layer = 3
    trace.install(None, None)
    assert torch.ops._C.top_k_per_row_decode(scores, 8, bounds, output, 8, 600, 1, 512) is sentinel
    assert len(calls) == 1
    assert [e["name"] for e in trace.events] == ["layer3.native_decode_topk#0/input", "layer3.native_decode_topk#0/output"]
    copied = trace.events[-1]["tensors"][0]["value"]
    output.fill_(-1)
    assert torch.equal(copied[0], torch.arange(512, dtype=torch.int32))
    size = trace.nbytes
    trace.record("scratch", "workspace", torch.ones(32))
    assert trace.nbytes == size
    trace.restore()
    assert torch.ops._C.top_k_per_row_decode is native
