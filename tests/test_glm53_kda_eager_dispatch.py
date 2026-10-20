"""Protect the experimental dispatch boundary so capture/compilation keep the original op."""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))
import kda_eager_dispatch as policy


class _CudaInput:
    device = torch.device("cuda:0")


@pytest.mark.parametrize("compiling,capturing,expected", [
    (True, False, "qualified"), (False, True, "qualified"), (False, False, "direct")])
def test_execution_policy_retains_original_capture_and_compilation(
    monkeypatch: pytest.MonkeyPatch, compiling: bool, capturing: bool, expected: str,
) -> None:
    calls = []

    def qualified(*args: Any) -> str:
        calls.append("qualified")
        return "qualified"

    def direct(*args: Any) -> str:
        calls.append("direct")
        return "direct"

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: compiling)
    monkeypatch.setattr(torch.cuda, "device", lambda device: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    monkeypatch.setattr(policy.kda_runtime, "kda_linear", qualified)
    monkeypatch.setattr(policy, "direct_linear", direct)
    assert policy.linear(_CudaInput(), [], None, 64, 64, 1, 1, 120, 64) == expected
    assert calls == [expected]


def test_invalid_mode_does_not_replace_the_qualified_apply() -> None:
    before = policy.kda_runtime.apply
    with pytest.raises(ValueError, match="Unknown KDA"):
        policy.select("uncaptured-fast")
    assert policy.kda_runtime.apply is before


def test_switch_restores_the_same_function_without_replacing_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(policy.kda_runtime, "FUSED_MAX_TOKENS", 8)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 1234)
    try:
        selected = policy.select("eager-direct")
        assert selected["mode"] == "eager-direct"
        assert selected["cuda_allocated_bytes"] == 1234
        assert not selected["weights_or_pools_replaced"]
        assert policy.kda_runtime.apply is policy._eager_apply
        policy.select("qualified")
        assert policy.kda_runtime.apply is policy._QUALIFIED_APPLY
    finally:
        policy.kda_runtime.apply = policy._QUALIFIED_APPLY
        policy._MODE = "qualified"
