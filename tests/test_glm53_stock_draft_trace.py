"""Observational diagnostics preserve results and bound their retained payloads."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


def load_observer():
    """Import the standalone operational observer without a vLLM dependency."""
    folder = Path(__file__).parents[1] / "deployment/glm53"
    sys.path.insert(0, str(folder))
    try:
        spec = importlib.util.spec_from_file_location("stock_draft_trace", folder / "draft_trace_worker.py")
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(folder))


def test_observer_detaches_and_keeps_eager_control_after_capture_limit(tmp_path):
    """Capture limits must not accidentally change the execution mode mid-request."""
    module = load_observer()
    calls = []
    result = object()

    def propose(input_batch, *, is_profile=False, dummy_run=False):
        calls.append((input_batch, is_profile, dummy_run))
        return result

    def sample(*args):
        return result

    def candidates(*args):
        return result

    spec = SimpleNamespace(propose=propose, model=SimpleNamespace(compute_candidates=candidates))
    runner = SimpleNamespace(sample=sample, speculator=spec)
    for mode, expected in (("native", False), ("draft_eager", True)):
        trace = module.DraftTrace(runner, tmp_path / (mode + ".jsonl"), mode, 1)
        trace.index = 1
        trace.attach()
        assert runner.speculator.propose("input") is result
        assert calls[-1] == ("input", expected, False)
        assert runner.sample(None, None, None) is result
        assert not trace.path.exists()
        trace.detach()
        assert runner.sample is sample
        assert spec.propose is propose
        assert spec.model.compute_candidates is candidates


def test_tensor_health_catches_nonfinite_outside_preview():
    """A small preview must not hide a NaN in the remainder of a native tensor."""
    report = load_observer().tensor_report(torch.tensor([1., 2., float("nan")]), 1)
    assert report["preview"] == [1.]
    assert report["finite"] == 2
    assert report["count"] == 3
    assert "rms" not in report
