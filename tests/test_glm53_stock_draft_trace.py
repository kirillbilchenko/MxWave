"""Observational diagnostics preserve results and bound their retained payloads."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
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


def test_config_preserves_weight_precision_and_declares_native_layer_aliases():
    """Only existing BF16 conv projections may be excluded from FP8 loading."""
    folder = Path(__file__).parents[1] / "deployment/glm53"
    spec = importlib.util.spec_from_file_location("drafter_compat", folder / "prepare_drafter_compat.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = {"num_hidden_layers": 6,
              "quantization_config": {"quant_method": "fp8", "ignored_layers": ["model.fc", "fc"]}}
    header = {f"layers.{i}.{s}.kernel_projection.weight": {"dtype": "BF16"}
              for i in range(6) for s in ("attention_conv", "mlp_conv")}
    result, added = module.corrected_config(config, header, 45)
    assert config["quantization_config"]["ignored_layers"] == ["model.fc", "fc"]
    assert len(added) == 24
    assert "model.layers.50.mlp_conv.kernel_projection" in added
    assert len(result["quantization_config"]["ignored_layers"]) == 26
    header["layers.0.attention_conv.kernel_projection.weight_scale_inv"] = {"dtype": "F32"}
    with pytest.raises(ValueError, match="Unexpected quantization scales"):
        module.corrected_config(config, header, 45)
