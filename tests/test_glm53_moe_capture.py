"""Ensure post-measurement operand capture cannot silently retain overwritten model views."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import torch


class Kernel:
    """Represent the public routed-expert contract without CUDA or model weights."""

    def apply(self, hidden_states: torch.Tensor, w1: torch.Tensor,
              topk_ids: torch.Tensor, topk_weights: torch.Tensor) -> torch.Tensor:
        """Return an independently allocated expert result."""
        return hidden_states * 2


def test_capture_owns_copies_filters_other_layers_and_restores(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))
    module = importlib.import_module("moe_capture")
    weight = torch.zeros(1)
    original = Kernel.apply
    capture = module.LayerCapture(Kernel, weight.data_ptr(), capturing=lambda: False)
    x = torch.ones(1, 4096, dtype=torch.bfloat16)
    ids = torch.arange(8, dtype=torch.int32).reshape(1, 8)
    weights = torch.full((1, 8), 0.125)
    try:
        Kernel().apply(x, weight.clone(), ids, weights)
        assert not capture.rows
        output = Kernel().apply(x, weight, ids, weights)
        x.zero_()
        output.zero_()
        assert capture.rows[1]["tensors"]["hidden_states"].sum() == 4096
        assert capture.rows[1]["tensors"]["output"].sum() == 8192
        report = capture.retain(tmp_path / "captured")
        assert report["status"] == "incomplete"
        assert report["python_wrapper_removed"] and Kernel.apply is original
        assert not capture.rows
    finally:
        capture.restore()


def test_capture_rejects_graph_instrumentation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))
    module = importlib.import_module("moe_capture")
    weight = torch.zeros(1)
    capture = module.LayerCapture(Kernel, weight.data_ptr(), capturing=lambda: True)
    try:
        with pytest.raises(RuntimeError, match="post-measurement eager"):
            Kernel().apply(torch.zeros(1, 4096), weight, torch.zeros(1, 8), torch.zeros(1, 8))
        assert not capture.rows
    finally:
        capture.restore()
