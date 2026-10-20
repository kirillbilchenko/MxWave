"""Boundary copy ordering, restoration, shared-buffer bounds and exact routing evidence."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

import pytest
import torch


@pytest.fixture
def capture(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load the independent adapter without importing a vLLM worker."""
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))
    return importlib.import_module("decode_boundary_capture")


def test_copies_keep_the_values_at_each_boundary_and_remove_methods(capture: ModuleType) -> None:
    """A later overwrite must not corrupt the earlier recorded operand."""
    model, ops, _, _ = importlib.import_module("decode_boundary_fixture_model").make_model(
        torch.device("cpu"), True)
    original = model.layers[3].forward
    trace = capture.BoundaryCapture(capturing=lambda: True)
    trace.install(model, ops)
    ids = torch.arange(4, dtype=torch.int32)
    output = model(ids, ids)
    first = next(e for e in trace.events if e["name"] == "layer3.moe_gemm#0/input")
    gold = first["tensors"][0]["value"].clone()
    output.fill_(100)
    assert torch.equal(first["tensors"][0]["value"], gold)
    assert trace.locks and trace.nbytes <= capture.MAX_CAPTURE_BYTES
    trace.restore()
    assert model.layers[3].forward == original and not trace.methods
    assert trace.metadata()["python_wrappers_removed"]


def test_trace_ignores_unobserved_eager_calls(capture: ModuleType) -> None:
    """Python preparation and prefill must not become the evidence for a graph replay."""
    model, ops, _, _ = importlib.import_module("decode_boundary_fixture_model").make_model(
        torch.device("cpu"), False)
    trace = capture.BoundaryCapture(capturing=lambda: False)
    trace.install(model, ops)
    ids = torch.arange(4, dtype=torch.int32)
    model(ids, ids)
    trace.restore()
    assert not trace.events and trace.nbytes == 0


def test_shared_topk_storage_records_only_four_live_rows(capture: ModuleType) -> None:
    """The shared scheduler-sized output buffer must not consume the capture budget."""
    trace = capture.BoundaryCapture()
    value = torch.arange(2048 * 128, dtype=torch.int32).reshape(2048, 128)
    trace._topk_storages.add(value.untyped_storage().data_ptr())
    trace.record("indexer/output", "result", value)
    saved = trace.events[0]["tensors"][0]["value"]
    assert saved.shape == (4, 128) and torch.equal(saved, value[:4])


def test_bit_metrics_detect_routing_changes_without_float_conversion(capture: ModuleType) -> None:
    """Large integer IDs must retain differences that a float cast could hide."""
    a = torch.tensor([2**40, 2**40 + 1], dtype=torch.int64)
    b = torch.tensor([2**40, 2**40], dtype=torch.int64)
    result = capture.boundary_metrics(a, b)
    assert result["metric_kind"] == "bytes" and not result["bytes_exact"]
    assert result["changed_elements_by_bits"] == 1
