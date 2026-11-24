"""Exact-byte diagnostic comparisons must work across devices and storage layouts."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.fixture
def worker_module(monkeypatch):
    """Import only the diagnostic, without requiring the serving engine."""
    directory = Path(__file__).parents[1] / "integrations/glm53_mxfp4"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        "glm53_diagnostic_test", directory / "diagnostic_worker.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def compare(worker_module):
    """Expose the exact-byte comparison contract."""
    return worker_module.compare_tensors


def test_compare_hashes_exact_bytes_and_uses_loaded_dtype(compare):
    actual = torch.tensor([1., -2., .5], dtype=torch.bfloat16)
    report = compare(actual, actual.float())
    assert report["bits_exact"]
    assert report["loaded_sha256"] == report["source_reference_sha256"]
    digest = hashlib.sha256(actual.view(torch.uint8).numpy().tobytes()).hexdigest()
    assert report["loaded_sha256"] == digest


def test_compare_distinguishes_signed_zero_and_shape(compare):
    assert not compare(torch.tensor([0.]), torch.tensor([-0.]))["bits_exact"]
    assert not compare(torch.ones(2, 2), torch.ones(4))["bits_exact"]


def test_compare_supports_scalar_empty_and_noncontiguous_tensors(compare):
    for value in (torch.tensor(.5), torch.empty(0), torch.arange(12).reshape(3, 4).T):
        assert compare(value, value.clone())["bits_exact"]


def test_compare_preserves_fp8_raw_bits_including_reserved_values(compare):
    for dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu):
        value = torch.arange(256, dtype=torch.uint8).view(dtype)
        assert compare(value, value.clone())["bits_exact"]
        other = value.clone()
        other.view(torch.uint8)[0] = 1
        assert not compare(value, other)["bits_exact"]


@pytest.fixture
def graph_modes(monkeypatch):
    """Provide mode constants for CPU-only failure handling, with no engine imports."""
    package = ModuleType("vllm")
    config = ModuleType("vllm.config")
    config.CUDAGraphMode = SimpleNamespace(NONE="NONE", FULL="FULL")
    package.config = config
    monkeypatch.setitem(sys.modules, "vllm", package)
    monkeypatch.setitem(sys.modules, "vllm.config", config)


def test_missing_graph_interface_returns_unsupported_without_killing_worker(
    worker_module, graph_modes,
):
    worker = worker_module.DiagnosticWorker()
    worker.model_runner = SimpleNamespace()
    assert worker.mxwave_set_eager(True)["status"] == "unsupported"


def test_failed_dispatch_restores_original_graph_state(worker_module, graph_modes):
    def fail(**kwargs):
        raise RuntimeError("dispatch failed")

    manager = SimpleNamespace(_graphs_captured=True, graphs={1: object()}, dispatch=fail)
    worker = worker_module.DiagnosticWorker()
    worker.model_runner = SimpleNamespace(cudagraph_manager=manager)
    report = worker.mxwave_set_eager(True)
    assert report["status"] == "failed"
    assert "dispatch failed" in report["error"]
    assert manager._graphs_captured is True
