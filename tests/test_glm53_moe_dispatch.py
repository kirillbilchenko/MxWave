"""The lighter dispatcher must preserve mutation and fake-tensor contracts."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode


@pytest.fixture
def plugin(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))
    return importlib.import_module("model_plugin")


def test_dispatch_schema_preserves_mutation_contract(plugin):
    schema = plugin.moe_operation("dispatcher")._schema
    assert not schema.returns
    writes = {a.name for a in schema.arguments if a.alias_info and a.alias_info.is_write}
    assert writes == {"output", "temporary", "locks"}
    assert plugin.moe_operation("custom") is plugin.moe_gemm
    with pytest.raises(ValueError, match="custom or dispatcher"):
        plugin.moe_operation("invalid")


def test_dispatch_fake_cuda_does_not_load_a_native_library(plugin, monkeypatch):
    def unexpected(*args):
        raise AssertionError("Fake execution must not touch a native library")

    monkeypatch.setattr(plugin, "native", unexpected)
    with FakeTensorMode():
        tensors = [torch.empty(1, device="cuda") for _ in range(10)]
        args = (*tensors, 4, 1024, 4096, 16, 8, 8, False, 64, 128, 1, 1)
        assert plugin.moe_operation("dispatcher")(*args) is None
        assert plugin.moe_operation("custom")(*args) is None


def test_dispatch_rejects_actual_cpu_inputs(plugin):
    tensors = [torch.empty(1) for _ in range(10)]
    args = (*tensors, 4, 1024, 4096, 16, 8, 8, False, 64, 128, 1, 1)
    with pytest.raises(NotImplementedError):
        plugin.moe_operation("dispatcher")(*args)
