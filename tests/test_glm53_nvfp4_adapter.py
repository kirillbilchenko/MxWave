"""Ownership contracts required to release dense NVFP4 scale allocations."""
from __future__ import annotations

import gc
import importlib.util
import sys
import types
import weakref
from pathlib import Path
from unittest.mock import patch

import pytest
import torch


@pytest.fixture(scope='module')
def adapter():
    """Load CPU ownership logic without importing the CUDA-only benchmark runtime."""
    stub = types.ModuleType('benchmark')
    stub.Native = object
    stub.pack = None
    path = Path(__file__).parents[1] / 'integrations/glm53_nvfp4/nv_adapter.py'
    spec = importlib.util.spec_from_file_location('test_nv_adapter_runtime', path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'benchmark': stub, spec.name: module}):
        spec.loader.exec_module(module)
        yield module


def descriptor(**values):
    """Model the installed bounded descriptor object graph, including shared nodes."""
    cls = type('Descriptor', (), {
        '__module__': 'vllm.model_executor.layers.fused_moe.config',
    })
    obj = cls()
    vars(obj).update(values)
    return obj


def test_replaces_shared_and_view_scale_owners_and_releases_raw(adapter):
    raw = torch.arange(32, dtype=torch.uint8)
    ref = weakref.ref(raw)
    replacement = torch.zeros(24, dtype=torch.uint8)
    first = descriptor(scale=raw)
    second = descriptor(scale=raw.view(4, 8))
    quant = descriptor(w1=first, w2=second)
    kernel = descriptor(quant_config=quant, cloned=descriptor(w1=second))
    kernel.cycle = kernel
    method = types.SimpleNamespace(moe_quant_config=quant, moe_kernel=kernel)
    assert adapter.replace_references(method, raw, replacement) == 2
    assert first.scale is second.scale is replacement
    del raw
    gc.collect()
    assert ref() is None


def test_preserves_unrelated_scales_and_arithmetic_metadata(adapter):
    raw = torch.arange(32, dtype=torch.uint8)
    other = raw.clone()
    global_scale = torch.tensor([2.0])
    replacement = torch.zeros(24, dtype=torch.uint8)
    quant = descriptor(w1=descriptor(scale=raw, global_scale=global_scale),
                       w2=descriptor(scale=other))
    method = types.SimpleNamespace(moe_quant_config=quant, moe_kernel=None)
    assert adapter.replace_references(method, raw, replacement) == 1
    assert quant.w1.scale is replacement
    assert quant.w1.global_scale is global_scale
    assert quant.w2.scale is other


def test_unknown_ownership_is_not_reported_as_a_success(adapter):
    raw = torch.zeros(32, dtype=torch.uint8)
    method = types.SimpleNamespace(moe_quant_config=types.SimpleNamespace(scale=raw),
                                   moe_kernel=None)
    assert adapter.replace_references(method, raw, torch.empty(24, dtype=torch.uint8)) == 0
