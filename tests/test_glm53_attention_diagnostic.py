"""Loaded-attention checks must distinguish sharded, replicated and padded rows."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file


@pytest.fixture
def diagnostic(monkeypatch):
    """Load the private checker without an installed vLLM engine."""
    folder = Path(__file__).parents[1] / "integrations/glm53_mxfp4"
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location("attention_diagnostic_test",
                                               folder / "attention_diagnostic.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_model(source: Path, rank: int) -> torch.nn.Module:
    """Create real safetensors and fused CPU BF16 parameters with TP2 source slices."""
    weights = {}
    layers = torch.nn.ModuleList()
    for layer, parts in enumerate((
        [("q_proj", 8), ("k_proj", 8), ("v_proj", 8), ("b_proj", 4),
         ("f_a_proj", 3), ("g_a_proj", 3)],
        [("q_a_proj", 5), ("kv_a_proj_with_mqa", 4)],
    )):
        slices = []
        for part, (name, rows) in enumerate(parts):
            value = (torch.arange(rows * 4).reshape(rows, 4) + layer * 128 + part * 16)
            value = value.to(torch.bfloat16)
            key = f"model.language_model.layers.{layer}.self_attn.{name}.weight"
            weights[key] = value
            if layer == 0 and part < 4:
                value = value[rank * (rows // 2):(rank + 1) * (rows // 2)]
            slices.append(value)
            if name == "kv_a_proj_with_mqa":
                slices.append(torch.zeros((2, 4), dtype=torch.bfloat16))
        weight = torch.cat(slices)
        projection = torch.nn.Linear(4, weight.shape[0], bias=False, dtype=torch.bfloat16)
        projection.weight = torch.nn.Parameter(weight, requires_grad=False)
        block = torch.nn.Module()
        block.self_attn = torch.nn.Module()
        setattr(block.self_attn, "in_proj_qkvbfg_a" if layer == 0 else "fused_qkv_a_proj",
                projection)
        layers.append(block)
    save_file(weights, source / "weights.safetensors")
    (source / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {key: "weights.safetensors" for key in weights}}))
    (source / "config.json").write_text(json.dumps({"text_config": {
        "num_hidden_layers": 2, "mla_nope": True, "qk_rope_head_dim": 2}}))
    model = torch.nn.Module()
    model.language_model = torch.nn.Module()
    model.language_model.model = torch.nn.Module()
    model.language_model.model.layers = layers
    return model


@pytest.mark.parametrize("rank", [0, 1])
def test_sharded_replicated_and_padded_rows_are_exact(diagnostic, tmp_path, rank):
    model = fixture_model(tmp_path, rank)
    report = diagnostic.check_attention_loaded(model, tmp_path, rank)
    assert report["status"] == "passed"
    assert report["seen_fused_layers"] == 2
    assert len(report["records"]) == 18
    assert all(row["bits_exact"] for row in report["records"])


def test_wrong_tp_slice_is_detected(diagnostic, tmp_path):
    model = fixture_model(tmp_path, rank=1)
    assert diagnostic.check_attention_loaded(model, tmp_path, rank=0)["status"] == "failed"


@pytest.mark.parametrize("fault", ["dtype", "scale", "replicated", "padding", "missing"])
def test_corrupt_or_misconfigured_loaded_attention_fails(diagnostic, tmp_path, fault):
    model = fixture_model(tmp_path, rank=1)
    layers = model.language_model.model.layers
    kda, mla = layers[0].self_attn.in_proj_qkvbfg_a, layers[1].self_attn.fused_qkv_a_proj
    if fault == "dtype":
        kda.weight = torch.nn.Parameter(kda.weight.float(), requires_grad=False)
    elif fault == "scale":
        kda.register_buffer("weight_scale", torch.ones(1, dtype=torch.uint8))
    elif fault == "replicated":
        kda.weight.view(torch.uint8)[-1, 0] ^= 1
    elif fault == "padding":
        mla.weight[-1].fill_(1)
    else:
        del layers[0].self_attn.in_proj_qkvbfg_a
    assert diagnostic.check_attention_loaded(model, tmp_path, rank=1)["status"] == "failed"


def test_graph_switch_is_the_exact_previously_qualified_method(diagnostic):
    assert (diagnostic.AttentionDiagnosticWorker.mxwave_set_eager
            is diagnostic.DiagnosticWorker.mxwave_set_eager)
    assert (diagnostic.AttentionDiagnosticWorker.mxwave_mhc_warmup
            is diagnostic.DiagnosticWorker.mxwave_mhc_warmup)


@pytest.mark.parametrize("rank", [0, 1])
def test_decoded_kda_view_keeps_source_row_corruption_visible(
    diagnostic, tmp_path, monkeypatch, rank,
):
    import kda_worker

    model = fixture_model(tmp_path, rank)
    projection = model.language_model.model.layers[0].self_attn.in_proj_qkvbfg_a
    decoded = torch.nn.functional.pad(projection.weight.detach().clone(), (0, 0, 0, 3))
    rows, width = projection.weight.shape
    projection.mxwave_kda_metadata = (rows + 3, width, 0, 0, 0, rows)
    projection.quant_method = object()
    projection.register_parameter("weight", None)
    package = ModuleType("mxwave_zipserv")
    package.runtime = SimpleNamespace(BUFFER_NAMES=(), PackedWeight=lambda *args: None,
                                      decompress=lambda packed: decoded)
    monkeypatch.setitem(sys.modules, "mxwave_zipserv", package)
    view = kda_worker.DecodedView(model)
    assert diagnostic.check_attention_loaded(view, tmp_path, rank)["status"] == "passed"
    decoded.view(torch.uint8)[0, 0] ^= 1
    assert diagnostic.check_attention_loaded(view, tmp_path, rank)["status"] == "failed"
    assert (kda_worker.KdaWorker.mxwave_set_eager
            is kda_worker.DiagnosticWorker.mxwave_set_eager)
    assert (kda_worker.KdaWorker.mxwave_mhc_warmup
            is kda_worker.DiagnosticWorker.mxwave_mhc_warmup)
