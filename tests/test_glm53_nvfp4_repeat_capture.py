"""Check diagnostic bounds, immutable snapshots and absence of arithmetic changes."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SOURCE = Path(__file__).parents[1] / "integrations/glm53_nvfp4/repeat_capture.py"
spec = importlib.util.spec_from_file_location("nv_repeat_capture_test", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = torch.nn.Identity()
        self.mlp = torch.nn.Identity()

    def hc_pre(self, x):
        return x

    hc_post = hc_fused_post_pre = hc_pre

    def forward(self, x):
        return self.mlp(self.self_attn(self.hc_pre(x))) + 1


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(256, 4)
        self.layers = torch.nn.ModuleList([Layer()])
        self.norm = torch.nn.Identity()

    def forward(self, input_ids, positions, inputs_embeds=None):
        embedded = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        return self.norm(self.layers[0](embedded))


def setup_capture(maximum=1024**2):
    model = Model()
    language = SimpleNamespace(model=model, compute_logits=lambda x: x * 2)
    capture = module.PrefillCapture(SimpleNamespace(model=language), maximum)
    return model, language, capture


def test_capture_does_not_change_result_and_clones_before_mutation(tmp_path):
    model, language, capture = setup_capture()
    ids = torch.arange(256)
    expected = model(ids, ids)
    capture.arm("test-one", focus=0)
    actual = model(ids, ids)
    assert torch.equal(expected, actual)
    logits = language.compute_logits(actual[-1:])
    expected_bytes = logits.view(torch.uint8).numpy().tobytes()
    logits.zero_()
    report = capture.finish(tmp_path / "one")
    event = next(e for e in report["events"] if e["label"] == "logits/output")
    assert (tmp_path / "one" / event["file"]).read_bytes() == expected_bytes
    assert report["forwards"] == 1 and report["logits_captured"]
    assert any(e["label"].startswith("layer-0/attention/input") for e in report["events"])
    assert not capture.tensors and not capture.enabled


def test_other_row_sizes_and_disarmed_calls_are_not_captured(tmp_path):
    model, language, capture = setup_capture()
    ids = torch.arange(4)
    capture.arm("small")
    language.compute_logits(model(ids, ids))
    report = capture.finish(tmp_path / "small")
    assert report["events"] == [] and report["bytes"] == 0


def test_precomputed_embeddings_capture_the_actual_multimodal_runner_path(tmp_path):
    model, language, capture = setup_capture()
    ids = torch.arange(256)
    capture.arm("preembedded")
    embeds = model.embed_tokens(ids)
    result = model(input_ids=None, positions=ids, inputs_embeds=embeds)
    language.compute_logits(result[-1:])
    report = capture.finish(tmp_path / "preembedded")
    assert report["forwards"] == 1 and report["logits_captured"]
    assert report["observed_model_calls"] == [
        {"rows": 256, "precomputed_embeddings": True, "has_token_ids": False}]
    assert report["events"][0]["label"] == "embedding/input/args/0"
    assert any(e["label"] == "model/input/kwargs/inputs_embeds" for e in report["events"])


def test_allocation_bound_and_rearming_validation():
    model, _, capture = setup_capture(10)
    with pytest.raises(ValueError):
        capture.arm("../unsafe")
    capture.arm("bounded")
    with pytest.raises(ValueError):
        capture.arm("again")
    with pytest.raises(RuntimeError, match="allocation bound"):
        model(torch.arange(256), torch.arange(256))
    assert capture.used <= 10
