"""The diagnostic must retain independent copies without changing native state."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from integrations.glm53_mxfp4 import drafter_context_probe
from integrations.glm53_mxfp4.drafter_verification_probe import VerificationCapture, state_slots


def test_physical_slots_keep_order_and_skip_invalid_padding() -> None:
    assert state_slots(torch.tensor([[7, -1, 2, 7, 3]])) == [7, 2, 3]


def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    monkeypatch.setitem(sys.modules, "drafter_context_probe", drafter_context_probe)
    (tmp_path / "probe-marker.json").write_text(json.dumps({"active": True, "case": "fixture"}))
    prefix = "model.layers.0.self_attn"
    layer_cls = type("Glm5NextLinearAttention", (), {})
    layer = layer_cls()
    layer.prefix = prefix
    layer.kv_cache = [torch.arange(24).reshape(8, 3), torch.arange(32).reshape(8, 4)]
    metadata = SimpleNamespace(
        num_spec_decodes=1, num_accepted_tokens=torch.tensor([2]),
        spec_state_indices_tensor=torch.tensor([[3, 1, 2, -1]]),
        spec_query_start_loc=torch.tensor([0, 4]),
        non_spec_state_indices_tensor=None, non_spec_query_start_loc=None,
        spec_token_indx=torch.arange(4), non_spec_token_indx=None, has_initial_state=None,
    )
    mapping = {prefix: metadata}

    def prepare_attn(input_batch, cudagraph_mode, for_capture=False):
        return mapping

    model = SimpleNamespace(
        compute_logits=lambda x: x, named_modules=lambda: [(prefix, layer)],
    )
    runner = SimpleNamespace(
        model=model, sample=lambda h, b, g: None,
        speculator=SimpleNamespace(
            propose=lambda *args, **kwargs: None,
            query_cudagraph_manager=SimpleNamespace(run_fullgraph=lambda d: None),
        ),
        model_state=SimpleNamespace(
            prepare_attn=prepare_attn, num_accepted_tokens_gpu=torch.tensor([2]),
        ),
        req_states=SimpleNamespace(
            num_computed_tokens=SimpleNamespace(gpu=torch.tensor([32])),
            last_sampled_tokens=torch.tensor([419]),
        ),
    )
    probe = VerificationCapture(runner, tmp_path, {
        "state_capture_layers": [0], "max_state_bytes_per_frame": 1024,
    }, 0)
    batch = SimpleNamespace(num_reqs=1, idx_mapping=torch.tensor([0]))
    return probe, batch, mapping, layer


def test_state_capture_is_independent_and_returns_original_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe, batch, mapping, layer = fixture(tmp_path, monkeypatch)
    original = [t.clone() for t in layer.kv_cache]
    assert probe.prepare_attn(batch, "FULL") is mapping
    captured = probe.before["tensors"]["before_layer_0_cache_1"]
    assert torch.equal(captured, original[1][[3, 1, 2]])
    assert all(torch.equal(a, b) for a, b in zip(original, layer.kv_cache, strict=True))
    layer.kv_cache[1].zero_()
    mapping["model.layers.0.self_attn"].num_accepted_tokens.zero_()
    assert torch.equal(captured, original[1][[3, 1, 2]])
    assert probe.before["tensors"]["before_layer_0_meta_num_accepted_tokens"].item() == 2


def test_state_capture_rejects_unbounded_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe, batch, _, _ = fixture(tmp_path, monkeypatch)
    probe.plan["max_state_bytes_per_frame"] = 1
    with pytest.raises(AssertionError):
        probe.prepare_attn(batch, "FULL")
