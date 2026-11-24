"""Keep target quantization policy and shared engine settings isolated from draft remapping."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))
import mtp_draft_loader as loader
from mtp_draft_loader import (
    bf16_projection_slices,
    clone_config_for_draft,
    pad_bf16_nope_weight,
)


def test_draft_config_starts_from_source_and_cannot_mutate_target_or_input() -> None:
    class Quant:
        def __init__(self, policy: dict[str, Any]) -> None:
            self.policy = policy

        @classmethod
        def from_config(cls, policy: dict[str, Any]) -> Any:
            return cls(policy)

    target = Quant({"targets": ["language_model.model.layers.45"]})
    source = {"targets": ["model.language_model.layers.45"]}
    engine = SimpleNamespace(quant_config=target, other_settings=object())
    draft = clone_config_for_draft(engine, source)
    assert draft is not engine and draft.quant_config is not target
    assert draft.other_settings is engine.other_settings
    draft.quant_config.policy["targets"][0] = "model.layers.45"
    assert target.policy == {"targets": ["language_model.model.layers.45"]}
    assert source == {"targets": ["model.language_model.layers.45"]}


def test_nope_padding_preserves_nan_sign_and_every_original_bf16_word() -> None:
    words = torch.tensor([[0, -32768, 32704, -1], [16256, -16512, 1, 32639]], dtype=torch.int16)
    value = words.view(torch.bfloat16)
    result = pad_bf16_nope_weight(value, 2, 1)
    assert result.shape == (3, 4)
    assert torch.equal(result[:2].view(torch.int16), words)
    assert torch.equal(result[2:].view(torch.int16), torch.zeros((1, 4), dtype=torch.int16))
    assert pad_bf16_nope_weight(result, 2, 1) is result
    with pytest.raises(ValueError, match="unexpected row"):
        pad_bf16_nope_weight(torch.zeros((4, 4), dtype=torch.bfloat16), 2, 1)


@pytest.mark.parametrize("rope_rows,nope", [(0, True), (0, False), (64, True), (64, False)])
def test_projection_checks_follow_declared_rope_geometry(rope_rows: int, nope: bool) -> None:
    q = torch.ones((1536, 4), dtype=torch.bfloat16)
    kv = torch.full((512 if nope else 512 + rope_rows, 4), -1, dtype=torch.bfloat16)
    weight = torch.cat((q, kv, torch.zeros((rope_rows if nope else 0, 4), dtype=q.dtype)))
    geometry = {"q_rank": 1536, "kv_rank": 512, "rope_rows": rope_rows,
                "hidden_size": 4, "nope": nope}
    checks = bf16_projection_slices(weight, q, kv, **geometry)
    assert len(checks) == 2 + int(nope and rope_rows > 0)
    assert all(torch.equal(left.view(torch.int16), right.view(torch.int16))
               for _, left, right in checks)
    with pytest.raises(ValueError, match="declared source geometry"):
        bf16_projection_slices(weight[:-1], q, kv, **geometry)


def test_loader_report_accepts_no_padding_and_still_requires_a_full_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(loader, "_INSTALLED", True)
    monkeypatch.setattr(loader, "_CALLS", [{"separate_quant_config": True}])
    monkeypatch.setattr(loader, "_WEIGHT_LOADS", [{"source_tensors": 1756}])
    monkeypatch.setattr(loader, "_PADDING", [])
    assert loader.report(2)["bf16_nope_padding"] == []
    monkeypatch.setattr(loader, "_WEIGHT_LOADS", [])
    with pytest.raises(RuntimeError, match="loader invocation"):
        loader.report(2)
