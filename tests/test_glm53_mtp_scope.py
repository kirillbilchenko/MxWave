"""Keep draft scale bypass confined to the single audited MTP expert layer."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))
from mtp_pilot_support import expert_scope


@pytest.mark.parametrize("layer", [3, 22, 44])
def test_base_experts_cannot_enter_draft_bypass(layer: int) -> None:
    assert expert_scope(f"language_model.model.layers.{layer}.mlp.experts") == "base"


@pytest.mark.parametrize("prefix", ["model.layers.45.mlp.experts",
    "draft_model.model.layers.45.mtp_block.mlp.experts"])
def test_only_known_mtp_owner_can_keep_stock_scales(prefix: str) -> None:
    assert expert_scope(prefix) == "draft"


@pytest.mark.parametrize("prefix", ["unknown", "model.layers.2.mlp.experts",
    "model.layers.46.mlp.experts", "model.layers.45.mlp.shared_experts"])
def test_unknown_or_out_of_scope_expert_owner_fails_closed(prefix: str) -> None:
    with pytest.raises(ValueError):
        expert_scope(prefix)
