"""Weight identity observation must retain outputs and expose duplicates without counting graphs."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))
from prefill_weight_audit import WeightAudit


def test_observation_passes_arguments_outputs_and_restores_exact_methods() -> None:
    runtime = SimpleNamespace(FUSED_MAX_TOKENS=8, apply=lambda layer, value: value)
    layer = SimpleNamespace(mxwave_kda_qualification={"prefix": "layer.projection"},
                            mxwave_kda_metadata=(64, 64))
    value = SimpleNamespace(numel=lambda: 1024)
    model = SimpleNamespace(forward=lambda x: runtime.apply(layer, x))
    original = (model.forward, runtime.apply)
    audit = WeightAudit(model, runtime, capture=lambda: False, compiling=lambda: False)
    audit.start()
    assert model.forward(value) is value
    result = audit.stop()
    assert (model.forward, runtime.apply) == original
    assert result["events"][0]["calls"] == [{"prefix": "layer.projection", "rows": 16,
                                             "mode": "reconstruction"}]


def test_duplicate_weights_are_visible_and_capture_calls_do_not_increment_events() -> None:
    runtime = SimpleNamespace(FUSED_MAX_TOKENS=8, apply=lambda layer, value: value)
    layer = SimpleNamespace(mxwave_kda_qualification={"prefix": "same.weight"},
                            mxwave_kda_metadata=(64, 64))
    value = SimpleNamespace(numel=lambda: 256)
    model = SimpleNamespace(forward=lambda x: runtime.apply(layer, runtime.apply(layer, x)))
    capturing = False
    audit = WeightAudit(model, runtime, capture=lambda: capturing, compiling=lambda: False)
    audit.start()
    model.forward(value)
    capturing = True
    assert model.forward(value) is value
    result = audit.stop()
    assert len(result["events"]) == 1
    assert result["events"][0]["duplicate_weight_prefixes"] == {"same.weight": 2}
    with pytest.raises(RuntimeError, match="ownership"):
        audit.stop()
