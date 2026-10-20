"""Selective KDA policy must preserve MLA, side linears and auxiliary stacks."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def policy(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load the pure policy without vLLM, CUDA or a native library."""
    root = Path(__file__).parents[1]
    spec = importlib.util.spec_from_file_location(
        "glm53_kda_policy_test", root / "integrations/glm53_mxfp4/kda_policy.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def config() -> dict:
    """Use the reviewed public source metadata, not guessed layer-name patterns."""
    root = Path(__file__).parents[1] / "benchmarks/glm53-flash-hf-audit-2026-10-15"
    return json.loads((root / "INCModel3--GLM-5.3-Flash-MXFP4-Mixed-CT-AutoRound/"
                      "config.json").read_bytes())


def test_exact_kda_paths_and_tp2_shapes(policy: ModuleType, config: dict) -> None:
    targets = policy.targets_from_config(config)
    assert len(targets) == 68
    prefix = "language_model.model.layers.0.self_attn."
    first, output = targets[prefix + "in_proj_qkvbfg_a"], targets[prefix + "o_proj"]
    assert (first.rows, first.width, first.native_rows) == (12576, 4096, 12608)
    assert (output.rows, output.width) == (4096, 4096)
    assert first.expected_shards == frozenset(range(6))
    assert output.expected_shards == frozenset({None})
    assert not any(".layers.3." in name or ".layers.45." in name or ".mlp." in name
                   or name.endswith("f_b_proj") for name in targets)


@pytest.mark.parametrize("fault", ["truncated", "conflicting", "unknown", "unaligned"])
def test_ambiguous_or_unsupported_metadata_fails(
    policy: ModuleType, config: dict, fault: str,
) -> None:
    value = copy.deepcopy(config)
    text = value["text_config"]
    if fault == "truncated":
        text["layer_types"].pop()
    elif fault == "conflicting":
        text["linear_attn_config"]["kda_layers"].remove(0)
    elif fault == "unknown":
        text["layer_types"][0] = "some_other_attention"
    else:
        text["linear_attn_config"]["num_heads"] = 63
    with pytest.raises(ValueError):
        policy.targets_from_config(value)


def test_wrong_checkpoint_is_rejected_before_selection(
    policy: ModuleType, tmp_path: Path,
) -> None:
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="audited imported checkpoint"):
        policy.load_policy(tmp_path)
