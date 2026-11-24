"""The opt-in architecture metadata repair must isolate inheritance and reject drift."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pytest


@pytest.fixture
def mapping_module():
    """Load metadata logic without importing vLLM or initializing a GPU."""
    path = Path(__file__).parents[1] / "integrations/glm53_mxfp4/attention_mapping.py"
    spec = importlib.util.spec_from_file_location("glm_attention_mapping_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repair_preserves_base_class_and_other_fused_mappings(mapping_module):
    class Base:
        packed_modules_mapping: ClassVar = {"gate_up_proj": ["gate_up_proj"]}

    class Glm(Base):
        pass

    repaired = mapping_module.register_model_mapping(Glm)
    assert Base.packed_modules_mapping == {"gate_up_proj": ["gate_up_proj"]}
    assert repaired["gate_up_proj"] == ["gate_up_proj"]
    assert repaired["in_proj_qkvbfg_a"] == [
        "q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj", "g_a_proj"]
    assert mapping_module.register_model_mapping(Glm) == repaired


def test_conflicting_runtime_definition_fails_without_partial_mutation(mapping_module):
    class Glm:
        packed_modules_mapping: ClassVar = {"fused_qkv_a_proj": ["different_order"]}

    original = Glm.packed_modules_mapping
    with pytest.raises(ValueError, match="Unexpected existing"):
        mapping_module.register_model_mapping(Glm)
    assert Glm.packed_modules_mapping is original
    assert "in_proj_qkvbfg_a" not in original


def test_mtp_aliases_and_namespace_leave_base_policy_independent(mapping_module, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))
    from mtp_mapping import register_mtp_mapping

    @dataclass
    class Mapper:
        orig_to_new_prefix: dict[str, str]

    class Base:
        packed_modules_mapping: ClassVar = {"gate_up_proj": ["gate_proj", "up_proj"]}
        hf_to_vllm_mapper: ClassVar = Mapper({"model.language_model.": "language_model.model."})

    class Draft(Base):
        pass

    mapping_module.register_model_mapping(Base)
    original = Base.hf_to_vllm_mapper
    register_mtp_mapping(Base, Draft)
    assert Base.hf_to_vllm_mapper is original
    assert original.orig_to_new_prefix == {"model.language_model.": "language_model.model."}
    assert Draft.hf_to_vllm_mapper.orig_to_new_prefix == {
        "model.language_model.": "model.", "lm_head": "model.layers.45.head"}
    Draft.packed_modules_mapping["gate_up_proj"].append("draft_only")
    assert Base.packed_modules_mapping["gate_up_proj"] == ["gate_proj", "up_proj"]
