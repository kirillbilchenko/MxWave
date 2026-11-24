"""Reject the padding/count regression and independently bind live checks to source bytes."""

from __future__ import annotations

import copy
import importlib
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def verifier(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Import the CPU verifier without loading vLLM or constructing a model."""
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))
    return importlib.import_module("verify_mtp_model")


def draft_check(nope: bool) -> tuple[dict, dict[str, str]]:
    """Represent the live checker contract; synthetic records are not GPU proof."""
    hashes = {"q": "a" * 64, "kv_source": "b" * 64}
    records = []
    for expert in (0, 127, 287):
        for name in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"):
            records.append({"name": f"expert{expert}.{name}"})
    for projection in ("gate_up_proj", "down_proj"):
        for key in ("weight", "weight_scale"):
            records.append({"name": "model.layers.45.mtp_block.mlp.shared_experts."
                            + projection + "." + key})
    for suffix, rows in (("q", 1536), ("kv_source", 512)):
        records.append({"name": "model.layers.45.mtp_block.self_attn.fused_qkv_a_proj." + suffix,
            "shape": [rows, 4096], "dtype": "torch.bfloat16",
            "loaded_sha256": hashes[suffix], "source_reference_sha256": hashes[suffix]})
    for row in records:
        row.update(bits_exact=True)
        row.setdefault("loaded_sha256", "c" * 64)
        row.setdefault("source_reference_sha256", "c" * 64)
    return {"status": "passed", "draft_loaded": True, "records": records,
        "projection_geometry": {"q_rank": 1536, "kv_rank": 512,
            "rope_rows": 0, "hidden_size": 4096, "nope": nope}}, hashes


@pytest.mark.parametrize("nope", [False, True])
def test_zero_rope_needs_two_projection_checks_for_either_nope_flag(
    verifier: ModuleType, nope: bool,
) -> None:
    check, hashes = draft_check(nope)
    verifier.validate_draft_check(check, 2, hashes)


@pytest.mark.parametrize("change", ["extra_empty_check", "padded_geometry", "missing_projection"])
def test_reject_the_old_nineteen_check_and_padding_assumptions(
    verifier: ModuleType, change: str,
) -> None:
    check, hashes = draft_check(True)
    if change == "extra_empty_check":
        check["records"].append(copy.deepcopy(check["records"][-1]))
    elif change == "padded_geometry":
        check["projection_geometry"]["rope_rows"] = 64
    else:
        check["records"].pop()
    with pytest.raises(ValueError, match="eighteen"):
        verifier.validate_draft_check(check, 2, hashes)


def test_equal_live_reference_hashes_must_also_match_independent_cpu_source(
    verifier: ModuleType,
) -> None:
    check, hashes = draft_check(True)
    check["records"][-1].update(loaded_sha256="d" * 64, source_reference_sha256="d" * 64)
    with pytest.raises(ValueError, match="actual full CPU source"):
        verifier.validate_draft_check(check, 2, hashes)


@pytest.mark.parametrize("change", ["source_count", "padding_event", "target_policy"])
def test_native_load_requires_complete_source_isolation_and_no_padding(
    verifier: ModuleType, change: str,
) -> None:
    bridge = {"status": "passed", "bf16_nope_padding": [], "calls": [{
        "target_policy_unchanged": True, "separate_quant_config": True,
        "source_config_sha256": "194d290ebb0f19aedc60d24ea9e236e33c86616e460f4a108a842f6fd5b8b198"}],
        "weight_loads": [{"source_tensors": 1756,
            "all_declared_nonvocabulary_parameters_loaded": True, "full_source_count_required": True}]}
    item = {"depth": 2, "storage": {"base_scale_layers": 42,
            "draft_modules": ["fixture"], "draft_loader": bridge}}
    verifier.validate_loader_snapshot(item, 2)
    if change == "source_count":
        bridge["weight_loads"][0]["source_tensors"] = 1755
    elif change == "padding_event":
        bridge["bf16_nope_padding"].append({"padded_rows": 64})
    else:
        bridge["calls"][0]["target_policy_unchanged"] = False
    with pytest.raises(ValueError):
        verifier.validate_loader_snapshot(item, 2)
