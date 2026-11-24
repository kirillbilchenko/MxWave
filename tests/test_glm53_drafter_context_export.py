"""Native export must preserve every unrelated byte and reject unqualified runs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from integrations.glm53_mxfp4.drafter_fp8_roundtrip import digest
from integrations.glm53_mxfp4.export_drafter_context import export_selected, patch_native


def test_native_patch_changes_only_context(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    weights = {
        "fc.weight": torch.ones(4, 8, dtype=torch.bfloat16),
        "layer.weight": torch.arange(32).reshape(4, 8).to(torch.float8_e4m3fn),
        "layer.weight_scale_inv": torch.tensor([[0.125]], dtype=torch.float32),
    }
    save_file(weights, str(source / "model.safetensors"))
    (source / "config.json").write_bytes(b'{"unchanged": true}\n')
    patch = tmp_path / "patch.safetensors"
    value = weights["fc.weight"].clone()
    value[0, 0] = 1.0078125
    save_file({"fc.weight": value}, str(patch))
    original_hash = digest(source / "model.safetensors")
    result = patch_native(source, patch, tmp_path / "export", expected_sha256=original_hash)
    assert result["tensor_count"] == 3 and result["config_bytes_identical"]
    assert [r["name"] for r in result["tensors"] if r["changed"]] == ["fc.weight"]
    assert torch.equal(load_file(str(tmp_path / "export/model.safetensors"))["fc.weight"], value)
    assert digest(source / "model.safetensors") == original_hash
    value[0, 0] = 5
    save_file({"fc.weight": value}, str(patch))
    with pytest.raises(ValueError, match="exceeds"):
        patch_native(source, patch, tmp_path / "invalid", expected_sha256=original_hash)
    assert not (tmp_path / "invalid").exists()


def test_no_export_without_selected_completed_run(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "report.json").write_text(json.dumps({
        "status": "complete", "offline_screen_pass": False,
        "original_drafter_parameters_unchanged": True,
    }))
    with pytest.raises(ValueError, match="No completed"):
        export_selected(tmp_path / "source", run, tmp_path / "export")
    assert not (tmp_path / "export").exists()
