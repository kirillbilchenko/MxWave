"""Verify physical exports match the measured patch composition and fail closed."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from test_kolibri_soft_rms import _fixture


def test_physical_soft_export_matches_every_measured_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, loader, baseline, _, _, trial, _, _ = _fixture(tmp_path, monkeypatch)
    exporter = importlib.import_module("export_kolibri_soft_rms")
    output = tmp_path / "export"
    before = load_file(str(baseline / "model.safetensors"))
    expected = dict(loader.composed_weights(before.items(), trial / "trial.json"))
    result = exporter.export(trial / "trial.json", output)
    actual = load_file(str(output / "model.safetensors"))
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        assert torch.equal(
            actual[name].reshape(-1).view(torch.uint8),
            value.reshape(-1).view(torch.uint8),
        )
    assert result["status"] == "complete" and result["verified_payloads"] == len(expected)
    assert (output / "config.json").read_bytes() == (baseline / "config.json").read_bytes()
    manifest = json.loads((output / "mxwave-manifest.json").read_text())
    assert manifest["activation_calibration"]["covered_targets"] == 1
    assert (
        manifest["payload_sha256"]
        == json.loads((trial / "trial.json").read_text())["effective_payload_sha256"]
    )
    with pytest.raises(FileExistsError):
        exporter.export(trial / "trial.json", output)


@pytest.mark.parametrize("kind", ["hash", "coverage"])
def test_soft_export_rejects_corrupted_measured_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    _, _, baseline, _, _, trial, _, _ = _fixture(tmp_path, monkeypatch)
    exporter = importlib.import_module("export_kolibri_soft_rms")
    if kind == "hash":
        path = baseline / "model.safetensors"
        data = bytearray(path.read_bytes())
        data[-1] ^= 1
        path.write_bytes(data)
    else:
        path = baseline / "config.json"
        config = json.loads(path.read_text())
        config["quantization_config"]["config_groups"] = {}
        config["quantization_config"]["ignore"] = []
        path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        exporter.export(trial / "trial.json", tmp_path / "bad-export")
    assert not (tmp_path / "bad-export/mxwave-manifest.json").exists()


def test_soft_export_resume_rejects_a_changed_completed_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, _, _, trial, _, _ = _fixture(tmp_path, monkeypatch)
    exporter = importlib.import_module("export_kolibri_soft_rms")
    output = tmp_path / "export"
    exporter.export(trial / "trial.json", output)
    (output / "mxwave-manifest.json").unlink()
    path = output / "model.safetensors"
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    assert hashlib.sha256(path.read_bytes()).hexdigest() != expected
    with pytest.raises(ValueError, match="Previously exported shard changed"):
        exporter.export(trial / "trial.json", output)
