"""Protect the config-only native drafter correction and portable preparation."""

from __future__ import annotations

import errno
import json
import struct

import pytest

from mxwave_glm53 import drafter
from mxwave_glm53.bundle import digest


@pytest.fixture
def draft_source(tmp_path, monkeypatch):
    """Pin a small header fixture with the real mixed-precision layout."""
    source = tmp_path / "original"
    source.mkdir()
    config = {"num_hidden_layers": 6, "quantization_config": {
        "quant_method": "fp8", "ignored_layers": ["model.fc", "fc"]}}
    header = {f"layers.{layer}.{side}.kernel_projection.weight": {
        "dtype": "BF16", "shape": [1024, 4096]}
        for layer in range(6) for side in ("attention_conv", "mlp_conv")}
    header_bytes = json.dumps(header).encode()
    (source / "model.safetensors").write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes)
    (source / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    pins = {name: digest(source / name) for name in ("config.json", "model.safetensors")}
    monkeypatch.setattr(drafter, "DRAFTER_FILES", pins)
    # Construct expected names independently of the transformation under test.
    for layer in range(6):
        for side in ("attention_conv", "mlp_conv"):
            for index in (layer, layer + 45):
                config["quantization_config"]["ignored_layers"].append(
                    f"model.layers.{index}.{side}.kernel_projection")
    expected = tmp_path / "expected-config.json"
    expected.write_text(json.dumps(config, indent=2) + "\n")
    monkeypatch.setattr(drafter, "PREPARED_DRAFTER_FILES",
                        {**pins, "config.json": digest(expected)})
    return source


@pytest.mark.parametrize("cross_device", [False, True])
def test_preparation_preserves_bytes_and_relocates(draft_source, tmp_path, monkeypatch, cross_device):
    # An existing source receipt must never be hard-linked and overwritten by our receipt.
    (draft_source / "mxwave-drafter.json").write_text("unrelated source receipt")
    originals = {p.name: p.read_bytes() for p in draft_source.iterdir()}
    if cross_device:
        def no_link(*args):
            raise OSError(errno.EXDEV, "Different filesystems")
        monkeypatch.setattr(drafter.os, "link", no_link)
    output = tmp_path / "prepared"
    receipt = drafter.prepare_drafter(draft_source, output)
    assert receipt["transfers"]["model.safetensors"] == ("copy" if cross_device else "hardlink")
    assert {p.name: p.read_bytes() for p in draft_source.iterdir()} == originals
    assert (output / "model.safetensors").read_bytes() == originals["model.safetensors"]
    assert (output / "config.json").read_bytes() == (tmp_path / "expected-config.json").read_bytes()
    moved = tmp_path / "relocated"
    output.rename(moved)
    drafter.verify_drafter(moved)
    (moved / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        drafter.verify_drafter(moved)


def test_original_config_is_rejected_with_preparation_instruction(draft_source):
    with pytest.raises(ValueError, match="prepare-drafter"):
        drafter.verify_drafter(draft_source)


def test_failed_preparation_preserves_existing_output_and_is_atomic(draft_source, tmp_path):
    output = tmp_path / "prepared"
    output.mkdir()
    (output / "keep").write_text("untouched")
    with pytest.raises(FileExistsError):
        drafter.prepare_drafter(draft_source, output)
    assert (output / "keep").read_text() == "untouched"
    (draft_source / "model.safetensors").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        drafter.prepare_drafter(draft_source, tmp_path / "new-output")
    assert not (tmp_path / "new-output").exists()


def test_declared_fp8_scales_cannot_be_silently_ignored():
    config = {"num_hidden_layers": 6, "quantization_config": {
        "quant_method": "fp8", "ignored_layers": ["model.fc", "fc"]}}
    header = {"layers.0.attention_conv.kernel_projection.weight": {
        "dtype": "BF16", "shape": [1024, 4096]},
        "layers.0.attention_conv.kernel_projection.weight_scale_inv": {"dtype": "F32"}}
    with pytest.raises(ValueError, match="Unexpected quantization scales"):
        drafter.corrected_config(config, header)
    assert config["quantization_config"]["ignored_layers"] == ["model.fc", "fc"]
