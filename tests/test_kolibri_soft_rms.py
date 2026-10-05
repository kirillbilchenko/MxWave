"""Verify compact trial reconstruction equals full softened quantization and fails closed."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from test_kolibri_mixed import _fixtures

from mxwave.calibration import save_calibration_data
from mxwave.core import quantize_mxfp4
from mxwave.routed_calibration import soften_routed_rms


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parent.parent / "scripts"))
    assembler = importlib.import_module("assemble_kolibri_mixed")
    retuner = importlib.import_module("retune_kolibri_soft_rms")
    loader = importlib.import_module("kolibri_soft_rms_trial")
    torch.manual_seed(731)
    experts, reference = _fixtures(tmp_path)
    baseline = tmp_path / "baseline"
    assembler.assemble(experts, reference, baseline)
    target = "model.layers.0.mlp.experts.0.gate_proj.weight"
    rms = torch.linspace(0.01, 1.0, 128)
    statistics = tmp_path / "rms.safetensors"
    save_calibration_data(
        statistics,
        {"rms": {target: rms}},
        {
            "policy": "kolibri1-routed-experts",
            "source_repository": "Aleph-Alpha/Kolibri-1-BF16",
            "source_revision": "7a8f290e7858825c3cf5e4c447ba68345de9f1d3",
            "num_sequences": "1",
            "sequence_length": "512",
            "num_tokens": "512",
        },
    )
    counts = tmp_path / "counts.json"
    counts.write_text(
        json.dumps(
            {
                "status": "passed",
                "file_sha256": hashlib.sha256(statistics.read_bytes()).hexdigest(),
                "layers": [{"layer": 0, "routed_observations": [128, 0]}],
            }
        )
    )
    trial = tmp_path / "trials/soft"
    retuner.retune(tmp_path / "bf16", baseline, statistics, counts, trial, device="cpu")
    return retuner, loader, baseline, statistics, counts, trial, target, rms


def test_compact_patch_equals_full_quantization_and_preserves_all_other_payloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retuner, loader, baseline, statistics, counts, trial, target, rms = _fixture(
        tmp_path, monkeypatch
    )
    before = load_file(str(baseline / "model.safetensors"))
    actual = dict(loader.composed_weights(before.items(), trial / "trial.json"))
    source = load_file(str(tmp_path / "bf16/model.safetensors"))[target]
    gamma = soften_routed_rms(rms, observations=128)
    packed, scales = quantize_mxfp4(source, gamma=gamma, method="mse", mse_clip_depth=4)
    expected = {
        target.removesuffix(".weight") + ".weight_packed": packed,
        target.removesuffix(".weight") + ".weight_scale": scales,
    }
    for name, value in actual.items():
        assert torch.equal(
            value.reshape(-1).view(torch.uint8),
            expected.get(name, before[name]).reshape(-1).view(torch.uint8),
        )
    audit = json.loads((trial / "load-audit.json").read_text())
    assert audit["status"] == "passed" and audit["verified_payloads"] == len(before)
    assert audit["patched_targets"] == 1
    with pytest.raises(FileExistsError):
        retuner.retune(tmp_path / "bf16", baseline, statistics, counts, trial, device="cpu")


@pytest.mark.parametrize("kind", ["patch_bytes", "omission", "duplicate", "counts"])
def test_compact_patch_rejects_changed_or_incomplete_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    _, loader, baseline, _, counts, trial, _, _ = _fixture(tmp_path, monkeypatch)
    weights = list(load_file(str(baseline / "model.safetensors")).items())
    if kind == "patch_bytes":
        patch = trial / "delta-00001.safetensors"
        tensors = load_file(str(patch))
        key = next(key for key in tensors if key.endswith("::packed"))
        tensors[key][0, 0] ^= 1
        save_file(tensors, str(patch))
    elif kind == "omission":
        weights.pop()
    elif kind == "duplicate":
        weights.append(weights[0])
    else:
        counts.write_text(counts.read_text() + " ")
    with pytest.raises(ValueError):
        list(loader.composed_weights(weights, trial / "trial.json"))
    assert not (trial / "load-audit.json").exists()
