"""Ensure private preuploads cannot be adopted for a different model release."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("changed", ["manifest_sha256", "trial_specification_sha256"])
def test_release_rejects_a_preupload_for_another_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: str,
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    release = importlib.import_module("prepare_kolibri_soft_release")
    view = tmp_path / "soft-rms-publication"
    view.mkdir()
    prepared = {"repository": release.REPOSITORY, "manifest_sha256": "selected"}
    upload = {
        **prepared, "status": "weights-verified-private", "verified_shards": 32,
        "trial_specification_sha256": "selected-specification",
    }
    upload[changed] = "different-checkpoint"
    for name, value in (("publication-prepared.json", prepared),
                        ("private-weight-upload.json", upload)):
        (view / name).write_text(json.dumps(value))
    (tmp_path / "checkpoint-soft-rms-export.json").write_text(
        json.dumps({"trial_specification_sha256": "selected-specification"})
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=lambda: object()))
    staged = []
    monkeypatch.setattr(release, "stage", lambda root: staged.append(root))
    with pytest.raises(ValueError, match="verified private softer-RMS checkpoint"):
        release.stage_preuploaded(tmp_path)
    assert not staged
    assert not (view / "upload-repository.json").exists()
