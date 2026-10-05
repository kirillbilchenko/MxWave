"""Protect the public release boundary and already-published resume behavior."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


def _module(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("publish_kolibri_gguf")


def _fixture() -> tuple[dict[str, Any], Any]:
    module = importlib.import_module("publish_kolibri_gguf")
    prepared = {
        "files": {
            "model.gguf": {"bytes": 3, "sha256": "selected", "git_blob_sha1": None},
            "README.md": {"bytes": 4, "sha256": "card", "git_blob_sha1": "card-blob"},
        },
        "strict_screen_status": "failed",
        "repository": module.REPOSITORY,
        "model_sha256": module.MODEL_SHA,
        "source_manifest_sha256": module.MANIFEST_SHA,
        "protocol_sha256": module.PROTOCOL_SHA,
        "runtime_mmq_precision": "q8",
        "experimental": True,
    }
    info = SimpleNamespace(
        private=True,
        sha="verified-revision",
        siblings=[
            SimpleNamespace(
                rfilename="model.gguf", size=3, lfs=SimpleNamespace(sha256="selected"), blob_id=None
            ),
            SimpleNamespace(rfilename="README.md", size=4, lfs=None, blob_id="card-blob"),
        ],
    )
    return prepared, info


def test_release_verifies_lfs_payload_and_model_card(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module(monkeypatch)
    prepared, info = _fixture()
    api = SimpleNamespace(model_info=lambda *args, **kwargs: info)
    assert module._verify(api, prepared, private=True) == "verified-revision"


@pytest.mark.parametrize("damage", ["visibility", "weight", "card", "foreign", "missing"])
def test_release_rejects_remote_tampering(monkeypatch: pytest.MonkeyPatch, damage: str) -> None:
    module = _module(monkeypatch)
    prepared, info = _fixture()
    if damage == "visibility":
        info.private = False
    elif damage == "weight":
        info.siblings[0].lfs.sha256 = "another-model"
    elif damage == "card":
        info.siblings[1].blob_id = "undisclosed-quality-card"
    elif damage == "foreign":
        info.siblings.append(SimpleNamespace(rfilename="foreign.gguf"))
    else:
        info.siblings.pop(0)
    api = SimpleNamespace(model_info=lambda *args, **kwargs: info)
    with pytest.raises(ValueError):
        module._verify(api, prepared, private=True)


def test_release_rejects_foreign_resume_before_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module(monkeypatch)
    gguf = tmp_path / "gguf"
    gguf.mkdir()
    (gguf / "publication-prepared.json").write_text(json.dumps(_fixture()[0]))
    (gguf / "upload-intent.json").write_text(
        json.dumps(
            {
                "repository": module.REPOSITORY,
                "prepared_sha256": "different-profile",
            }
        )
    )
    api = SimpleNamespace(whoami=lambda: {"name": "kirillbilchenko"})
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=lambda: api))
    with pytest.raises(ValueError, match="different prepared release"):
        module.release(tmp_path)


def test_public_resume_verifies_without_uploading_or_changing_visibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module(monkeypatch)
    prepared, info = _fixture()
    info.private = False
    gguf = tmp_path / "gguf"
    gguf.mkdir()
    path = gguf / "publication-prepared.json"
    path.write_text(json.dumps(prepared))
    (gguf / "upload-intent.json").write_text(
        json.dumps(
            {
                "repository": module.REPOSITORY,
                "prepared_sha256": module._sha(path),
            }
        )
    )
    api = SimpleNamespace(
        whoami=lambda: {"name": "kirillbilchenko"},
        repo_exists=lambda *args: True,
        model_info=lambda *args, **kwargs: info,
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=lambda **kwargs: api))
    result = module.release(tmp_path)
    assert result["status"] == "published" and result["experimental"] is True
    assert result["revision"] == "verified-revision"
