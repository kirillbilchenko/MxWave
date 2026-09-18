"""Tests for exact next-token divergence artifact compatibility."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


def _load_script() -> ModuleType:
    path = Path("scripts/evaluate_next_token_kl.py")
    spec = importlib.util.spec_from_file_location("evaluate_next_token_kl", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "format_name",
    ["mxwave-next-token-contexts-v1", "mxstream-next-token-contexts-v1"],
)
def test_context_loader_accepts_current_and_frozen_pre_rename_formats(
    tmp_path: Path,
    format_name: str,
) -> None:
    path = tmp_path / "contexts.json"
    document = {
        "format": format_name,
        "num_contexts": 2,
        "contexts": [{"token_ids": [1]}, {"token_ids": [2]}],
    }
    path.write_text(json.dumps(document))

    loaded, digest = _load_script()._load_context_manifest(path, 1)

    assert loaded["num_contexts"] == 1
    assert loaded["contexts"] == [{"token_ids": [1]}]
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()


def test_context_loader_rejects_unknown_format(tmp_path: Path) -> None:
    path = tmp_path / "contexts.json"
    path.write_text(json.dumps({"format": "unknown", "contexts": [{"token_ids": [1]}]}))

    with pytest.raises(ValueError, match="Unsupported context manifest"):
        _load_script()._load_context_manifest(path, None)
