"""Actual decode comparisons must respect generated-token conditioning and file hashes."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


@pytest.fixture
def decoder(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    root = Path(__file__).parents[1] / "integrations/glm53_mxfp4"
    monkeypatch.syspath_prepend(str(root))
    spec = importlib.util.spec_from_file_location("glm53_decode_quality_test", root / "decode_quality.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _report(root: Path) -> dict:
    raw = np.log(np.array([.1, .2, .7], dtype="<f4")).tobytes()
    (root / "step.f32").write_bytes(raw)
    return {"status": "complete", "vocabulary": 3, "generated_tokens": 3, "contexts": [
        {"index": 0, "prompt_sha256": "fixed", "token_ids": [2, 2, 2], "steps": [
            {"position": i, "kind": "prefill" if i == 0 else "decode",
             "conditioning_token_ids": [2] * i, "file": "step.f32",
             "sha256": hashlib.sha256(raw).hexdigest()} for i in range(3)]}]}


def test_identical_actual_decode_has_zero_kl(decoder: ModuleType, tmp_path: Path) -> None:
    report = _report(tmp_path)
    result = decoder.compare(report, report, tmp_path)
    assert result["decode_steps"] == 2
    assert result["all_decode_conditioning_matches"]
    assert result["mean_decode_kl"] == 0
    assert result["mean_decode_nll_difference"] == 0


def test_changed_generation_is_not_compared_as_same_conditioning(
    decoder: ModuleType, tmp_path: Path,
) -> None:
    report = _report(tmp_path)
    candidate = copy.deepcopy(report)
    candidate["contexts"][0]["token_ids"][0] = 1
    for step in candidate["contexts"][0]["steps"][1:]:
        step["conditioning_token_ids"][0] = 1
    result = decoder.compare(report, candidate, tmp_path)
    assert result["decode_steps"] == 0
    assert result["expected_decode_steps"] == 2
    assert not result["all_decode_conditioning_matches"]
    assert not result["all_generated_ids_match"]
    assert result["mean_decode_kl"] is None


def test_changed_distribution_fails_integrity_gate(decoder: ModuleType, tmp_path: Path) -> None:
    report = _report(tmp_path)
    (tmp_path / "step.f32").write_bytes(b"bad")
    with pytest.raises(ValueError, match="changed"):
        decoder.compare(report, report, tmp_path)
