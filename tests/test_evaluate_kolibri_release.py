"""Prevent incomplete measurements and malformed outputs from passing release checks."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file


def _script(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parent.parent / "scripts"))
    return importlib.import_module("evaluate_kolibri_release")


def test_behavior_checks_validate_complete_json_and_tool_calls(monkeypatch: pytest.MonkeyPatch):
    script = _script(monkeypatch)
    cases = {case["id"]: case for case in script._behavior_cases()}
    assert script._check(cases["math-en"], "391")
    assert not script._check(cases["math-en"], "391 is the answer. 400 is also possible.")
    assert script._check(cases["json"], '{"count": 3, "model": "kolibri"}')
    assert not script._check(cases["json"], '{"count": "3", "model": "kolibri"}')
    assert script._check(cases["tool"],
                         '<tool_call>{"name":"get_weather","arguments":{"city":"Berlin"}}'
                         '</tool_call>')
    assert not script._check(cases["tool"], '{"name":"get_weather"}')


def _measurements(script, root: Path, delta: float):
    protocol = {
        "windows": [{"id": f"{domain}-0"} for domain in ("en", "de", "code")],
        "behavior_cases": [{"id": "math-en"}], "scope": "fixture",
        "quality_gate": {"max_mean_delta_nll": 0.05, "max_domain_delta_nll": 0.1,
                         "max_mean_next_token_kl": 0.03, "require_behavior_checks": True},
    }
    script._write(root / "validation/protocol.json", protocol)
    digest = script._sha((root / "validation/protocol.json").read_bytes())
    for label, offset in (("fp8", 0), ("mse", delta)):
        report = {
            "protocol_sha256": digest, "behavior": [{"id": "math-en", "passed": True}],
            "windows": [{"id": f"{domain}-0", "domain": domain,
                         "token_ids_sha256": domain, "scored_tokens": 10,
                         "nll": 10 * (1 + offset), "mean_nll": 1 + offset}
                        for domain in ("en", "de", "code")],
        }
        script._write(root / f"measurements/{label}.json", report)
        save_file({"logprobs": torch.log_softmax(torch.tensor([[2.0] + [0.0] * 7] * 3), dim=-1)},
                  str(root / f"measurements/{label}.safetensors"))


def test_release_gate_detects_regression_and_rejects_missing_checks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    script = _script(monkeypatch)
    args = argparse.Namespace(root=str(tmp_path))
    _measurements(script, tmp_path, delta=0.02)
    script.compare(args)
    assert json.loads((tmp_path / "quality-comparison.json").read_text())["status"] == "passed"
    _measurements(script, tmp_path, delta=0.2)
    script.compare(args)
    assert json.loads((tmp_path / "quality-comparison.json").read_text())["status"] == "failed"
    candidate_path = tmp_path / "measurements/mse.json"
    candidate = json.loads(candidate_path.read_text())
    candidate["behavior"] = []
    script._write(candidate_path, candidate)
    with pytest.raises(ValueError, match="Behavior measurements"):
        script.compare(args)


def test_failed_release_screen_prevents_artifact_preparation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    _script(monkeypatch)
    publisher = importlib.import_module("publish_kolibri_release")
    for name, status in (
        ("conversion-result.json", "complete"), ("payload-verification.json", "passed"),
        ("quality-comparison.json", "failed"), ("qualification-result.json", "passed"),
    ):
        (tmp_path / name).write_text(json.dumps({"status": status}))
    with pytest.raises(ValueError, match="Release prerequisite failed: quality-comparison"):
        publisher._require_qualified(tmp_path)
    assert not (tmp_path / "checkpoint-mse").exists()


def test_experimental_release_preserves_failed_screen_and_requires_complete_checks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    script = _script(monkeypatch)
    publisher = importlib.import_module("publish_kolibri_release")
    _measurements(script, tmp_path, delta=0.02)
    script.compare(argparse.Namespace(root=str(tmp_path)))
    quality_path = tmp_path / "quality-comparison.json"
    quality = json.loads(quality_path.read_text())
    quality["mean_next_token_kl"] = 0.054
    quality["status"] = "failed"
    script._write(quality_path, quality)
    for name, status in (
        ("conversion-result.json", "complete"), ("payload-verification.json", "passed"),
        ("qualification-result.json", "failed"),
    ):
        script._write(tmp_path / name, {"status": status})
    original = quality_path.read_bytes()
    publisher._require_qualified(tmp_path, experimental=True)
    assert quality_path.read_bytes() == original
    with pytest.raises(ValueError, match="Release prerequisite failed"):
        publisher._require_qualified(tmp_path)
    quality["behavior"]["mse"] = []
    script._write(quality_path, quality)
    with pytest.raises(ValueError, match="behavior check"):
        publisher._require_qualified(tmp_path, experimental=True)
    quality["behavior"]["mse"] = [{"id": "math-en", "passed": True}]
    quality["domains"]["code"]["delta_mean_nll"] = 0.2
    script._write(quality_path, quality)
    with pytest.raises(ValueError, match="domain likelihood bound"):
        publisher._require_qualified(tmp_path, experimental=True)
