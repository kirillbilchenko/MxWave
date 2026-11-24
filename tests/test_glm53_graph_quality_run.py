"""Fail-fast and frozen-input contracts for the paired graph-four quality experiment."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load the HTTP-only measurement code without a GPU or running service."""
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))
    return importlib.import_module("measure_graph_quality")


def test_original_numerical_limits_require_complete_identical_conditioning(client: ModuleType) -> None:
    """Neither a small mean nor matching greedy ids can hide missing decode coverage."""
    value = {"expected_decode_steps": 48, "decode_steps": 48,
        "all_decode_conditioning_matches": True, "all_generated_ids_match": True,
        "mean_decode_kl": 1e-5, "mean_decode_nll_difference": -1e-4}
    gate = {"mean_kl": 1e-5, "absolute_mean_nll": 1e-4}
    assert client.numerical_passed(value, gate)
    for key, wrong in (("mean_decode_kl", 1.001e-5),
                       ("mean_decode_nll_difference", -1.001e-4),
                       ("all_generated_ids_match", False),
                       ("all_decode_conditioning_matches", False), ("decode_steps", 47)):
        assert not client.numerical_passed({**value, key: wrong}, gate)


def test_repeat_failure_stops_without_task_or_timing_retry(
    client: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve the first failed repetition and avoid the expensive phases after it."""
    suite = {"strict_control_stability_gate": {"mean_kl": 1e-5, "absolute_mean_nll": 1e-4}}
    (tmp_path / "quality-protocol.json").write_text(json.dumps(suite))
    (tmp_path / "graph-quality-plan.json").write_text("{}")
    (tmp_path / "audit.json").write_text("{}")
    (tmp_path / "protocol.json").write_text(json.dumps({"quality": [[0] * 258] * 16}))
    monkeypatch.setattr(client, "validate_protocol", lambda plan, protocol: None)
    monkeypatch.setattr(client, "source_check", lambda label, audit: [])
    monkeypatch.setattr(client, "SANITY", ())
    monkeypatch.setattr(client, "rpc", lambda base, method, *args: (
        [{"status": "complete", "eager": False}] * 2 if method == "mxwave_set_eager"
        else [{"status": "passed"}] * 2))
    monkeypatch.setattr(client, "snapshot", lambda base: [
        {"graphs_enabled": True, "graph_modes": {"1": "FULL", "4": "FULL"}}] * 2)
    calls = []

    def collect(*args: Any) -> dict[str, Any]:
        calls.append(args[3])
        return {"decode": {"label": args[3]}}

    monkeypatch.setattr(client, "quality_leg", collect)
    monkeypatch.setattr(client, "compare", lambda *args: {"expected_decode_steps": 48,
        "decode_steps": 48, "all_decode_conditioning_matches": True,
        "all_generated_ids_match": True, "mean_decode_kl": 2e-5,
        "mean_decode_nll_difference": 0.0})

    def unexpected(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Expensive phase called after the unchanged repeat gate failed")

    for name in ("heldout", "tasks", "batch"):
        monkeypatch.setattr(client, name, unexpected)
    monkeypatch.setattr(sys, "argv", ["measure_graph_quality.py", "--root", str(tmp_path),
        "--label", "control", "--vocabulary", "2", "--data", str(tmp_path),
        "--cpu-audit", str(tmp_path / "audit.json")])
    with pytest.raises(RuntimeError, match="Unchanged numerical limit failed"):
        client.main()
    assert calls == ["graph-control-r0", "graph-control-r1"]
    retained = json.loads((tmp_path / "control-quality-measurement.json").read_bytes())
    assert retained["status"] == "failed" and len(retained["numerical"]) == 2
    assert retained["comparisons"][0]["passed"] is False
    assert retained["heldout"] == retained["timing"] == []


def test_prepared_protocol_tampering_fails_before_pause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bind a prospective run to both current source inventories and every protocol file."""
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))
    quality = importlib.import_module("run_quality_model")
    monkeypatch.setattr(quality.c, "ROOT", tmp_path)
    names = ("protocol.json", "quality-protocol.json", "graph-quality-plan.json")
    for name in names:
        (tmp_path / name).write_text("{}")
    import hashlib

    bindings = [{"code": {"qualified.py": "source-digest"}}] * 2
    prepared = {"bindings": bindings, "protocol_files_sha256": {
        name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() for name in names}}
    (tmp_path / "prepared.json").write_text(json.dumps(prepared))
    assert quality.check_prepared(tmp_path, tmp_path, bindings, "prepared.json") == prepared
    (tmp_path / "graph-quality-plan.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="Frozen paired-run protocol changed"):
        quality.check_prepared(tmp_path, tmp_path, bindings, "prepared.json")
    with pytest.raises(ValueError, match="source differs"):
        quality.check_prepared(tmp_path, tmp_path, [{"code": {}}] * 2, "prepared.json")
