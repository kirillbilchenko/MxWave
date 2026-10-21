"""Full training cannot start from failed capture, failed retention or changed plans."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from integrations.glm53_mxfp4 import continue_drafter_full_training as continuation


@pytest.mark.parametrize("failure", ["capture", "finalizer", "plan", "restoration"])
def test_failed_capture_cannot_reach_transfer_or_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    config = tmp_path / "continuation.json"
    config.write_text(json.dumps({"remote_root": "/owned/capture", "plan_sha256": "frozen",
                                  "wait_seconds": 60}))
    plan = tmp_path / "plan.json"
    plan.write_text("{}")
    (tmp_path / "continuation-source-manifest.json").write_text(json.dumps({
        p.name: continuation.digest(p) for p in (config, plan)
    }))
    status: dict[str, Any] = {
        "phase": "complete_restored", "service_paused": False, "error": None,
        "plan_sha256": "frozen", "runner": None,
        "postprocess": {"status": "verified"},
        "restoration": {"health_200": True, "originals_running": True},
    }
    if failure == "capture":
        status.update(phase="failed_restored", error="repeat failure")
    elif failure == "finalizer":
        status["runner"] = {"step": "finalize", "returncode": 1}
    elif failure == "plan":
        status["plan_sha256"] = "changed"
    else:
        status["restoration"]["health_200"] = False
    calls = []

    def read(host: str, script: str) -> str:
        calls.append(host)
        assert len(calls) == 1, "Failed capture must not proceed to reading dataset manifests"
        return json.dumps(status)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Failed capture must not launch a subprocess")

    monkeypatch.setattr(continuation, "remote", read)
    monkeypatch.setattr(continuation.subprocess, "run", forbidden)
    monkeypatch.setattr(continuation.subprocess, "Popen", forbidden)
    with pytest.raises(ValueError):
        continuation.run(config)
    state = json.loads((tmp_path / "continuation-status.json").read_bytes())
    assert state["phase"] == "failed" and not state["sealed_test_opened"]
    assert not (tmp_path / "capture").exists() and not (tmp_path / "run").exists()


def test_dataset_manifest_must_bind_the_exact_qualified_capture(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    manifest = {"status": "verified", "production_restored": True,
                "all_rank_features_bitwise_equal": True, "sealed_test_opened": False,
                "training_performed": False, "plan_sha256": "other",
                "frozen_dataset_hashes": {}, "ordering_source_sha256": {},
                "ordering_qualification_review_sha256": "proof"}
    path.write_text(json.dumps(manifest))
    plan = {"frozen_dataset_hashes": {}, "ordering_source_sha256": {},
            "ordering_qualification_review_sha256": "proof"}
    with pytest.raises(ValueError, match="verified corrected capture"):
        continuation.verify_dataset(tmp_path, plan, continuation.digest(path), "expected")
