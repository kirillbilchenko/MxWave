"""A deliberate speed-repeat stop must not weaken the trace restoration gate."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Any

import pytest

from integrations.glm53_mxfp4.launch_trace_after_timing_stop import (
    digest,
    run,
    verify_archive,
    verify_stop,
)


def stopped() -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the permitted stop with restored original service."""
    state = {
        "pid": 10, "start_ticks": "20", "phase": "failed_restored", "service_paused": False,
        "error": "RuntimeError('Diagnostic interrupted by signal 15')",
        "restoration": {
            "health_200": True, "originals_running": True,
            "models": {"data": [{"id": "glm-5.3-flash", "max_model_len": 500000}]},
        },
    }
    request = {
        "status": "baseline_repeat_stopped_for_user_priority",
        "parent_plan_sha256": "parent", "trace_plan_sha256": "trace",
        "controller_pid": 10, "controller_start_ticks": "20", "phase_at_stop": "starting_a2",
        "data_deletion_requested": False, "training_started": False, "speed_qualified": False,
        "completed_measurements": dict.fromkeys(("a1", "b1", "c1"), "sha"),
    }
    return state, request


def test_explicit_priority_stop_with_restoration_is_accepted() -> None:
    state, request = stopped()
    verify_stop(state, request, "parent", "trace")


@pytest.mark.parametrize("field,value", [
    ("status", "failed"), ("parent_plan_sha256", "changed"), ("trace_plan_sha256", "changed"),
    ("controller_pid", 11), ("controller_start_ticks", "21"), ("phase_at_stop", "measuring_c1"),
    ("completed_measurements", {"a1": "sha", "b1": "sha"}),
    ("data_deletion_requested", True), ("training_started", True), ("speed_qualified", True),
])
def test_unrelated_stop_cannot_launch(field: str, value: Any) -> None:
    state, request = stopped()
    request[field] = value
    with pytest.raises(ValueError, match="priority stop"):
        verify_stop(state, request, "parent", "trace")


@pytest.mark.parametrize("field,value", [
    ("phase", "restoring_original_production"), ("service_paused", True),
    ("service_paused", None), ("error", "RuntimeError('GPU error')"),
])
def test_unexpected_failure_cannot_launch(field: str, value: Any) -> None:
    state, request = stopped()
    state[field] = value
    with pytest.raises(ValueError, match="restoration handler"):
        verify_stop(state, request, "parent", "trace")


@pytest.mark.parametrize("field,value", [
    ("health_200", False), ("originals_running", False),
    ("models", {"data": [{"id": "glm-5.3-flash", "max_model_len": 8192}]}),
])
def test_wrong_service_cannot_launch(field: str, value: Any) -> None:
    state, request = stopped()
    state["restoration"][field] = value
    with pytest.raises(ValueError, match="500k"):
        verify_stop(state, request, "parent", "trace")


def archive(tmp_path: Path, changed_payload: bool = False) -> dict[str, str]:
    """Build a timing archive with its independent manifest and receipt."""
    payloads = {f"{arm}-measurement.json": arm.encode() for arm in ("a1", "b1", "c1")}
    files = [{"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
             for name, data in payloads.items()]
    completed = {r["path"].split("-")[0]: r["sha256"] for r in files}
    payloads["evidence-manifest.json"] = json.dumps({"files": files}).encode()
    if changed_payload:
        payloads["c1-measurement.json"] = b"changed"
    path = tmp_path / "evidence-v1.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        for name, data in payloads.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    (tmp_path / "archive-receipt.json").write_text(json.dumps({
        "sha256": digest(path), "bytes": path.stat().st_size,
    }))
    return completed


def test_archive_verifies_retained_arm_hashes(tmp_path: Path) -> None:
    completed = archive(tmp_path)
    assert verify_archive(tmp_path, completed)["payloads"] == 3
    completed["c1"] = "changed"
    with pytest.raises(ValueError, match="Completed timing arm"):
        verify_archive(tmp_path, completed)


def test_archive_manifest_rejects_changed_payload(tmp_path: Path) -> None:
    completed = archive(tmp_path, changed_payload=True)
    with pytest.raises(ValueError, match="payloads differ"):
        verify_archive(tmp_path, completed)


def test_archive_receipt_rejects_changed_archive(tmp_path: Path) -> None:
    completed = archive(tmp_path)
    with (tmp_path / "evidence-v1.tar.gz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="archive differs"):
        verify_archive(tmp_path, completed)


def test_repeat_invocation_preserves_active_status(tmp_path: Path) -> None:
    status = tmp_path / "after-priority-status.json"
    status.write_text('{"phase": "run-prepared"}')
    with pytest.raises(FileExistsError, match="Do not restart"):
        run(tmp_path)
    assert status.read_text() == '{"phase": "run-prepared"}'
