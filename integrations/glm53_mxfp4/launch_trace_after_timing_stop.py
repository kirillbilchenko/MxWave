"""Prioritize the frozen native trace after an explicitly stopped speed repeat."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
    """Hash a retained file with bounded memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_stop(state: dict[str, Any], request: dict[str, Any], parent_sha: str,
                trace_sha: str) -> None:
    """Require the recorded priority stop and healthy original 500k service."""
    if (request.get("status") != "baseline_repeat_stopped_for_user_priority"
            or request.get("parent_plan_sha256") != parent_sha
            or request.get("trace_plan_sha256") != trace_sha
            or request.get("controller_pid") != state.get("pid")
            or request.get("controller_start_ticks") != state.get("start_ticks")
            or request.get("phase_at_stop") not in ("starting_a2", "measuring_a2")
            or request.get("data_deletion_requested") is not False
            or request.get("training_started") is not False
            or request.get("speed_qualified") is not False
            or set(request.get("completed_measurements", {})) != {"a1", "b1", "c1"}):
        raise ValueError("Missing or mismatched user-priority stop")
    if (state.get("phase") != "failed_restored" or state.get("service_paused") is not False
            or state.get("error") != "RuntimeError('Diagnostic interrupted by signal 15')"):
        raise ValueError("Speed comparison did not stop through the expected restoration handler")
    restored = state.get("restoration", {})
    if (not restored.get("health_200") or not restored.get("originals_running")
            or not any(m.get("id") == "glm-5.3-flash" and m.get("max_model_len") == 500000
                       for m in restored.get("models", {}).get("data", []))):
        raise ValueError("Original 500k production restoration not verified")


def verify_archive(root: Path, completed: dict[str, str]) -> dict[str, Any]:
    """Independently verify the immutable archive, including completed measurement hashes."""
    archive = root / "evidence-v1.tar.gz"
    receipt = json.loads((root / "archive-receipt.json").read_bytes())
    sha = digest(archive)
    if sha != receipt["sha256"] or archive.stat().st_size != receipt["bytes"]:
        raise ValueError("Retained timing archive differs")
    got, manifest = {}, None
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            if not member.isfile() or member.name in got:
                raise ValueError("Unexpected archive member")
            stream = tar.extractfile(member)
            assert stream is not None
            if member.name == "evidence-manifest.json":
                if manifest is not None:
                    raise ValueError("Duplicate evidence manifest")
                manifest = json.load(stream)
            else:
                got[member.name] = {
                    "bytes": member.size,
                    "sha256": hashlib.file_digest(stream, "sha256").hexdigest(),
                }
    assert manifest is not None
    want = {r["path"]: {"bytes": r["bytes"], "sha256": r["sha256"]}
            for r in manifest["files"]}
    if len(want) != len(manifest["files"]) or got != want:
        raise ValueError("Retained timing payloads differ from the manifest")
    for arm, expected in completed.items():
        if got[f"{arm}-measurement.json"]["sha256"] != expected:
            raise ValueError("Completed timing arm changed")
    return {"status": "independently_verified", "sha256": sha,
            "bytes": archive.stat().st_size, "payloads": len(got)}


def run(root: Path) -> None:
    """Wait for verified cancellation/restoration, then run unchanged trace sources."""
    parent = root.parent / "drafter-pretrained-modal-v1"
    status_path = root / "after-priority-status.json"

    def status(phase: str, **extra: Any) -> None:
        value = {"phase": phase, "pid": os.getpid(), "epoch": time.time(), **extra}
        tmp = status_path.with_suffix(".incomplete")
        tmp.write_text(json.dumps(value, indent=2) + "\n")
        tmp.replace(status_path)

    if status_path.exists() or (root / "prepared.json").exists():
        raise FileExistsError("Do not restart an existing trace")
    try:
        sources = json.loads((root / "source-manifest.json").read_bytes())
        if not all(digest(root / name) == sha for name, sha in sources.items()):
            raise ValueError("Frozen trace sources changed")
        plan = json.loads((root / "plan.json").read_bytes())
        parent_sha, trace_sha = digest(parent / "plan.json"), digest(root / "plan.json")
        if parent_sha != plan["parent_plan_sha256"]:
            raise ValueError("Parent speed plan changed")
        request = json.loads((parent / "trace-priority-stop.json").read_bytes())
        proc = Path(f"/proc/{request['waiter_pid']}/stat")
        if proc.exists():
            fields = proc.read_text().rsplit(")", 1)[1].split()
            if fields[19] == request["waiter_start_ticks"] and fields[0] != "Z":
                raise ValueError("Superseded waiter remains running")
        deadline = time.monotonic() + 1500
        while True:
            state = json.loads((parent / "controller.json").read_bytes())
            path = parent / "run-prepared-status.json"
            done = json.loads(path.read_bytes()) if path.exists() else {}
            if state["phase"] == "failed_restored" and done.get("step") == "finalize":
                if done.get("returncode") != 0 or done.get("control_returncode") != 1:
                    raise ValueError("Expected cancelled control and successful finalization")
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("Restoration/finalization timed out; do not start another outage")
            status("waiting_for_priority_stop_restoration", parent_phase=state["phase"])
            time.sleep(5)
        verify_stop(state, request, parent_sha, trace_sha)
        archive = verify_archive(parent, request["completed_measurements"])
        for arm, sha in request["completed_measurements"].items():
            path = parent / f"{arm}-measurement.json"
            measurement = json.loads(path.read_bytes())
            if (digest(path) != sha or measurement["status"] != "complete"
                    or not measurement["repeat_tokens_and_counters_exact"]
                    or len(measurement["cases"]) != 48):
                raise ValueError("Completed comparison arm not preserved")
            if arm in ("a1", "b1"):
                rows = [r for r in measurement["cases"]
                        if r["case"] == plan["calibration_cases"][0]["id"]]
                if len(rows) != 6 or any(r["result"]["token_ids"][:64]
                                        != plan["expected_outputs"][arm] for r in rows):
                    raise ValueError("Required native diagnostic prefix did not repeat")
        sys.path.insert(0, str(root.parent / "full-c16admissionv1-src"))
        import full_model_control as control

        if not control.health(8000):
            raise ValueError("Original service is no longer healthy")
        readiness = {"status": "priority_stop_restored_and_evidence_verified",
                     "epoch": time.time(), "request": request, "archive": archive,
                     "restoration": state["restoration"], "plan_sha256": trace_sha,
                     "launcher_sha256": digest(Path(__file__)), "speed_qualified": False}
        (root / "after-priority-readiness.json").write_text(json.dumps(readiness, indent=2) + "\n")
        for step in ("prepare", "run-prepared"):
            status(step, final_speed_repeat_skipped=True)
            with (root / (step + "-after-priority.log")).open("xb") as log:
                result = subprocess.run(
                    ["python3", "-S", "-B", "-u", str(root / (step + ".py"))],
                    stdout=log, stderr=subprocess.STDOUT, check=False,
                )
            if result.returncode:
                raise RuntimeError(f"{step} failed: {result.returncode}")
        state = json.loads((root / "controller.json").read_bytes())
        done = json.loads((root / "run-prepared-status.json").read_bytes())
        if (state["phase"] != "complete_restored" or state["service_paused"]
                or done.get("step") != "finalize" or done.get("returncode") != 0
                or done.get("control_returncode") != 0):
            raise ValueError("Native diagnostic did not complete and restore")
        status("complete_retained_requires_review", training_started=False)
    except BaseException as error:
        status("failed", error=repr(error))
        raise


def main() -> None:
    """Launch only the already frozen native diagnostic after the recorded user-priority stop."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    run(args.root)


if __name__ == "__main__":
    main()
