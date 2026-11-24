"""Launch the existing pretrained screen after an explicit user stop and verified restoration."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def verify_readiness(
    state: dict[str, Any], request: dict[str, Any], plan_sha: str,
    preserved: list[dict[str, Any]],
) -> None:
    """Accept only the recorded user stop, restored original service and matching retained data."""
    if (request.get("status") != "stop_requested_by_user"
            or request.get("capture_plan_sha256") != plan_sha
            or request.get("capture_controller_pid") != state.get("pid")
            or request.get("data_deletion_requested") is not False
            or request.get("training_started") is not False):
        raise ValueError("Missing or mismatched explicit user-stop record")
    if (state.get("phase") != "failed_restored" or state.get("service_paused")
            or state.get("error") != "RuntimeError('Diagnostic interrupted by signal 15')"):
        raise ValueError("Capture did not stop through the expected restoration handler")
    restored = state["restoration"]
    if not restored.get("health_200") or not restored.get("originals_running"):
        raise ValueError("Original service restoration not verified")
    if not any(row.get("id") == "glm-5.3-flash" and row.get("max_model_len") == 500000
               for row in restored.get("models", {}).get("data", [])):
        raise ValueError("Restored model identity or context differs")
    if len(preserved) != 2:
        raise ValueError("Both independent preservation reviews are required")
    for rank, review in enumerate(preserved):
        if (review.get("status") != "partial_chunks_independently_verified"
                or review.get("rank") != rank or review.get("plan_sha256") != plan_sha
                or review.get("different_elements") != 0
                or review.get("training_dataset_complete") is not False
                or review.get("training_started") is not False
                or review.get("sealed_test_opened") is not False
                or review.get("verified_prompts", 0) <= 0):
            raise ValueError("Partial capture is not independently preserved")
    for key in ("paired_receipts", "verified_prompts", "accepted_bytes", "feature_rows",
                "numpy_bf16_elements_compared", "verified_rows"):
        if preserved[0][key] != preserved[1][key]:
            raise ValueError(f"Preserved rank data differs: {key}")


def run(root: Path) -> None:
    """Wait for user-stop restoration, then use the unchanged bounded preparation/controller."""
    parent = root.parent / "drafter-full-capture-v1"
    status_path = root / "after-stop-status.json"

    def status(phase: str, **extra: Any) -> None:
        data = {"phase": phase, "pid": os.getpid(), "epoch": time.time(), **extra}
        temp = status_path.with_suffix(".incomplete")
        temp.write_text(json.dumps(data, indent=2) + "\n")
        temp.replace(status_path)

    try:
        if status_path.exists() or (root / "prepared.json").exists():
            raise FileExistsError("Do not restart an existing prepared comparison")
        sources = json.loads((root / "source-manifest.json").read_bytes())
        for name, sha in sources.items():
            with (root / name).open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != sha:
                    raise ValueError("Frozen pretrained comparison source differs")
        plan = json.loads((root / "plan.json").read_bytes())
        with (parent / "plan.json").open("rb") as stream:
            plan_sha = hashlib.file_digest(stream, "sha256").hexdigest()
        if plan_sha != plan["wait_capture_sha256"]:
            raise ValueError("Parent capture plan differs")
        request = json.loads((parent / "user-stop-request.json").read_bytes())
        if Path(f"/proc/{request['waiting_trial_pid']}").exists():
            raise ValueError("The superseded waiter is still running")
        deadline = time.monotonic() + 1500
        while True:
            state = json.loads((parent / "controller.json").read_bytes())
            done_path = parent / "run-prepared-status.json"
            done = json.loads(done_path.read_bytes()) if done_path.exists() else {}
            if (state["phase"] == "failed_restored" and not state["service_paused"]
                    and done.get("step") == "finalize" and done.get("returncode") == 0
                    and (parent / "partial-capture.json").exists()):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("Restoration/finalization did not complete; no new outage")
            status("waiting_for_user_stop_restoration", capture_phase=state["phase"])
            time.sleep(5)
        sys.path.insert(0, str(root.parent / "full-c16admissionv1-src"))
        import full_model_control as control

        preserved = [control.remote_json(rank, parent / f"user-stop-preservation-r{rank}.json")
                     for rank in (0, 1)]
        verify_readiness(state, request, plan_sha, preserved)
        if not control.health(8000):
            raise ValueError("Original service is no longer healthy")
        readiness = {"status": "user_stop_restored_and_data_preserved", "epoch": time.time(),
                     "request": request, "restoration": state["restoration"],
                     "verified_prompts": preserved[0]["verified_prompts"],
                     "accepted_bytes_per_rank": preserved[0]["accepted_bytes"],
                     "capture_complete": False, "training_started": False,
                     "plan_sha256": hashlib.sha256((root / "plan.json").read_bytes()).hexdigest(),
                     "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        (root / "after-stop-readiness.json").write_text(json.dumps(readiness, indent=2) + "\n")
        for step in ("prepare", "run-prepared"):
            status(step, capture_stopped_by_user=True)
            with (root / (step + "-after-stop.log")).open("xb") as log:
                result = subprocess.run(
                    ["python3", "-S", "-B", "-u", str(root / (step + ".py"))],
                    stdout=log, stderr=subprocess.STDOUT, check=False,
                )
            if result.returncode:
                raise RuntimeError(f"{step} failed: {result.returncode}")
        current = json.loads((root / "controller.json").read_bytes())
        done = json.loads((root / "run-prepared-status.json").read_bytes())
        if (current["phase"] != "complete_restored" or current["service_paused"]
                or done.get("step") != "finalize" or done.get("returncode") != 0
                or done.get("control_returncode") != 0):
            raise ValueError("Comparison did not complete and restore")
        status("complete_retained_requires_review", training_started=False)
    except BaseException as error:
        status("failed", error=repr(error))
        raise


def main() -> None:
    """Run the explicit-stop continuation without changing frozen experiment sources."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    run(args.root)


if __name__ == "__main__":
    main()
