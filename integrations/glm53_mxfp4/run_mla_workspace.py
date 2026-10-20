"""Run the fixed isolated MLA cases under the existing idle and restoration contract."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import full_model_control as c

REVISION = "mlaworkspacev1"


def main() -> None:
    """Use fresh bounded GPU processes and restore both exact original containers once."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", type=int)
    args = parser.parse_args()
    path = c.ROOT / f"mla-workspace-controller-{REVISION}.json"
    if args.watch is not None:
        c.watch(REVISION, args.watch, path)
        return
    if path.exists():
        raise FileExistsError("Preserve the first isolated MLA attempt")
    root = c.ROOT / "mla-workspace-v1"
    code = c.ROOT / f"full-{REVISION}-src"
    prepared = json.loads((root / "prepared.json").read_bytes())
    lock = (c.ROOT / "diagnostics-v1/component.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state: dict[str, Any] = {"phase": "preparing", "service_paused": False,
        "model_loads": 0, "qualification_attempted": False, "started_epoch": time.time(),
        "cases": [],
        "scope": "Fixed synthetic sparse-MLA operation, cache, plan and private workspace"}
    success = False

    def save(phase: str) -> None:
        state.update(phase=phase, updated_epoch=time.time())
        c.write(path, state)

    def terminated(signum: int, frame: Any) -> None:
        raise SystemExit(f"Signal {signum}")

    signal.signal(signal.SIGTERM, terminated)
    signal.signal(signal.SIGINT, terminated)
    try:
        save("preparing")
        bindings = [c.preflight(rank, code) for rank in (0, 1)]
        if bindings != prepared["bindings"]:
            raise ValueError("Frozen component source changed")
        state["prepared_sha256"] = hashlib.sha256((root / "prepared.json").read_bytes()).hexdigest()
        plan_path = root / "plan.json"
        if hashlib.sha256(plan_path.read_bytes()).hexdigest() != prepared["plan_sha256"]:
            raise ValueError("Component plan changed")
        plan = json.loads(plan_path.read_bytes())
        originals = [c.inspect(rank, c.ORIGINAL_IDS[rank]) for rank in (0, 1)]
        if any(not item["State"]["Running"] or item["Image"] != c.IMAGE for item in originals):
            raise RuntimeError("Require both original GLM containers running")
        state["originals"] = [c.identity(item) for item in originals]
        spec = importlib.util.spec_from_file_location("quiet_guard", c.OLD /
            "triple-v2/src/integrations/marlin_scales/quiet_window.py")
        assert spec and spec.loader
        guard = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(guard)
        previous, quiet = None, 0
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            now = guard.activity()
            quiet = quiet + 1 if not now["busy"] and now["counters"] == previous else 0
            previous = now["counters"]
            state["last_activity"] = now
            save("waiting_for_idle")
            if quiet >= 3:
                final = guard.activity()
                if not final["busy"] and final["counters"] == previous:
                    break
            time.sleep(5)
        else:
            raise TimeoutError("No idle window within thirty minutes")
        watchdog = subprocess.Popen(["python3", "-S", "-u", str(Path(__file__).resolve()),
            "--watch", str(os.getpid())], stdout=(root / "watchdog.log").open("xb"),
            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        state["watchdog_pid"] = watchdog.pid
        for _ in range(30):
            ack = c.ROOT / f"watchdog-{REVISION}.json"
            if ack.exists() and json.loads(ack.read_bytes()).get("ready"):
                break
            time.sleep(1)
        else:
            raise RuntimeError("Independent restoration watchdog did not arm")
        final = guard.activity()
        if final["busy"] or final["counters"] != previous:
            raise RuntimeError("Service became busy before the pause")
        state.update(service_paused=True, restore_deadline_epoch=time.time() + 2700)
        save("pausing_glm")
        for rank in (0, 1):
            c.run(rank, ["docker", "stop", "--timeout", "30", c.ORIGINAL_IDS[rank]], timeout=90)
        for index, case in enumerate(plan["cases"]):
            name = f"mxwave-glm53-full-{REVISION}-raw-r1"
            # --rm removes only this owned disposable process after its logs/exit are captured.
            argv = ["docker", "run", "--rm", "--name", name, "--gpus", "all", "--network", "none",
                "--cpus", "4", "--memory", "6g", "--memory-swap", "6g", "--pids-limit", "256",
                "--ipc", "host", "--ulimit", "memlock=-1", "-e", "CUDA_VISIBLE_DEVICES=0",
                "-e", "PYTHONPATH=/work", "-e", "PYTHONDONTWRITEBYTECODE=1",
                "-e", "CUBLAS_WORKSPACE_CONFIG=:4096:8", "-e", "OMP_NUM_THREADS=2",
                "-e", "OPENBLAS_NUM_THREADS=2", "-e", "XDG_CACHE_HOME=/results/cache",
                "-v", f"{code}:/work:ro", "-v", f"{root / 'plan.json'}:/plan.json:ro",
                "-v", f"{root / 'results'}:/results", "--entrypoint", "/usr/bin/timeout", c.IMAGE,
                "--signal=TERM", "--kill-after=5", "240", "/usr/bin/python3", "-u",
                "/work/diagnose_mla_workspace.py", "--plan", "/plan.json", "--case", str(index),
                "--output", f"/results/case-{index}"]
            save(f"running_case_{index}")
            capture = ("import json,subprocess,sys;p=subprocess.run(json.loads(sys.argv[1]),"
                       "stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);"
                       "print(json.dumps({'exit_code':p.returncode,'output':p.stdout}))")
            result = json.loads(c.run(1,
                ["python3", "-S", "-c", capture, json.dumps(argv)], timeout=300))
            (root / f"case-{index}.log").write_text(result["output"])
            report = c.remote_json(1, root / f"results/case-{index}/report.json")
            state["cases"].append({"index": index, "case": case, "exit_code": result["exit_code"],
                                    "report": report})
            save(f"collected_case_{index}")
        success = all(row["exit_code"] == 0 and row["report"]["status"] == "complete"
                      for row in state["cases"])
        save("measured_pending_restoration")
    except BaseException as exc:
        state["error"] = repr(exc)
        save("failed_pending_restoration" if state["service_paused"] else "preparation_failed")
        raise
    finally:
        if state["service_paused"]:
            save("restoring")
            state["restoration"] = c.restore(REVISION, path)
            state["service_paused"] = False
            save("complete" if success else "failed_restored")
        lock.close()


if __name__ == "__main__":
    main()
