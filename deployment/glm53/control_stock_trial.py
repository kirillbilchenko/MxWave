"""Bounded two-Spark stock-vLLM smoke trial with an independent restoration watchdog.

This operational harness is separate from the portable serving recipe. Run on
Spark1 only after preparation, with the user's authorization to pause production.
It never promotes the candidate or changes an original container configuration.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any

BASE = Path("/home/kirya/local-spark/experiments/glm53-mxfp4-audit-2026-10-15/full-model-v1")
RELEASE = BASE / "glm53-package-v1"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-i",
       "/home/kirya/.ssh/id_glm_cluster", "kirya@192.168.192.2"]
ORIGINAL = (
    "8e0426b9a4183b8fdaa2ed2d4047924e9554b2cb37950e8f31bce52589d07e11",
    "fb147a890d8bdd2b475588935a3e1d6d7c60828f894bc980bc7b6e7a40d00224",
)
TERMINAL = ("complete_restored", "failed_restored", "preparation_failed")


def run(rank: int, command: list[str], timeout: float = 90) -> str:
    """Run explicit commands, keeping production credentials out of retained state."""
    argv = command if rank == 0 else [*SSH, shlex.join(command)]
    result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError(f"rank {rank}, {command[:2]}: {result.stderr[-2000:]}")
    return result.stdout


def inspect(rank: int, container: str) -> dict[str, Any]:
    """Read a container privately; retain only its digested identity."""
    return json.loads(run(rank, ["docker", "inspect", container]))[0]


def identity(value: dict[str, Any]) -> dict[str, str]:
    """Bind image, settings and mounts without publishing secret environment values."""
    fields = {"config": value["Config"], "host": value["HostConfig"],
              "mounts": sorted(value["Mounts"], key=lambda m: m["Destination"])}
    return {"id": value["Id"], "image": value["Image"], **{
        key: hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()
        for key, item in fields.items()}}


def write(path: Path, value: Any) -> None:
    """Save state atomically for the independent watchdog."""
    temporary = path.with_suffix(".incomplete")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def ticks(pid: int) -> str | None:
    """Distinguish the controller from a subsequently reused Linux PID."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except FileNotFoundError:
        return None


def health(port: int) -> bool:
    """Check readiness without printing authentication material."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as response:
            return response.status == 200
    except OSError:
        return False


def production_models() -> dict[str, Any]:
    """Verify the original authenticated 500k service after restoration."""
    key = Path("/home/kirya/local-spark/runtime/secrets/llm_api_key").read_text().strip()
    request = urllib.request.Request("http://127.0.0.1:8000/v1/models",
                                     headers={"Authorization": "Bearer " + key})
    with urllib.request.urlopen(request, timeout=20) as response:
        result = json.load(response)
    if not any(m["id"] == "glm-5.3-flash" and m["max_model_len"] == 500000
               for m in result["data"]):
        raise RuntimeError("Original production context/model changed")
    return result


def restore(root: Path) -> None:
    """Stop only this trial's containers and restore the exact original identities."""
    with (root / "restore.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state_path = root / "controller.json"
        state = json.loads(state_path.read_bytes())
        if state["phase"] in TERMINAL:
            return
        state["phase"] = "restoring"
        write(state_path, state)
        for entry in state["owned"]:
            if run(entry["rank"], ["docker", "ps", "-q", "--filter",
                                   f"name=^/{entry['name']}$"]).strip():
                run(entry["rank"], ["docker", "stop", "--timeout", "20", entry["name"]])
        for rank in (1, 0):
            current = inspect(rank, ORIGINAL[rank])
            if identity(current) != state["originals"][rank]:
                raise RuntimeError("Original container identity changed; preserve it")
            if not current["State"]["Running"]:
                run(rank, ["docker", "start", ORIGINAL[rank]], timeout=120)
        deadline = time.monotonic() + 1200
        while not health(8000):
            if time.monotonic() > deadline or not all(
                inspect(r, ORIGINAL[r])["State"]["Running"] for r in (0, 1)
            ):
                raise RuntimeError("Original production restoration did not complete")
            time.sleep(5)
        state["restoration"] = {"epoch": time.time(), "models": production_models(),
                                "identities": [identity(inspect(r, ORIGINAL[r])) for r in (0, 1)]}
        state.update(service_paused=False, phase="failed_restored" if state.get("error")
                     else "complete_restored", finished_epoch=time.time())
        write(state_path, state)


def watch(root: Path, pid: int, start: str) -> None:
    """Restore independently if the laptop/controller disappears or its lease expires."""
    write(root / "watchdog.json", {"pid": os.getpid(), "parent": pid, "ready": True})
    while True:
        state = json.loads((root / "controller.json").read_bytes())
        if state["phase"] in TERMINAL:
            return
        dead = ticks(pid) != start
        expired = time.time() > state.get("deadline_epoch", float("inf"))
        if state["service_paused"] and (dead or expired):
            if not dead:
                os.kill(pid, signal.SIGTERM)
            for attempt in range(3):
                try:
                    restore(root)
                    return
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(10)
        if dead:
            return
        time.sleep(3)


def retain_logs(root: Path, entries: list[dict[str, Any]]) -> None:
    """Keep stdout and stderr for every trial container, including failures."""
    for entry in entries:
        command = ["docker", "logs", entry["name"]]
        argv = command if entry["rank"] == 0 else [*SSH, shlex.join(command)]
        result = subprocess.run(argv, capture_output=True, timeout=30, check=False)
        (root / (entry["name"] + ".log")).write_bytes(result.stdout + b"\n" + result.stderr)


def control(root: Path) -> None:
    """Run the frozen preflight, component checks and one full-model smoke, then restore."""
    plan = json.loads((root / "prepared.json").read_bytes())
    if plan["status"] != "passed" or (root / "controller.json").exists():
        raise RuntimeError("Require a fresh, fully prepared trial directory")
    for rank in (0, 1):
        if identity(inspect(rank, ORIGINAL[rank])) != plan["originals"][rank]:
            raise RuntimeError("Production changed after preparation")
    production_models()
    for _ in range(2):
        with urllib.request.urlopen("http://127.0.0.1:8000/metrics", timeout=10) as response:
            metrics = response.read().decode()
        values = [float(s.rsplit(" ", 1)[1]) for s in metrics.splitlines()
                  if re.match(r"vllm:num_requests_(running|waiting)(\{| )", s)]
        if not values or any(values):
            raise RuntimeError("Production is busy; no outage started")
        time.sleep(3)
    entries = plan["owned"]
    state = {"phase": "prepared", "pid": os.getpid(), "start_ticks": ticks(os.getpid()),
             "service_paused": False, "originals": plan["originals"], "owned": entries,
             "started_epoch": time.time(), "scope": plan["scope"]}
    path = root / "controller.json"
    write(path, state)

    def terminated(signum: int, frame: Any) -> None:
        raise RuntimeError(f"Trial interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, terminated)
    signal.signal(signal.SIGINT, terminated)
    with (root / "watchdog.log").open("xb") as output:
        watcher = subprocess.Popen(
            ["python3", "-S", str(Path(__file__).resolve()), "--root", str(root),
             "--watch", str(os.getpid()), state["start_ticks"]], stdin=subprocess.DEVNULL,
            stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        ready_deadline = time.monotonic() + 30
        while not (root / "watchdog.json").exists():
            if watcher.poll() is not None or time.monotonic() > ready_deadline:
                raise RuntimeError("Watchdog did not initialize")
            time.sleep(.2)
        state.update(service_paused=True, deadline_epoch=time.time() + plan["outage_seconds"],
                     phase="stopping_production", watchdog_pid=watcher.pid)
        write(path, state)
        for rank in (0, 1):
            run(rank, ["docker", "stop", "--timeout", "30", ORIGINAL[rank]])
        state["phase"] = "gpu_components"
        write(path, state)
        component = plan["component"]
        run(component["rank"], component["argv"])
        exit_code = run(component["rank"], ["docker", "wait", component["name"]], timeout=180)
        if exit_code.strip() != "0":
            raise RuntimeError("Native GPU component check failed; full model not started")
        state["phase"] = "loading_stock_vllm"
        write(path, state)
        for entry in reversed(plan["servers"]):
            run(entry["rank"], entry["argv"])
        while not health(plan["port"]):
            if time.time() > state["deadline_epoch"] - 180:
                raise RuntimeError("Model readiness exceeded bounded trial time")
            for entry in plan["servers"]:
                if not inspect(entry["rank"], entry["name"])["State"]["Running"]:
                    raise RuntimeError(f"Candidate rank {entry['rank']} exited before readiness")
            time.sleep(5)
        state["phase"] = "smoke_requests"
        write(path, state)
        subprocess.run(["python3", "-S", str(root / "smoke_client.py"), "--root", str(root),
                        "--port", str(plan["port"])], check=True, timeout=150)
        state["phase"] = "checks_complete"
        write(path, state)
    except BaseException as exc:
        state["error"] = repr(exc)
        state["phase"] = "failed"
        write(path, state)
        raise
    finally:
        try:
            retain_logs(root, entries)
        finally:
            if state["service_paused"]:
                restore(root)
            else:
                state["phase"] = "preparation_failed"
                write(path, state)


def main() -> None:
    """Run or restore one explicitly prepared operational trial."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--watch", nargs=2, metavar=("PID", "START_TICKS"))
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()
    if args.watch:
        watch(args.root, int(args.watch[0]), args.watch[1])
    elif args.restore:
        restore(args.root)
    else:
        control(args.root)


if __name__ == "__main__":
    main()
