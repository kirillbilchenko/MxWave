"""Run the frozen deployment comparison on Spark, with independent exact restoration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

from control_stock_trial import (
    ORIGINAL,
    TERMINAL,
    health,
    identity,
    inspect,
    production_models,
    restore,
    retain_logs,
    run,
    ticks,
    write,
)


def watch(root: Path, pid: int, start: str) -> None:
    """Stop an orphaned benchmark client before restoring the original service."""
    write(root / "watchdog.json", {"pid": os.getpid(), "parent": pid, "ready": True})
    while True:
        state = json.loads((root / "controller.json").read_bytes())
        if state["phase"] in TERMINAL:
            return
        dead = ticks(pid) != start
        expired = time.time() > state.get("deadline_epoch", float("inf"))
        if state["service_paused"] and (dead or expired):
            child = state.get("client_pid")
            if child and ticks(child) == state.get("client_start_ticks"):
                try:
                    if os.getpgid(child) == child:
                        os.killpg(child, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            if not dead:
                os.kill(pid, signal.SIGTERM)
            else:
                state["error"] = "Controller disappeared; watchdog stopped client and restored"
                write(root / "controller.json", state)
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


def telemetry(root: Path, stage: dict[str, Any], stopped: threading.Event) -> None:
    """Record clocks, thermals, power, throttle reasons and comparable process RSS."""
    path = root / (stage["name"] + "-telemetry.jsonl")
    with path.open("x") as output:
        while not stopped.is_set():
            for rank in (0, 1):
                row: dict[str, Any] = {"epoch": time.time(), "rank": rank}
                try:
                    row["nvidia_smi_xml"] = run(rank, ["nvidia-smi", "-q", "-x"], timeout=15)
                    container = stage["servers"][rank]["name"]
                    row["docker_stats"] = json.loads(run(rank, [
                        "docker", "stats", "--no-stream", "--format", "{{json .}}", container],
                        timeout=15))
                    row["processes"] = run(rank, [
                        "docker", "top", container, "-eo", "pid,ppid,rss,comm"], timeout=15)
                except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
                    row["error"] = repr(exc)
                output.write(json.dumps(row) + "\n")
                output.flush()
            stopped.wait(5)


def start_stage(root: Path, stage: dict[str, Any], state: dict[str, Any]) -> None:
    """Stop the previous arm and start exactly the frozen next one."""
    # All candidates and quality clones are owned by this run; originals are never removed.
    for entry in state["owned"]:
        if run(entry["rank"], ["docker", "ps", "-q", "--filter",
                               f"name=^/{entry['name']}$"]).strip():
            run(entry["rank"], ["docker", "stop", "--timeout", "20", entry["name"]])
    if stage["kind"] != "production":
        for rank in (0, 1):
            if inspect(rank, ORIGINAL[rank])["State"]["Running"]:
                run(rank, ["docker", "stop", "--timeout", "30", ORIGINAL[rank]])
    for entry in reversed(stage["servers"]):
        if stage["kind"] == "production":
            if identity(inspect(entry["rank"], ORIGINAL[entry["rank"]])) != (
                    state["originals"][entry["rank"]]):
                raise RuntimeError("Original production identity changed")
            if not inspect(entry["rank"], ORIGINAL[entry["rank"]])["State"]["Running"]:
                run(entry["rank"], ["docker", "start", ORIGINAL[entry["rank"]]], timeout=120)
        else:
            run(entry["rank"], entry["argv"], timeout=120)
    deadline = min(time.time() + 1200, state["deadline_epoch"] - 180)
    while not health(stage["port"]):
        if time.time() > deadline:
            raise RuntimeError("Stage readiness exceeded its bounded startup allowance")
        for entry in stage["servers"]:
            if not inspect(entry["rank"], entry["name"])["State"]["Running"]:
                raise RuntimeError(f"{stage['name']} rank {entry['rank']} exited during startup")
        time.sleep(5)
    if stage["kind"] == "production":
        production_models()
    write(root / (stage["name"] + "-ready.json"), {
        "epoch": time.time(),
        "identities": [identity(inspect(e["rank"], e["name"])) for e in stage["servers"]]})


def control(root: Path) -> None:
    """Execute timing before full speculation-off GSM8K, without tuning between arms."""
    plan = json.loads((root / "prepared.json").read_bytes())
    path = root / "controller.json"
    if plan["status"] != "passed" or path.exists():
        raise RuntimeError("Require a fresh CPU-validated immutable preparation")
    for name, digest in plan["files"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Frozen preparation changed: {name}")
    for rank in (0, 1):
        if identity(inspect(rank, ORIGINAL[rank])) != plan["originals"][rank]:
            raise RuntimeError("Production changed since preparation")
    production_models()
    for _ in range(2):
        with urllib.request.urlopen("http://127.0.0.1:8000/metrics", timeout=10) as response:
            raw = response.read().decode()
        values = [float(line.rsplit(" ", 1)[1]) for line in raw.splitlines()
                  if re.match(r"vllm:num_requests_(running|waiting)(\{| )", line)]
        if not values or any(values):
            raise RuntimeError("Production is busy; no comparison started")
        time.sleep(3)
    state: dict[str, Any] = {
        "phase": "prepared", "pid": os.getpid(), "start_ticks": ticks(os.getpid()),
        "originals": plan["originals"], "owned": plan["owned"], "service_paused": False,
        "started_epoch": time.time(), "completed_stages": [], "qualification_passed": False}
    write(path, state)

    def terminated(signum: int, frame: Any) -> None:
        raise InterruptedError(f"Controller interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, terminated)
    signal.signal(signal.SIGINT, terminated)
    with (root / "watchdog.log").open("xb") as log:
        watcher = subprocess.Popen([
            "python3", "-S", str(Path(__file__).resolve()), "--root", str(root),
            "--watch", str(os.getpid()), state["start_ticks"]],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    child: subprocess.Popen[bytes] | None = None
    try:
        ready = time.monotonic() + 30
        while not (root / "watchdog.json").exists():
            if watcher.poll() is not None or time.monotonic() > ready:
                raise RuntimeError("Independent restoration watchdog failed to initialize")
            time.sleep(.2)
        state.update(service_paused=True, watchdog_pid=watcher.pid,
                     deadline_epoch=time.time() + plan["outage_seconds"])
        write(path, state)
        timing_blocked = False
        for stage in plan["stages"]:
            if stage["mode"] == "timing" and timing_blocked:
                state.setdefault("skipped_stages", []).append(stage["name"])
                write(path, state)
                continue
            state.update(phase="loading", stage=stage["name"])
            write(path, state)
            stopped = threading.Event()
            sampler = threading.Thread(target=telemetry, args=(root, stage, stopped), daemon=True)
            sampler.start()
            try:
                start_stage(root, stage, state)
                state["phase"] = stage["mode"]
                write(path, state)
                command = ["python3", "-S", str(root / "compare_client.py"), "--root", str(root),
                           "--arm", stage["arm"], "--port", str(stage["port"]),
                           "--mode", stage["mode"]]
                if stage["mode"] == "timing":
                    command += ["--trial", str(stage["trial"])]
                with (root / (stage["name"] + "-client.log")).open("xb") as log:
                    child = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                             stdout=log, stderr=subprocess.STDOUT,
                                             start_new_session=True)
                    state.update(client_pid=child.pid, client_start_ticks=ticks(child.pid))
                    write(path, state)
                    code = child.wait(timeout=min(stage["client_timeout_seconds"],
                                                  state["deadline_epoch"] - time.time() - 180))
                if code != 0:
                    raise RuntimeError(f"{stage['name']} collection failed; retain all partial data")
                child = None
                state.update(client_pid=None, client_start_ticks=None)
                state["completed_stages"].append(stage["name"])
                write(path, state)
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                # A timing/capacity failure must not silently erase the requested
                # independent target-quality measurement.
                state.setdefault("stage_errors", []).append(
                    {"stage": stage["name"], "error": repr(exc), "epoch": time.time()})
                state["error"] = "One or more comparison stages failed; see retained stage_errors"
                if stage["mode"] in ("timing", "compatibility"):
                    timing_blocked = True
                write(path, state)
            finally:
                if child is not None and child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(timeout=10)
                stopped.set()
                sampler.join(timeout=65)
                logs = root / "logs" / stage["name"]
                logs.mkdir(parents=True)
                retain_logs(logs, stage["servers"])
                subprocess.run(["python3", "-S", str(root / "compare_client.py"),
                                "--root", str(root), "--mode", "summary"],
                               check=False, timeout=60)
        state["phase"] = "collection_complete"
        write(path, state)
    except BaseException as exc:
        state.update(phase="failed", error=repr(exc))
        write(path, state)
        raise
    finally:
        if state["service_paused"]:
            restore(root)


def main() -> None:
    """Run the prepared suite, or its independent restoration watchdog."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--watch", nargs=2)
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
