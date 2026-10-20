"""Run one isolated expert replay with the previously qualified restoration controller."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any


def helper(root: Path) -> Any:
    """Load the hash-pinned restoration implementation, binding only this fresh run directory."""
    plan = json.loads((root / "plan.json").read_bytes())
    path = root.parent / "drafter-components-v2/control.py"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == plan["restore_controller_sha256"]
    spec = importlib.util.spec_from_file_location("retained_drafter_controller", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT, module.STATE = root, root / "controller.json"
    return module


def run(root: Path, h: Any) -> None:
    """Pause only verified idle production, execute the bounded fixture and restore on exit."""
    assert not h.STATE.exists()
    prepared = json.loads((root / "prepared.json").read_bytes())
    plan = json.loads((root / "plan.json").read_bytes())
    assert prepared["status"] == "passed"
    outage_seconds = plan.get("outage_seconds", 900)
    replay_seconds = plan.get("replay_seconds", outage_seconds)
    assert 0 < replay_seconds <= outage_seconds <= 2100
    assert hashlib.sha256((root / "plan.json").read_bytes()).hexdigest() == prepared["plan_sha256"]
    for name, sha in plan["source_sha256"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == sha
    prior = json.loads((root.parent / "drafter-components-v2/controller.json").read_bytes())
    assert prior["phase"] == "complete_restored" and not prior["service_paused"]
    state = {"phase": "prepared", "pid": os.getpid(), "start_ticks": h.ticks(os.getpid()),
             "started_epoch": time.time(), "service_paused": False}
    h.save(state)

    def terminated(signum: int, frame: Any) -> None:
        raise RuntimeError(f"Component replay interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, terminated)
    signal.signal(signal.SIGINT, terminated)
    with (root / "watchdog.log").open("xb") as output:
        watcher = subprocess.Popen(
            ["python3", "-S", "-B", "-u", str(Path(__file__)), "--root", str(root),
             "--watch", str(os.getpid()), h.ticks(os.getpid())],
            stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    state["watchdog_pid"] = watcher.pid
    h.save(state)
    entry = prepared["launches"]["replay"][0]
    try:
        deadline = time.monotonic() + 30
        while not (root / "watchdog.json").exists():
            assert watcher.poll() is None and time.monotonic() < deadline
            time.sleep(0.2)
        assert [h.c.identity(h.c.inspect(r, prepared["live"][r]["id"])) for r in (0, 1)] == (
            prepared["live"]
        )
        assert h.c.health(8000)
        h.quiet()
        state.update(service_paused=True, outage_deadline_epoch=time.time() + outage_seconds)
        h.save(state, "stopping_original_production")
        for rank in (0, 1):
            h.c.run(rank, ["docker", "stop", "--timeout", "30", prepared["live"][rank]["id"]],
                    timeout=90)
        h.save(state, "replaying_expert")
        replay_deadline = time.time() + replay_seconds
        h.c.run(0, entry["argv"], timeout=120)
        while True:
            status = h.c.inspect(0, entry["name"])["State"]
            state["container_state"] = status
            h.save(state)
            if not status["Running"]:
                assert status["ExitCode"] == 0 and not status["OOMKilled"], status
                break
            assert time.time() < min(replay_deadline, state["outage_deadline_epoch"])
            time.sleep(3)
        report = json.loads((root / "result/report.json").read_bytes())
        assert report["status"] == "complete" and len(report["ranks"]) == 2
        h.save(state, "replay_complete")
    except BaseException as error:
        state["error"] = repr(error)
        h.save(state, "failed" if state["service_paused"] else "preparation_failed")
        raise
    finally:
        try:
            h.logs("replay", [entry])
        finally:
            if state["service_paused"]:
                h.restore()


def main() -> None:
    """Dispatch normal execution or the independent restoration watchdog."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--watch", nargs=2)
    args = parser.parse_args()
    h = helper(args.root)
    if args.watch:
        h.watch(int(args.watch[0]), args.watch[1])
    else:
        run(args.root, h)


if __name__ == "__main__":
    main()
