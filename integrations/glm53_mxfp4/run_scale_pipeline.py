"""Run prepared scale profiles and benchmarks under one GLM pause, then restore once."""

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
from collections.abc import Callable
from pathlib import Path
from typing import Any

import full_model_control as control
from run_bf16_kernels import command, memory_available


def jobs(plan: dict[str, Any]) -> list[dict[str, str]]:
    """Accept only immutable component roots and known two-process protocols."""
    entries = plan.get("jobs")
    if not isinstance(entries, list) or not 1 <= len(entries) <= 6:
        raise ValueError("Require one to six prepared component jobs")
    seen: set[str] = set()
    result = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise TypeError("Component jobs must be objects")
        name, root, kind = (entry.get(k) for k in ("name", "root", "kind"))
        if (not isinstance(name, str) or not name.isalnum() or name in seen
                or not isinstance(root, str) or not root.startswith("bf16-kernels-")
                or Path(root).name != root or kind not in ("counters", "timing")):
            raise ValueError("Invalid or duplicate component job")
        seen.add(name)
        result.append({"name": name, "root": root, "kind": kind})
    if len({r["root"] for r in result}) != len(result):
        raise ValueError("Every job needs its own report directory")
    return result


def validate(root: Path) -> dict[str, Any]:
    """Hash-bind code, both native libraries and the independent source fixture before pausing."""
    program = """import pathlib,hashlib,json,sys
r=pathlib.Path(sys.argv[1]); base=pathlib.Path(sys.argv[2])
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
m=json.loads((r/'build/manifest.json').read_bytes())
assert set(m['variants'])=={'aligned','sparse'}
for name,want in m['libraries_sha256'].items():
 assert pathlib.Path(name).name==name and sha(r/'build/build'/name)==want
prior=base/'scale-sparse-cpu-v2/results/screen.json'; p=json.loads(prior.read_bytes())
assert p['status']=='passed' and len(p['cases'])==4 and p['format']=='mxwave-palette16-v2'
assert p['codec_sha256']==sha(r/'src/scale_tiles.py')
fixture=base.parent/'grouped-moe-full-layer-v1/fixture'; f=json.loads((fixture/'manifest.json').read_bytes())
assert f['expert_ids']==list(range(288))
for row in f['tensors']:
 assert pathlib.Path(row['path']).name==row['path'] and sha(fixture/row['path'])==row['sha256']
print(json.dumps({'build_sha256':sha(r/'build/manifest.json'),
 'libraries_sha256':m['libraries_sha256'],'fixture_sha256':sha(fixture/'manifest.json'),
 'source_sha256':{str(p.relative_to(r/'src')):sha(p) for p in sorted((r/'src').rglob('*'))
                  if p.is_file() and p.suffix in ('.py','.cu','.cuh','.json')}}))
"""
    result = json.loads(control.run(1, ["python3", "-S", "-c", program,
                                       str(root), str(control.ROOT)], timeout=180))
    visible = """import pathlib,json,hashlib,sys
r=pathlib.Path('/audit')/sys.argv[1]; m=json.loads((r/'build/manifest.json').read_bytes()); out={}
for name,want in m['libraries_sha256'].items():
 p=r/'build/build'/name
 assert p.resolve().is_relative_to(pathlib.Path('/audit')),str(p)
 got=hashlib.sha256(p.read_bytes()).hexdigest();assert got==want
 out[name]=got
print(json.dumps(out))
"""
    argv = ["docker", "run", "--rm", "--network", "none", "--cpus", "2",
        "--memory", "512m", "--memory-swap", "512m", "-e", "NVIDIA_VISIBLE_DEVICES=void",
        "-e", "CUDA_VISIBLE_DEVICES=", "-v", f"{control.ROOT}:/audit:ro",
        "--entrypoint", "/usr/bin/python3", control.IMAGE, "-S", "-c", visible, root.name]
    result["container_visible_libraries"] = json.loads(control.run(1, argv, timeout=120))
    if result["container_visible_libraries"] != result["libraries_sha256"]:
        raise ValueError("Container-visible native libraries differ from host preflight")
    return result


def run_job(revision: str, job: dict[str, str], save: Callable[[str], None]) -> dict[str, Any]:
    """Retain both arms and logs, removing only our exited containers between jobs."""
    root = control.ROOT / job["root"]
    root.mkdir(exist_ok=True)
    control.run(1, ["mkdir", "-p", str(root / "results")])
    result: dict[str, Any] = {"job": job, "stages": [], "started_epoch": time.time()}
    for confirmation in (False, True):
        stage = "sparse-confirm" if confirmation else "sparse"
        argv = command(revision, root, profile=False, sparse=True,
                       sparse_ncu=job["kind"] == "counters", confirmation=confirmation)
        name = argv[argv.index("--name") + 1]
        save("starting_" + job["name"] + "_" + stage)
        control.run(1, argv)
        deadline = time.monotonic() + 630
        while time.monotonic() < deadline:
            info = control.inspect(1, name)
            if not info["State"]["Running"]:
                break
            time.sleep(2)
        else:
            raise TimeoutError("Bounded component container exceeded deadline")
        program = ("import subprocess,sys; p=subprocess.run(['docker','logs',sys.argv[1]],"
                   "stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=True);"
                   "sys.stdout.buffer.write(p.stdout)")
        (root / (stage + ".log")).write_text(
            control.run(1, ["python3", "-S", "-c", program, name]))
        control.write(root / (stage + "-container.json"),
                      {"id": info["Id"], "state": info["State"], "command": argv})
        if info["State"]["ExitCode"]:
            raise RuntimeError(f"{job['name']}/{stage} exited {info['State']['ExitCode']}")
        report = control.remote_json(1, root / f"results/{stage}.json")
        control.write(root / (stage + ".json"), report)
        if report.get("status") != "passed":
            raise RuntimeError("Component did not complete its correctness gates")
        result["stages"].append({"stage": stage, "status": "passed", "container_id": info["Id"]})
        # The process has exited; retain its ID/state, command and both log streams before removal.
        control.run(1, ["docker", "rm", info["Id"]])
    result.update(status="collected", finished_epoch=time.time())
    return result


def main() -> None:
    """Pause production once for the prepared batch; a detached watchdog guards the whole batch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--watch", type=int)
    args = parser.parse_args()
    if not args.revision.isalnum():
        raise ValueError("Use a fresh alphanumeric batch revision")
    path = control.ROOT / f"scale-pipeline-controller-{args.revision}.json"
    if args.watch is not None:
        control.watch(args.revision, args.watch, path)
        return
    if args.plan is None or path.exists():
        raise ValueError("Require a prepared plan and an unused batch revision")
    plan = jobs(json.loads(args.plan.read_bytes()))
    lock = (control.ROOT / "diagnostics-v1/component.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state: dict[str, Any] = {"phase": "preparing", "service_paused": False,
        "started_epoch": time.time(), "runs": [], "controller_pid": os.getpid(),
        "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        "scope": "Component-only palette pipeline batch; one GLM pause and final restoration"}

    def save(phase: str) -> None:
        state.update(phase=phase, updated_epoch=time.time())
        control.write(path, state)

    def terminated(signum: int, frame: Any) -> None:
        raise SystemExit(f"Signal {signum}")

    signal.signal(signal.SIGTERM, terminated)
    signal.signal(signal.SIGINT, terminated)
    success = False
    try:
        save("preparing")
        originals = [control.inspect(rank, control.ORIGINAL_IDS[rank]) for rank in (0, 1)]
        if any(not r["State"]["Running"] or r["Image"] != control.IMAGE for r in originals):
            raise ValueError("Original service must be running with its recorded image before the batch")
        state["originals"] = [control.identity(r) for r in originals]
        state["preflight"] = {j["name"]: validate(control.ROOT / j["root"]) for j in plan}
        spec = importlib.util.spec_from_file_location("quiet_guard", control.OLD /
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
                break
            time.sleep(5)
        else:
            raise TimeoutError("No idle window before the batch deadline")
        log = (control.ROOT / f"scale-pipeline-watchdog-{args.revision}.log").open("xb")
        watcher = subprocess.Popen(["python3", "-u", str(Path(__file__).resolve()),
            "--revision", args.revision, "--watch", str(os.getpid())],
            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        state["watchdog_pid"] = watcher.pid
        for _ in range(30):
            ack = control.ROOT / f"watchdog-{args.revision}.json"
            if ack.exists() and json.loads(ack.read_bytes()).get("ready"):
                break
            time.sleep(1)
        else:
            raise RuntimeError("Independent batch watchdog did not arm")
        final = guard.activity()
        if final["busy"] or final["counters"] != previous:
            raise RuntimeError("GLM became active before the pause")
        state.update(service_paused=True, restore_deadline_epoch=time.time() + 8400)
        save("pausing_glm")
        for rank in (0, 1):
            control.run(rank, ["docker", "stop", "--timeout", "30", control.ORIGINAL_IDS[rank]],
                        timeout=90)
        if memory_available(1) < 30 * 1024**3:
            raise RuntimeError("Insufficient released memory for the bounded component batch")
        for job in plan:
            state["runs"].append(run_job(args.revision, job, save))
            save("finished_" + job["name"])
        success = True
        save("measured_pending_restoration")
    except BaseException as exc:
        state["error"] = repr(exc)
        save("failed_pending_restoration" if state["service_paused"] else "preparation_failed")
        raise
    finally:
        if state["service_paused"]:
            save("restoring")
            state["restoration"] = control.restore(args.revision, path)
            state["service_paused"] = False
            save("complete" if success else "failed_restored")
        lock.close()


if __name__ == "__main__":
    main()
