"""Measure two bounded KDA memory diagnostics with exact original-service restoration."""

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

import full_model_control as control
from kda_memory_probe import METRICS, assess, protocol

REVISION = "kdamemoryv1"
DIRECTORY = "kda-memory-v1"
SOURCE_DIRECTORY = DIRECTORY + "-src-v2"
PREPARED_NAME = DIRECTORY + "-prepared-v2.json"


def command(rank: int, *, profile: bool) -> list[str]:
    """Use separate timed/counter processes and restrict profiler capabilities to its job."""
    if rank not in (0, 1):
        raise ValueError("Require a declared Spark rank")
    root = control.ROOT / DIRECTORY
    stage, mode = ("profile", "aligned") if profile else ("timing", "raw")
    argv = ["docker", "run", "-d", "--name", f"mxwave-glm53-full-{REVISION}-{mode}-r{rank}",
            "--gpus", "all", "--network", "none", "--cpus", "2", "--memory", "8g",
            "--memory-swap", "8g", "--pids-limit", "512"]
    if not profile:
        argv.extend(["--user", f"{os.getuid()}:{os.getgid()}"])
    for source, target, readonly in (
        (control.ROOT / SOURCE_DIRECTORY, "/src", True),
        (control.ROOT, "/audit", True), (control.OLD, "/experiment", True),
        (root / f"rank{rank}", "/result", False),
    ):
        argv.extend(["-v", f"{source}:{target}" + (":ro" if readonly else "")])
    environment = {
        "PYTHONPATH": "/src:/experiment/triple-v2/src/integrations/zipserv",
        "MXWAVE_ZIPSERV_LIBRARY": "/audit/native/libmxwave_kda_zipserv.so",
        "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "TRITON_CACHE_DIR": f"/result/{stage}-cache/triton",
        "CUDA_CACHE_PATH": f"/result/{stage}-cache/cuda",
        "XDG_CACHE_HOME": f"/result/{stage}-cache",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }
    for key, value in environment.items():
        argv.extend(["-e", f"{key}={value}"])
    if profile:
        argv.extend(["--cap-add", "SYS_ADMIN", "-v",
                     "/opt/nvidia/nsight-compute/2025.3.1:/ncu:ro"])
    timeout = protocol()["profile_timeout_seconds" if profile else "component_timeout_seconds"]
    argv.extend(["--entrypoint", "/usr/bin/timeout", control.IMAGE, "--signal=TERM",
                 "--kill-after=10", str(timeout)])
    if profile:
        argv.extend(["/ncu/ncu", "--profile-from-start", "off", "--nvtx",
                     "--clock-control", "none", "--cache-control", "all", "--launch-count", "80",
                     "--metrics", ",".join(METRICS), "--section", "LaunchStats",
                     "--section", "SchedulerStats", "--section", "WarpStateStats",
                     "--export", "/result/kda-memory", "--csv", "--page", "raw",
                     "--log-file", "/result/ncu.csv"])
    argv.extend(["/usr/bin/python3", "-u", "/src/benchmark_kda_memory.py",
                 "--report", f"/result/{stage}.json"])
    if profile:
        argv.append("--profile")
    return argv


def input_bindings(rank: int) -> dict[str, Any]:
    """Bind actual immutable fixtures and the installed profiler without initializing CUDA."""
    script = """import hashlib,json,pathlib,subprocess,sys
r=pathlib.Path(sys.argv[1]);f=r/'bf16-attention-v1/artifacts-v2'
def sha(p):
 with p.open('rb') as s:return hashlib.file_digest(s,'sha256').hexdigest()
m=json.loads((f/'prepared.json').read_bytes());rows=[x for x in m['cases'] if x['name'].startswith('kda-')]
assert [x['name'] for x in rows]==['kda-input-l0-r0','kda-output-l0-r0']
artifacts={}
for x in rows:
 p=f/x['artifact'];assert p.name==x['artifact'] and sha(p)==x['artifact_sha256']
 artifacts[p.name]=sha(p)
ncu=pathlib.Path('/opt/nvidia/nsight-compute/2025.3.1/ncu');assert ncu.is_file()
print(json.dumps({'manifest_sha256':sha(f/'prepared.json'),'artifacts':artifacts,
 'ncu_sha256':sha(ncu),'ncu_version':subprocess.check_output([str(ncu),'--version'],text=True)}))
"""
    return json.loads(control.run(rank, ["python3", "-S", "-c", script, str(control.ROOT)],
                                  timeout=120))


def preflight() -> dict[str, Any]:
    """Require the frozen protocol, unchanged sources and preceding verified evidence."""
    path = control.ROOT / PREPARED_NAME
    prepared = json.loads(path.read_bytes())
    if prepared["protocol"] != protocol():
        raise ValueError("Prepared protocol differs from the prospective component contract")
    bindings = [control.preflight(rank, control.ROOT / SOURCE_DIRECTORY)
                for rank in (0, 1)]
    if bindings != prepared["bindings"] or bindings[0] != bindings[1]:
        raise ValueError("Source bindings changed after preparation")
    inputs = [input_bindings(rank) for rank in (0, 1)]
    if inputs != prepared["input_bindings"]:
        raise ValueError("Fixture or profiler changed after preparation")
    proof_path = control.ROOT / "warmed-model-v1-recovery-v2-evidence-check.json"
    if hashlib.sha256(proof_path.read_bytes()).hexdigest() != prepared["parent_proof_sha256"]:
        raise ValueError("Preceding verified model evidence changed")
    return {"prepared_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bindings": bindings, "originals": prepared["originals"],
            "protocol": prepared["protocol"], "input_bindings": inputs}


def main() -> None:
    """Finish both declared GPU processes and restore the same original containers on exit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", type=int)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    path = control.ROOT / (DIRECTORY + "-controller.json")
    if args.watch is not None:
        control.watch(REVISION, args.watch, path)
        return
    if args.check_only:
        checked = preflight()
        print(json.dumps({"status": "passed", "prepared_sha256": checked["prepared_sha256"],
                          "source_files": len(checked["bindings"][0]["code"])}))
        return
    if path.exists():
        raise FileExistsError("Preserve the first declared memory diagnostic attempt")
    root = control.ROOT / DIRECTORY
    root.mkdir()
    lock = (control.ROOT / "diagnostics-v1/component.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state: dict[str, Any] = {"phase": "preparing", "runs": [], "service_paused": False,
                             "pid": os.getpid(), "started_epoch": time.time(),
                             "model_quality_or_serving_speed_qualified": False}
    success = False

    def save(phase: str) -> None:
        state.update(phase=phase, updated_epoch=time.time())
        control.write(path, state)

    def terminated(signum: int, frame: Any) -> None:
        raise SystemExit(f"Signal {signum}")

    signal.signal(signal.SIGTERM, terminated)
    signal.signal(signal.SIGINT, terminated)
    try:
        save("preparing")
        state["preflight"] = preflight()
        originals = [control.inspect(rank, control.ORIGINAL_IDS[rank]) for rank in (0, 1)]
        identities = [control.identity(row) for row in originals]
        if (not control.health(8000) or any(not r["State"]["Running"] for r in originals)
                or identities != state["preflight"]["originals"]):
            raise RuntimeError("Require the exact healthy original service before the component")
        state["originals"] = identities
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
                final = guard.activity()
                if not final["busy"] and final["counters"] == previous:
                    break
            time.sleep(5)
        else:
            raise TimeoutError("No idle window within thirty minutes; service unchanged")
        watchdog = subprocess.Popen(["python3", "-S", "-u", str(Path(__file__).resolve()),
            "--watch", str(os.getpid())], stdout=(root / "watchdog.log").open("xb"),
            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        state["watchdog_pid"] = watchdog.pid
        for _ in range(30):
            ack = control.ROOT / f"watchdog-{REVISION}.json"
            if ack.exists():
                value = json.loads(ack.read_bytes())
                if (value.get("ready") and value["pid"] == watchdog.pid
                        and value["parent"] == os.getpid()):
                    break
            time.sleep(1)
        else:
            raise RuntimeError("Independent restoration watchdog did not arm")
        state.update(service_paused=True, restore_deadline_epoch=time.time() + 2700)
        save("pausing_glm")
        for rank in (0, 1):
            control.run(rank, ["docker", "stop", "--timeout", "30", control.ORIGINAL_IDS[rank]],
                        timeout=90)
        state["memory_available_after_pause"] = []
        for rank in (0, 1):
            for _ in range(30):
                available = int(control.run(rank, ["python3", "-S", "-c",
                    ("from pathlib import Path;print(next(int(x.split()[1])*1024 for x in "
                     "Path('/proc/meminfo').read_text().splitlines() "
                     "if x.startswith('MemAvailable:')))" )]))
                if available >= 30 * 1024**3:
                    break
                time.sleep(2)
            else:
                raise RuntimeError("Original GLM did not release sufficient unified memory")
            state["memory_available_after_pause"].append(available)
        for profile in (False, True):
            stage = "profile" if profile else "timing"
            started_entries = {}
            for rank in (0, 1):
                control.run(rank, ["mkdir", "-p", str(root / f"rank{rank}")])
                argv = command(rank, profile=profile)
                name = argv[argv.index("--name") + 1]
                entry = {"rank": rank, "stage": stage, "container": name,
                         "started_epoch": time.time()}
                state["runs"].append(entry)
                started_entries[rank] = entry
                save("launching_" + stage)
                control.run(rank, argv)
            timeout = protocol()["profile_timeout_seconds" if profile else "component_timeout_seconds"]
            deadline = time.monotonic() + timeout + 30
            pending = {0, 1}
            while pending and time.monotonic() < deadline:
                for rank in sorted(pending):
                    entry = started_entries[rank]
                    info = control.inspect(rank, entry["container"])
                    if info["State"]["Running"]:
                        continue
                    entry.update(exit_code=info["State"]["ExitCode"],
                                 OOMKilled=info["State"]["OOMKilled"], finished_epoch=time.time())
                    if profile:
                        # Root is required for the driver's restricted counter access.
                        # Restore only our mounted result tree to the experiment owner.
                        script = (
                            "import os,sys;from pathlib import Path;"
                            "root=Path('/result');uid,gid=map(int,sys.argv[1:]);"
                            "paths=[root,*root.rglob('*')];"
                            "[(os.chown(p,uid,gid)) for p in paths if not p.is_symlink()]"
                        )
                        control.run(rank, ["docker", "run", "--rm", "--network", "none",
                            "--cpus", "1", "--memory", "128m", "--pids-limit", "64",
                            "-v", f"{root / f'rank{rank}'}:/result",
                            "--entrypoint", "/usr/bin/python3", control.IMAGE, "-S", "-c",
                            script, str(os.getuid()), str(os.getgid())], timeout=90)
                    log = control.run(rank, ["python3", "-S", "-c",
                        ("import subprocess,sys; r=subprocess.run(['docker','logs',sys.argv[1]],"
                         "stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=True);"
                         "sys.stdout.buffer.write(r.stdout)"), entry["container"]])
                    (root / f"rank{rank}-{stage}-container.log").write_text(log)
                    try:
                        report = control.remote_json(rank, root / f"rank{rank}/{stage}.json")
                        entry["report_status"] = report["status"]
                        control.write(root / f"rank{rank}-{stage}-report.json", report)
                        if report["status"] == "complete":
                            result = assess(report)
                            if result != report["assessment"]:
                                raise ValueError("Diagnostic assessment does not reproduce")
                            entry["independent_assessment"] = {
                                k: v for k, v in result.items() if k != "summaries"}
                    except (RuntimeError, ValueError, KeyError, OSError, TypeError,
                            subprocess.TimeoutExpired) as exc:
                        entry["report_error"] = repr(exc)
                    pending.remove(rank)
                    save(stage + "_running" if pending else stage + "_finished")
                if pending:
                    time.sleep(2)
            if pending:
                raise TimeoutError("Component exceeded its finite execution limit")
            if not all(not r["exit_code"] and not r["OOMKilled"] and not r.get("report_error")
                       and r.get("independent_assessment", {}).get("diagnostic_complete")
                       for r in started_entries.values()):
                raise RuntimeError("Complete exact diagnostic required before next stage")
        state["source_after"] = [control.preflight(rank, control.ROOT / SOURCE_DIRECTORY)
                                  for rank in (0, 1)]
        if state["source_after"] != state["preflight"]["bindings"]:
            raise ValueError("Source bindings changed during the component")
        state["input_bindings_after"] = [input_bindings(rank) for rank in (0, 1)]
        if state["input_bindings_after"] != state["preflight"]["input_bindings"]:
            raise ValueError("Fixture or profiler changed during the component")
        success = all(not r["exit_code"] and not r["OOMKilled"] and not r.get("report_error")
                      and r.get("independent_assessment", {}).get("diagnostic_complete")
                      for r in state["runs"])
        state["diagnostic_complete"] = success
        save("measured_pending_restoration")
    except BaseException as exc:
        state["error"] = repr(exc)
        save("failed_pending_restoration" if state["service_paused"] else "preparation_failed")
        raise
    finally:
        if state["service_paused"]:
            save("restoring")
            state["restoration"] = control.restore(REVISION, path)
            state["service_paused"] = False
            save("complete" if success else "failed_restored")
        lock.close()


if __name__ == "__main__":
    main()
