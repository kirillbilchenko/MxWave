"""Run the bounded FP32 split timing comparison with original-service restoration."""

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
from fp32_speed_probe import assess, protocol

REVISION = "fp32speedv1"
DIRECTORY = "fp32-speed-v1"
SOURCE_DIRECTORY = DIRECTORY + "-src"
PREPARED_NAME = DIRECTORY + "-prepared.json"


def command(rank: int, *, profile: bool = False) -> list[str]:
    """Limit both timing jobs to declared ranks, unprivileged users and finite resources."""
    if rank not in (0, 1) or profile:
        raise ValueError("Require a declared rank and the unprofiled timing job")
    root = control.ROOT / DIRECTORY
    argv = ["docker", "run", "-d", "--name", f"mxwave-glm53-full-{REVISION}-raw-r{rank}",
            "--gpus", "all", "--network", "none", "--cpus", "2", "--memory", "8g",
            "--memory-swap", "8g", "--pids-limit", "512", "--user", f"{os.getuid()}:{os.getgid()}"]
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
        "TRITON_CACHE_DIR": "/result/cache/triton", "CUDA_CACHE_PATH": "/result/cache/cuda",
        "XDG_CACHE_HOME": "/result/cache", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }
    for key, value in environment.items():
        argv.extend(["-e", f"{key}={value}"])
    argv.extend(["--entrypoint", "/usr/bin/timeout", control.IMAGE, "--signal=TERM",
                 "--kill-after=10", str(protocol()["component_timeout_seconds"]),
                 "/usr/bin/python3", "-u", "/src/benchmark_fp32_speed.py",
                 "--report", "/result/timing.json", "--rank", str(rank),
                 "--plan", f"/audit/{DIRECTORY}-overlay/plan.json"])
    return argv


def input_bindings(rank: int) -> dict[str, Any]:
    """Bind the exact model evidence and prospective rank-specific selection without CUDA."""
    script = """import hashlib,json,pathlib,sys
r=pathlib.Path(sys.argv[1]);o=r/'fp32-speed-v1-overlay'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
plan=json.loads((o/'plan.json').read_bytes())
build=r/'fp32-split-v1-native/build';manifest=build/'native-build.json'
assert sha(manifest)==plan['native_build_sha256']
native=json.loads(manifest.read_bytes());assert native['status']=='built_pending_gpu_qualification'
files={}
for arm in native['arms']:
 for name,digest in arm['source_sha256'].items():
  f=build/arm['arm']/name;assert sha(f)==digest;files[str(f.relative_to(r))]=digest
 f=build/arm['library'];assert sha(f)==arm['library_sha256'];files[str(f.relative_to(r))]=sha(f)
for case in plan['selected_cases']:
 f=r/pathlib.PurePosixPath(case['artifact_path']).relative_to('/audit')
 assert sha(f)==case['artifact_sha256'];files[str(f.relative_to(r))]=sha(f)
rank=int(sys.argv[2]);parent=o/f'parent-rank{rank}.json'
assert sha(parent)==plan['parent_report_sha256'][str(rank)]
previous=json.loads(parent.read_bytes())
for case in previous['cases']:
 for row in case['checks']:
  if row['activation']!=plan['protocol']['activation']:continue
  for name in ('activation','original','rebuilt','candidate'):
   item=row['artifacts'][name];rel=pathlib.PurePosixPath(item['path'])
   assert not rel.is_absolute() and len(rel.parts)==1
   f=r/'fp32-split-v1'/f'rank{rank}'/rel
   assert f.stat().st_size==item['bytes'] and sha(f)==item['sha256']
   files[str(f.relative_to(r))]=item['sha256']
print(json.dumps({'plan_sha256':sha(o/'plan.json'),'native_build_sha256':sha(manifest),'parent_report_sha256':sha(parent),'files':files}))
"""
    return json.loads(control.run(rank, ["python3", "-S", "-c", script, str(control.ROOT), str(rank)]))


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
        raise ValueError("Parent model evidence or prospective plan changed after preparation")
    proof_path = control.ROOT / "fp32-split-v1-evidence-check.json"
    if hashlib.sha256(proof_path.read_bytes()).hexdigest() != prepared["parent_proof_sha256"]:
        raise ValueError("Preceding verified model evidence changed")
    return {"prepared_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bindings": bindings, "originals": prepared["originals"],
            "protocol": prepared["protocol"], "input_bindings": inputs}


def main() -> None:
    """Finish both paired GPU processes and restore the same original containers on exit."""
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
        raise FileExistsError("Preserve the first declared FP32 precision attempt")
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
        state.update(service_paused=True, restore_deadline_epoch=time.time() + 1800)
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
        for profile in (False,):
            stage = "timing"
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
            timeout = protocol()["component_timeout_seconds"]
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
            raise ValueError("Parent model evidence or prospective plan changed during the component")
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
