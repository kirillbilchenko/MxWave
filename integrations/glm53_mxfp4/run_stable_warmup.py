"""Run two bounded fresh-cache warmup components with exact original-service restoration."""

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
from benchmark_stable_warmup import assess, protocol

REVISION = "stablewarmv1"
DIRECTORY = "stable-warmup-v1"


def command(rank: int) -> list[str]:
    """Constrain an isolated component to its own cache, GPU scratch and report directory."""
    if rank not in (0, 1):
        raise ValueError("Require one of the two declared Spark ranks")
    root = control.ROOT / DIRECTORY
    argv = ["docker", "run", "-d", "--name", f"mxwave-glm53-full-{REVISION}-raw-r{rank}",
            "--gpus", "all", "--network", "none", "--cpus", "2", "--memory", "4g",
            "--memory-swap", "4g", "--pids-limit", "512",
            "--user", f"{os.getuid()}:{os.getgid()}"]
    for source, target, readonly in (
        (control.ROOT / (DIRECTORY + "-src"), "/src", True),
        (control.ROOT, "/audit", True), (control.OLD, "/experiment", True),
        (root / f"rank{rank}", "/result", False),
    ):
        argv.extend(["-v", f"{source}:{target}" + (":ro" if readonly else "")])
    environment = {
        "PYTHONPATH": "/src:/experiment/triple-v2/src/integrations/zipserv",
        "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "TORCHINDUCTOR_CACHE_DIR": "/result/cache/inductor",
        "TRITON_CACHE_DIR": "/result/cache/triton", "CUDA_CACHE_PATH": "/result/cache/cuda",
        "XDG_CACHE_HOME": "/result/cache",
    }
    for key, value in environment.items():
        argv.extend(["-e", f"{key}={value}"])
    argv.extend(["--entrypoint", "/usr/bin/timeout", control.IMAGE, "--signal=TERM",
                 "--kill-after=10", "600", "/usr/bin/python3", "-u",
                 "/src/benchmark_stable_warmup.py", "--report", "/result/report.json"])
    return argv


def preflight() -> dict[str, Any]:
    """Require the frozen protocol, unchanged sources and preceding verified evidence."""
    path = control.ROOT / (DIRECTORY + "-prepared.json")
    prepared = json.loads(path.read_bytes())
    if prepared["protocol"] != protocol():
        raise ValueError("Prepared protocol differs from the prospective component contract")
    bindings = [control.preflight(rank, control.ROOT / (DIRECTORY + "-src"))
                for rank in (0, 1)]
    if bindings != prepared["bindings"] or bindings[0] != bindings[1]:
        raise ValueError("Source bindings changed after preparation")
    proof_path = control.ROOT / "latency-probe-v1-cross-host-verification.json"
    if hashlib.sha256(proof_path.read_bytes()).hexdigest() != prepared["parent_proof_sha256"]:
        raise ValueError("Preceding independent latency verification changed")
    return {"prepared_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bindings": bindings, "originals": prepared["originals"],
            "protocol": prepared["protocol"]}


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
        raise FileExistsError("Preserve the first declared warmup component attempt")
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
        state.update(service_paused=True, restore_deadline_epoch=time.time() + 2100)
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
        for rank in (0, 1):
            control.run(rank, ["mkdir", "-p", str(root / f"rank{rank}")])
            argv = command(rank)
            name = argv[argv.index("--name") + 1]
            entry = {"rank": rank, "container": name, "started_epoch": time.time()}
            state["runs"].append(entry)
            save("launching_components")
            control.run(rank, argv)
        deadline = time.monotonic() + 625
        pending = {0, 1}
        while pending and time.monotonic() < deadline:
            for rank in sorted(pending):
                entry = state["runs"][rank]
                info = control.inspect(rank, entry["container"])
                if info["State"]["Running"]:
                    continue
                entry.update(exit_code=info["State"]["ExitCode"],
                             OOMKilled=info["State"]["OOMKilled"], finished_epoch=time.time())
                log = control.run(rank, ["python3", "-S", "-c",
                    ("import subprocess,sys; r=subprocess.run(['docker','logs',sys.argv[1]],"
                    "stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=True);"
                    "sys.stdout.buffer.write(r.stdout)"), entry["container"]])
                (root / f"rank{rank}-container.log").write_text(log)
                try:
                    report = control.remote_json(rank, root / f"rank{rank}/report.json")
                    entry["report"] = report
                    control.write(root / f"rank{rank}-report.json", report)
                    if report["status"] == "complete":
                        entry["independent_assessment"] = assess(report)
                        if entry["independent_assessment"] != report["assessment"]:
                            raise ValueError("Component assessment does not reproduce")
                except (RuntimeError, ValueError, KeyError, OSError, TypeError,
                        subprocess.TimeoutExpired) as exc:
                    entry["report_error"] = repr(exc)
                pending.remove(rank)
                save("components_running" if pending else "components_finished")
            if pending:
                time.sleep(2)
        if pending:
            raise TimeoutError("Component exceeded its finite execution limit")
        state["source_after"] = [control.preflight(rank, control.ROOT / (DIRECTORY + "-src"))
                                  for rank in (0, 1)]
        if state["source_after"] != state["preflight"]["bindings"]:
            raise ValueError("Source bindings changed during the component")
        success = all(not r["exit_code"] and not r["OOMKilled"] and not r.get("report_error")
                      and r.get("independent_assessment", {}).get("component_passed")
                      for r in state["runs"])
        state["component_passed"] = success
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
