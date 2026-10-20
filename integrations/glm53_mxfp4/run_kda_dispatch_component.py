"""Run the frozen paired-graph component on Spark2 with one idle GLM pause and restoration."""

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

REVISION = "kdadirectv2"
DIRECTORY = "kda-direct-v2"


def command(stage: str, root: Path) -> list[str]:
    """Bound each fresh component process; use names covered by the restoration watchdog."""
    if stage not in ("initial", "confirm"):
        raise ValueError("Require one of the two predeclared processes")
    mode = "raw" if stage == "initial" else "aligned"
    result = root / "results" / stage
    argv = ["docker", "run", "-d", "--name", f"mxwave-glm53-full-{REVISION}-{mode}-r1",
            "--gpus", "all", "--network", "none", "--cpus", "2", "--memory", "3g",
            "--memory-swap", "3g", "--pids-limit", "512", "--user", f"{os.getuid()}:{os.getgid()}"]
    for source, target, readonly in ((root.with_name(DIRECTORY + "-src"), "/src", True),
                                     (control.ROOT, "/audit", True), (control.OLD, "/experiment", True),
                                     (result, "/result", False)):
        argv.extend(["-v", f"{source}:{target}" + (":ro" if readonly else "")])
    environment = {
        "PYTHONPATH": "/src:/experiment/triple-v2/src/integrations/zipserv",
        "PYTHONDONTWRITEBYTECODE": "1", "MXWAVE_GLM53_KDA_FUSED_MAX_TOKENS": "8",
        "MXWAVE_ZIPSERV_LIBRARY": "/audit/native/libmxwave_kda_zipserv.so",
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "TORCHINDUCTOR_CACHE_DIR": "/result/cache/inductor",
        "TRITON_CACHE_DIR": "/result/cache/triton", "CUDA_CACHE_PATH": "/result/cache/cuda",
        "XDG_CACHE_HOME": "/result/cache"}
    for key, value in environment.items():
        argv.extend(["-e", f"{key}={value}"])
    argv.extend(["--entrypoint", "/usr/bin/timeout", control.IMAGE, "--signal=TERM",
                 "--kill-after=10", "600", "/usr/bin/python3", "-u",
                 "/src/benchmark_kda_direct_dispatch.py", "--source", "/audit/source",
                 "--fixtures", "/audit/bf16-attention-v1/artifacts-v2", "--report",
                 "/result/report.json", "--paired-graphs"])
    return argv


def preflight() -> dict[str, Any]:
    """Verify frozen plan/source on both nodes and hash the real fixtures before any pause."""
    prepared_path = control.ROOT / (DIRECTORY + "-prepared.json")
    prepared = json.loads(prepared_path.read_bytes())
    protocol = prepared["protocol"]
    if (protocol["processes"] != ["initial", "confirm"] or not protocol["run_both_regardless_of_score"]
            or protocol["m4_graph_ratio_maximum"] != 1.02
            or protocol["identical_control_max_min_ratio_maximum"] != 1.02
            or protocol["replays_per_sample"] != 20 or protocol["samples_per_order"] != 30):
        raise ValueError("Frozen prospective component protocol changed")
    code = control.ROOT / (DIRECTORY + "-src")
    bindings = [control.preflight(rank, code) for rank in (0, 1)]
    if bindings != prepared["bindings"] or bindings[0] != bindings[1]:
        raise ValueError("Source, libraries or checkpoint metadata changed after preparation")
    script = """import hashlib,json,sys
from pathlib import Path
r=Path(sys.argv[1]);p=r/'bf16-attention-v1/artifacts-v2/prepared.json'
d=json.loads(p.read_bytes());out={'prepared_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'artifacts':{}}
for c in d['cases']:
 if not c['name'].startswith('kda-'):continue
 assert Path(c['artifact']).name==c['artifact']
 f=p.parent/c['artifact'];assert hashlib.sha256(f.read_bytes()).hexdigest()==c['artifact_sha256']
 out['artifacts'][c['artifact']]=c['artifact_sha256']
assert len(out['artifacts'])==2
print(json.dumps(out))
"""
    fixtures = json.loads(control.run(1, ["python3", "-S", "-c", script, str(control.ROOT)]))
    if fixtures != prepared["fixtures"]:
        raise ValueError("Fixture binding changed after preparation")
    return {"prepared_sha256": hashlib.sha256(prepared_path.read_bytes()).hexdigest(),
            "source_inventory_sha256": control.signature(bindings[0]["code"]),
            "fixtures": fixtures, "protocol": protocol}


def main() -> None:
    """Keep both attempts and the previous failure; always restore the same original GLM containers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", type=int)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    path = control.ROOT / (DIRECTORY + "-controller.json")
    if args.watch is not None:
        control.watch(REVISION, args.watch, path)
        return
    if args.check_only:
        print(json.dumps(preflight()))
        return
    if path.exists():
        raise FileExistsError("Preserve the first isolated component attempt")
    root = control.ROOT / DIRECTORY
    root.mkdir(exist_ok=True)
    lock = (control.ROOT / "diagnostics-v1/component.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state: dict[str, Any] = {"phase": "preparing", "runs": [], "service_paused": False,
                             "pid": os.getpid(), "started_epoch": time.time(),
                             "prototype_wired_into_adapter": False,
                             "previous_v1_performance_qualified_remains": False}
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
        if any(not row["State"]["Running"] or row["Image"] != control.IMAGE for row in originals):
            raise RuntimeError("Require the two original GLM ranks running before the component")
        state["originals"] = [control.identity(row) for row in originals]
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
            if ack.exists() and json.loads(ack.read_bytes()).get("ready"):
                break
            time.sleep(1)
        else:
            raise RuntimeError("Independent restoration watchdog did not arm")
        state.update(service_paused=True, restore_deadline_epoch=time.time() + 1800)
        save("pausing_glm")
        for rank in (0, 1):
            control.run(rank, ["docker", "stop", "--timeout", "30", control.ORIGINAL_IDS[rank]],
                        timeout=90)
        for _ in range(30):
            available = int(control.run(1, ["python3", "-S", "-c",
                ("from pathlib import Path;print(next(int(x.split()[1])*1024 for x in "
                 "Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))" )]))
            if available >= 30 * 1024**3:
                break
            time.sleep(2)
        else:
            raise RuntimeError("Original GLM did not release its unified-memory allocation")
        state["memory_available_after_pause"] = available
        # Both fresh processes were declared before measurement, independent of score.
        for stage in ("initial", "confirm"):
            control.run(1, ["mkdir", "-p", str(root / "results" / stage)])
            argv = command(stage, root)
            name = argv[argv.index("--name") + 1]
            entry: dict[str, Any] = {"stage": stage, "container": name, "started_epoch": time.time()}
            state["runs"].append(entry)
            save("starting_" + stage)
            control.run(1, argv)
            deadline = time.monotonic() + 625
            while time.monotonic() < deadline:
                info = control.inspect(1, name)
                if not info["State"]["Running"]:
                    entry["exit_code"] = info["State"]["ExitCode"]
                    entry["report"] = control.remote_json(1, root / "results" / stage / "report.json")
                    entry["finished_epoch"] = time.time()
                    control.write(root / (stage + ".json"), entry["report"])
                    if (entry["exit_code"] or entry["report"]["status"] != "complete"
                            or not entry["report"]["correctness_passed"]):
                        raise RuntimeError("Component correctness failed; inspect retained results")
                    save("finished_" + stage)
                    break
                time.sleep(2)
            else:
                raise TimeoutError("Component exceeded its finite execution limit")
        state["performance_qualified"] = all(r["report"]["performance_gate_passed"]
                                              for r in state["runs"])
        state["correctness_passed"] = True
        success = True
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
