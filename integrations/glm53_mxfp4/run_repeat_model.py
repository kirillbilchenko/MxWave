"""Detached one-load reference diagnostic with exact identity restoration and a watchdog."""

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
from run_quality_model import cli_check, launch, wait_container


def repeat_launch(
    rank: int, revision: str, original: dict[str, Any], vocabulary: int, manifest_sha: str,
) -> tuple[list[str], dict[str, Any]]:
    """Load the uncompressed reference once; pin cuBLAS workspace before CUDA initialization."""
    argv, meta = launch(rank, revision, "control", original, vocabulary, manifest_sha)
    for command in (argv, meta["command"]):
        command[command.index("--worker-extension-cls") + 1] = "repeat_worker.RepeatWorker"
    key, value = "CUBLAS_WORKSPACE_CONFIG", ":4096:8"
    meta["environment"][key] = value
    at = argv.index("--entrypoint")
    argv[at:at] = ["-e", key + "=" + value]
    return argv, meta


def main() -> None:
    """Run one bounded reference investigation; reuse existing quality and throughput evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--watch", type=int)
    args = parser.parse_args()
    revision = args.revision
    if not revision.isalnum():
        raise ValueError("Use a fresh alphanumeric revision")
    code = c.ROOT / f"full-{revision}-src"
    path = c.ROOT / f"repeat-model-controller-{revision}.json"
    if args.watch is not None:
        c.watch(revision, args.watch, path)
        return
    if path.exists():
        raise FileExistsError("Preserve prior reference diagnostics")
    lock = (c.ROOT / "diagnostics-v1/component.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state: dict[str, Any] = {
        "phase": "preparing", "service_paused": False, "started_epoch": time.time(),
        "purpose": "One reference load: policy ablation, first MoE snapshots and real-input replay",
        "throughput_benchmark_run": False, "quality_qualification_attempted": False,
    }

    def save(phase: str) -> None:
        state.update(phase=phase, updated_epoch=time.time())
        c.write(path, state)

    def terminated(signum: int, frame: Any) -> None:
        raise SystemExit(f"Signal {signum}")

    signal.signal(signal.SIGTERM, terminated)
    signal.signal(signal.SIGINT, terminated)
    success = False
    save("preparing")
    try:
        originals = [c.inspect(rank, c.ORIGINAL_IDS[rank]) for rank in (0, 1)]
        if any(not x["State"]["Running"] for x in originals):
            raise RuntimeError("Require the original service running before preparation")
        state["originals"] = [c.identity(x) for x in originals]
        bound = [c.preflight(rank, code) for rank in (0, 1)]
        if bound[0] != bound[1]:
            raise RuntimeError("Code/source bindings differ between Sparks")
        state["bindings"] = bound
        manifest_sha = hashlib.sha256((c.ROOT / "vocab-v2/prepared.json").read_bytes()).hexdigest()
        launches = [
            repeat_launch(rank, revision, originals[rank], bound[0]["vocabulary"], manifest_sha)
            for rank in (0, 1)
        ]
        state["launches"] = [x[1] for x in launches]
        state["cpu_cli_validation"] = cli_check(launches[0][1])
        results = c.ROOT / f"run-{revision}/results"
        results.mkdir(parents=True, exist_ok=False)
        protocol = json.loads((c.ROOT / "run-qualitybestv2/results/protocol.json").read_bytes())
        if len(protocol["quality"]) != 16:
            raise ValueError("Frozen diagnostic contexts changed")
        c.write(results / "protocol.json", protocol)
        spec = importlib.util.spec_from_file_location(
            "idle_guard", c.OLD / "triple-v2/src/integrations/marlin_scales/quiet_window.py"
        )
        assert spec and spec.loader
        guard = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(guard)
        previous, quiet = None, 0
        for _ in range(360):
            activity = guard.activity()
            quiet = quiet + 1 if not activity["busy"] and activity["counters"] == previous else 0
            previous = activity["counters"]
            state["last_activity"] = activity
            save("waiting_for_idle")
            if quiet >= 3:
                break
            time.sleep(5)
        else:
            raise TimeoutError("No idle window within thirty minutes")
        watchdog = subprocess.Popen(
            ["python3", "-u", str(Path(__file__).resolve()), "--revision", revision,
             "--watch", str(os.getpid())],
            stdout=(c.ROOT / f"watchdog-{revision}.log").open("xb"),
            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
        )
        state["watchdog_pid"] = watchdog.pid
        for _ in range(30):
            ack = c.ROOT / f"watchdog-{revision}.json"
            if ack.exists() and json.loads(ack.read_bytes()).get("ready"):
                break
            time.sleep(1)
        else:
            raise RuntimeError("Independent restoration watchdog did not arm")
        final = guard.activity()
        if final["busy"] or final["counters"] != previous:
            raise RuntimeError("GLM became busy before pause; production remains running")
        state.update(service_paused=True, restore_deadline_epoch=time.time() + 3600)
        save("pausing_glm")
        for rank in (0, 1):
            c.run(rank, ["docker", "stop", "--timeout", "30", c.ORIGINAL_IDS[rank]], timeout=90)
        save("starting_reference")
        for rank in (1, 0):
            c.run(rank, launches[rank][0], timeout=120)
        deadline = time.monotonic() + 1200
        while time.monotonic() < deadline:
            if c.health(18784):
                break
            if any(not c.inspect(rank, launches[rank][1]["name"])["State"]["Running"]
                   for rank in (0, 1)):
                raise RuntimeError("Reference engine exited during startup")
            time.sleep(5)
        else:
            raise TimeoutError("Reference startup deadline")
        state["ready_epoch"] = time.time()
        save("measuring_reference")
        client_name = f"mxwave-glm53-bench-{revision}-raw"
        argv = [
            "docker", "run", "-d", "--name", client_name, "--network", "host",
            "--cpus", "2", "--memory", "2g", "--memory-swap", "2g",
            "--user", f"{os.getuid()}:{os.getgid()}",
            "-e", "NVIDIA_VISIBLE_DEVICES=void", "-e", "CUDA_VISIBLE_DEVICES=",
            "-e", f"PYTHONPATH=/audit/full-model-v1/full-{revision}-src",
            "-v", f"{c.AUDIT}:/audit:ro", "-v", f"{results}:/results",
            "--entrypoint", "/usr/bin/python3", c.IMAGE, "-u",
            f"/audit/full-model-v1/full-{revision}-src/measure_repeat_model.py",
            "--root", "/results", "--vocabulary", str(bound[0]["vocabulary"]),
        ]
        c.run(0, argv)
        state["client_state"] = wait_container(0, client_name, 900)
        if state["client_state"]["ExitCode"]:
            raise RuntimeError("Reference diagnostic client failed; retain partial evidence")
        state["measurement"] = json.loads((results / "reference-repeat-measurement.json").read_bytes())
        if state["measurement"]["status"] != "complete":
            raise RuntimeError("Reference diagnostic evidence is incomplete")
        success = True
        save("measured_pending_restoration")
    except BaseException as exc:
        state["error"] = repr(exc)
        save("failed_pending_restoration" if state["service_paused"] else "preparation_failed")
        raise
    finally:
        if state["service_paused"]:
            save("restoring")
            state["restoration"] = c.restore(revision, path)
            state["service_paused"] = False
            save("complete" if success else "failed_restored")
        lock.close()


if __name__ == "__main__":
    main()
