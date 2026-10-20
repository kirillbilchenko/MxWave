"""Load one qualified v9 model for idle-only eager dispatch switching, then restore GLM."""

from __future__ import annotations

import argparse
import ast
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
import run_quality_model as quality

REVISION = "eagerdispatchv1"
PREPARED_NAME = "eager-dispatch-model-v1-prepared.json"
WORKER_CLASS = "eager_dispatch_worker.EagerDispatchWorker"
CAPTURE_SIZES = [1]
CLIENT_SCRIPT = "measure_eager_model_dispatch.py"
REPORT_NAME = "eager-dispatch-model-review.json"
ENTRY_SCRIPT = Path(__file__).resolve()
SCOPE = "Two host dispatch modes on the same resident v9 candidate; no stock baseline reload"
ARM_LABEL = "candidate"
CONTROL_REVISION = "stablequalityv1"
GPU_FIXTURE_SCRIPT: str | None = None
CLIENT_MEMORY = "4g"
CLIENT_WAIT_SECONDS = 1800


def launch(rank: int, original: dict[str, Any], vocabulary: int,
           manifest_sha: str) -> tuple[list[str], dict[str, Any]]:
    """Keep v9 storage and transport, with an explicitly recorded graph capture configuration."""
    if ARM_LABEL not in ("control", "candidate"):
        raise ValueError("Unknown single-load model arm")
    argv, meta = quality.launch(rank, REVISION, ARM_LABEL, original, vocabulary, manifest_sha)
    for command in (argv, meta["command"]):
        command[command.index("--worker-extension-cls") + 1] = WORKER_CLASS
        command[command.index("--compilation-config") + 1] = json.dumps({
            "mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": CAPTURE_SIZES,
            "max_cudagraph_capture_size": max(CAPTURE_SIZES)})
    extra = {"MXWAVE_STABLE_MOE_DISPATCH": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    meta["environment"].update(extra)
    for key, value in extra.items():
        at = next((i for i, entry in enumerate(argv) if entry.startswith(key + "=")), None)
        if at is None:
            offset = argv.index("--entrypoint")
            argv[offset:offset] = ["-e", key + "=" + value]
        else:
            argv[at] = key + "=" + value
    return argv, meta


def native_check(code: Path) -> list[dict[str, Any]]:
    """Reuse the hash-bound existing v9 source/library/dataset preflight before any service pause."""
    runner = code / "run_quality_model.py"
    tree = ast.parse(runner.read_bytes())
    main_function = next(node for node in tree.body
                         if isinstance(node, ast.FunctionDef) and node.name == "main")
    checks = [node.value.value for node in ast.walk(main_function)
              if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
              and isinstance(node.value.value, str)
              and any(isinstance(target, ast.Name) and target.id == "check" for target in node.targets)]
    if len(checks) != 1 or "qualification-binding.json" not in checks[0]:
        raise ValueError("Existing v9 qualification preflight changed")
    return [json.loads(c.run(rank, ["python3", "-S", "-c", checks[0], str(c.ROOT), str(code)]))
            for rank in (0, 1)]


def main() -> None:
    """Arm independent restoration before an idle pause and keep every measured mode/order."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", type=int)
    args = parser.parse_args()
    path = c.ROOT / f"quality-model-controller-{REVISION}.json"
    if args.watch is not None:
        c.watch(REVISION, args.watch, path)
        return
    if path.exists():
        raise FileExistsError("Preserve the first single-load model comparison")
    code = c.ROOT / f"full-{REVISION}-src"
    prepared_path = c.ROOT / PREPARED_NAME
    prepared = json.loads(prepared_path.read_bytes())
    root = c.ROOT / f"run-{REVISION}"
    results = root / "results"
    lock = (c.ROOT / "diagnostics-v1/component.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state: dict[str, Any] = {"phase": "preparing", "service_paused": False,
        "started_epoch": time.time(), "model_loads": 1,
        "scope": SCOPE}
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
        if bindings != prepared["bindings"] or bindings[0] != bindings[1]:
            raise ValueError("Prepared model code/source binding changed")
        dependency_path = c.ROOT / "kda-eager-v1-controller.json"
        dependency = json.loads(dependency_path.read_bytes())
        if (dependency["phase"] != "complete" or not dependency["performance_qualified"]
                or not dependency["correctness_passed"] or len(dependency["runs"]) != 2):
            raise ValueError("Eager-only component did not pass both frozen processes")
        for name in ("kda_eager_dispatch.py", "eager_dispatch_worker.py", "kda_runtime.py",
                     "kda_direct_dispatch.py", "moe_runtime.py", "scale_tiles.py"):
            if bindings[0]["code"][name] != prepared["component_source_sha256"][name]:
                raise ValueError("Qualified component/runtime source changed: " + name)
        state["component_proof_sha256"] = hashlib.sha256(dependency_path.read_bytes()).hexdigest()
        state["prepared_sha256"] = hashlib.sha256(prepared_path.read_bytes()).hexdigest()
        state["candidate_native"] = native_check(code)
        if state["candidate_native"][0] != state["candidate_native"][1]:
            raise ValueError("Native/source/dataset preflight differs across ranks")
        originals = [c.inspect(rank, c.ORIGINAL_IDS[rank]) for rank in (0, 1)]
        if any(not row["State"]["Running"] or row["Image"] != c.IMAGE for row in originals):
            raise RuntimeError("Require both original GLM containers running before this comparison")
        state["originals"] = [c.identity(row) for row in originals]
        manifest_sha = hashlib.sha256((c.ROOT / "vocab-v2/prepared.json").read_bytes()).hexdigest()
        launches = [launch(rank, originals[rank], bindings[0]["vocabulary"], manifest_sha)
                    for rank in (0, 1)]
        saved = json.loads((c.ROOT / f"quality-model-controller-{CONTROL_REVISION}.json").read_bytes())
        c.validate_transport_control(saved["launches"][ARM_LABEL], [item[1] for item in launches])
        state["launches"] = [item[1] for item in launches]
        state["cpu_cli_validation"] = quality.cli_check(launches[0][1])
        if not results.is_dir() or any((results / name).exists() for name in
                (REPORT_NAME, "qualified-a-decode-quality.json")):
            raise ValueError("Prepared results missing or already measured")
        for name, expected in prepared["protocol_files_sha256"].items():
            if hashlib.sha256((results / name).read_bytes()).hexdigest() != expected:
                raise ValueError("Frozen protocol changed: " + name)
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
            raise TimeoutError("No idle service window within thirty minutes")
        watchdog = subprocess.Popen(["python3", "-S", "-u", str(ENTRY_SCRIPT),
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
            raise RuntimeError("Service became active before the pause")
        state.update(service_paused=True, restore_deadline_epoch=time.time() + 5400)
        save("pausing_glm")
        for rank in (0, 1):
            c.run(rank, ["docker", "stop", "--timeout", "30", c.ORIGINAL_IDS[rank]], timeout=90)
        if GPU_FIXTURE_SCRIPT is not None:
            save("gpu_fixture_before_model_load")
            fixture = ["docker", "run", "--rm", "--name", f"mxwave-glm53-bench-{REVISION}-fixture",
                "--gpus", "all", "--network", "none", "--ipc", "host", "--cpus", "2",
                "--memory", "6g", "--memory-swap", "6g", "--pids-limit", "256",
                "--ulimit", "memlock=-1", "-e", "CUDA_VISIBLE_DEVICES=0",
                "-e", "PYTHONPATH=/work", "-e", "PYTHONDONTWRITEBYTECODE=1",
                "-v", f"{code}:/work:ro", "--entrypoint", "/usr/bin/timeout", c.IMAGE,
                "--signal=TERM", "--kill-after=5", "180", "/usr/bin/python3", "-u",
                "/work/" + GPU_FIXTURE_SCRIPT]
            capture = ("import subprocess,json,sys; "
                "p=subprocess.run(json.loads(sys.argv[1]),stdout=subprocess.PIPE,"
                "stderr=subprocess.STDOUT,text=True); "
                "print(json.dumps({'exit_code':p.returncode,'output':p.stdout}))")
            outcome = json.loads(c.run(1, ["python3", "-S", "-c", capture,
                                           json.dumps(fixture)], timeout=240))
            (root / "gpu-fixture.log").write_text(outcome["output"])
            if (outcome["exit_code"] or not outcome["output"].rstrip().endswith(
                    "MXWAVE_DECODE_FIXTURE_PASSED")):
                raise RuntimeError("Pre-model CUDA fixture failed; see retained gpu-fixture.log")
            state["gpu_fixture"] = next(json.loads(line) for line in outcome["output"].splitlines()
                                       if line.startswith('{"status": "passed"'))
            c.write(results / "gpu-fixture-check.json", state["gpu_fixture"])
        for rank in (1, 0):
            c.run(rank, launches[rank][0], timeout=120)
        deadline = time.monotonic() + 1200
        save("loading_one_model")
        while time.monotonic() < deadline:
            if c.health(18784):
                break
            if any(not c.inspect(rank, launches[rank][1]["name"])["State"]["Running"]
                   for rank in (0, 1)):
                raise RuntimeError("Experimental model exited during startup")
            time.sleep(5)
        else:
            raise TimeoutError("Single model startup exceeded twenty minutes")
        state["ready_epoch"] = time.time()
        stage = "raw" if ARM_LABEL == "control" else "aligned"
        name = f"mxwave-glm53-bench-{REVISION}-{stage}"
        argv = ["docker", "run", "-d", "--name", name, "--network", "host", "--cpus", "2",
                "--memory", CLIENT_MEMORY, "--memory-swap", CLIENT_MEMORY,
                "--user", f"{os.getuid()}:{os.getgid()}",
                "-e", "NVIDIA_VISIBLE_DEVICES=void", "-e", "CUDA_VISIBLE_DEVICES=",
                "-e", f"PYTHONPATH=/audit/full-model-v1/full-{REVISION}-src",
                "-v", f"{c.AUDIT}:/audit:ro", "-v", f"{results}:/results",
                "--entrypoint", "/usr/bin/timeout", c.IMAGE, "--signal=TERM", "--kill-after=10",
                str(CLIENT_WAIT_SECONDS), "/usr/bin/python3", "-u",
                f"/audit/full-model-v1/full-{REVISION}-src/{CLIENT_SCRIPT}",
                "--root", "/results", "--vocabulary", str(bindings[0]["vocabulary"])]
        save("measuring_quality_then_interleaved_speed_then_profiles")
        c.run(0, argv)
        state["client_state"] = quality.wait_container(0, name, CLIENT_WAIT_SECONDS + 30)
        if state["client_state"]["ExitCode"]:
            raise RuntimeError("Dispatch comparison client failed; retain partial report and logs")
        state["review"] = json.loads((results / REPORT_NAME).read_bytes())
        if state["review"]["status"] != "complete":
            raise RuntimeError("Dispatch model report is incomplete")
        success = True
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
