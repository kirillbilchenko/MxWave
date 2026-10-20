"""One authorized serving pause, one model load, and automatic original-container restoration."""

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
from typing import Any

import full_model_control as control


def main() -> None:
    """Reuse the qualified launcher/watchdog, stopping at the source/runtime correctness gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--attention-mapping", action="store_true")
    args = parser.parse_args()
    revision = args.revision
    if not revision.isalnum():
        raise ValueError("Use an alphanumeric immutable revision")
    root, code = control.ROOT, control.ROOT / f"full-{revision}-src"
    path = root / f"full-controller-{revision}.json"
    if path.exists():
        raise FileExistsError("Preserve prior single-load diagnostic evidence")
    lock = (root / "diagnostics-v1/component.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state: dict[str, Any] = {"phase": "preparing", "controller_pid": os.getpid(),
        "started_epoch": time.time(), "service_paused": False, "purpose": "One-load correctness diagnostic"}

    def save(phase: str) -> None:
        state.update(phase=phase, updated_epoch=time.time())
        control.write(path, state)

    def terminate(signum: int, frame: Any) -> None:
        raise SystemExit(f"Signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    save("preparing")
    success = False
    try:
        evidence = {}
        for name in ("mxfp8v2", "moev1", "mhcv2", "dispatchv1"):
            proof = json.loads((root / f"diagnostics-v1/controller-{name}.json").read_bytes())
            if proof["phase"] != "complete_idle":
                raise ValueError("Finish bounded numerical/warmup component gates first")
            evidence[name] = proof["attempts"][-1]["report"]
        if not (evidence["mxfp8v2"]["passed"] and evidence["moev1"]["passed"]
                and evidence["mhcv2"]["zero_monitored_jit_after_warmup"]
                and evidence["dispatchv1"]["passed"]
                and evidence["dispatchv1"]["checker"]["passed"]):
            raise ValueError("A component gate did not pass")
        checker_sha = hashlib.sha256((code / "diagnostic_worker.py").read_bytes()).hexdigest()
        if evidence["dispatchv1"]["code_sha256"]["diagnostic_worker.py"] != checker_sha:
            raise ValueError("Qualify checker and graph switching on a tiny engine first")
        if args.attention_mapping:
            audit = json.loads((root / "diagnostics-v1/attention-dispatch-v1.json").read_bytes())
            mapping_sha = hashlib.sha256((code / "attention_mapping.py").read_bytes()).hexdigest()
            if (not audit["passed"] or audit["unexpected_quantized_selectors_after"] != 0
                    or audit["runtime_source_sha256"]["mapping"] != mapping_sha):
                raise ValueError("Qualify this exact attention metadata repair on CPU first")
            state["attention_mapping_audit"] = {
                "mapping_sha256": mapping_sha, "fused_counts": audit["fused_counts"]}
        state["component_status"] = {name: value["status"] for name, value in evidence.items()}
        bound = [control.preflight(rank, code) for rank in (0, 1)]
        if bound[0] != bound[1]:
            raise ValueError("Both Sparks must use identical source/code/library metadata")
        state["preflight"] = bound[0]
        originals = [control.inspect(rank, control.ORIGINAL_IDS[rank]) for rank in (0, 1)]
        if any(value["Image"] != control.IMAGE or not value["State"]["Running"] for value in originals):
            raise ValueError("The authorized original service identity or state changed")
        state["originals"] = [control.identity(value) for value in originals]
        launches = [control.engine(rank, "stock", revision, originals[rank], bound[0]["vocabulary"],
                                   attention_mapping=args.attention_mapping)
                    for rank in (0, 1)]
        for argv, metadata in launches:
            for command in (argv, metadata["command"]):
                command[command.index("--worker-extension-cls") + 1] = (
                    "attention_diagnostic.AttentionDiagnosticWorker" if args.attention_mapping
                    else "diagnostic_worker.DiagnosticWorker")
        state["launches"] = [value[1] for value in launches]
        # Validate actual image CLI/plugin discovery before the authorized pause.
        cpu = ["docker", "run", "--rm", "--network", "none", "--cpus", "2",
            "--memory", "2g", "--memory-swap", "2g", "-v", f"{code}:/src:ro",
            "-v", f"{root / 'source'}:/model:ro", "-e", "PYTHONPATH=/src",
            "-e", "CUDA_VISIBLE_DEVICES=", "-e", "NVIDIA_VISIBLE_DEVICES=void",
            "-e", "MXWAVE_VALIDATE_ONLY=1", "-e", "MXWAVE_MOE_MODE=stock",
            "--entrypoint", "/usr/bin/python3", control.IMAGE, "-u", "/src/engine_boot.py",
            *launches[0][1]["command"]]
        if args.attention_mapping:
            additions = ["-e", "MXWAVE_GLM53_ATTENTION_MAPPING=1", "-e",
                         "PYTHONPATH=/src:/overlay/overlay"]
            for source, target, access in launches[0][1]["mounts"]:
                if target == "/overlay" or target.startswith(
                        "/usr/local/lib/python3.12/dist-packages/vllm/"):
                    additions.extend(["-v", f"{source}:{target}:{access}"])
            cpu[cpu.index("--entrypoint"):cpu.index("--entrypoint")] = additions
        state["cpu_validation"] = control.run(0, cpu, timeout=180)
        spec = importlib.util.spec_from_file_location("guard", control.OLD /
            "triple-v2/src/integrations/marlin_scales/quiet_window.py")
        assert spec and spec.loader
        guard = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(guard)
        save("waiting_for_idle")
        previous, quiet = None, 0
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            now = guard.activity()
            quiet = quiet + 1 if not now["busy"] and previous == now["counters"] else 0
            previous = now["counters"]
            if quiet >= 4:
                state["idle_before_pause"] = now
                break
            time.sleep(5)
        else:
            raise TimeoutError("No idle window for the authorized full-model check")
        watchdog = subprocess.Popen(["python3", "-u", str(code / "full_model_control.py"),
            "--revision", revision, "--watch", str(os.getpid())],
            stdout=(root / f"watchdog-{revision}.log").open("xb"), stderr=subprocess.STDOUT,
            start_new_session=True)
        state["watchdog_pid"] = watchdog.pid
        for _ in range(20):
            ack = root / f"watchdog-{revision}.json"
            if ack.exists() and json.loads(ack.read_bytes()).get("ready"):
                break
            time.sleep(.5)
        else:
            raise RuntimeError("Restoration watchdog did not arm")
        if guard.activity()["busy"]:
            raise RuntimeError("GLM became active; leave the serving containers running")
        state.update(service_paused=True, restore_deadline_epoch=time.time() + 2400)
        save("pausing_service")
        for rank in (0, 1):
            control.run(rank, ["docker", "stop", "--timeout", "30", control.ORIGINAL_IDS[rank]], 90)
        save("loading_diagnostic_model")
        for rank in (1, 0):
            control.run(rank, launches[rank][0], 120)
        deadline = time.monotonic() + 1200
        while time.monotonic() < deadline:
            if control.health(18784):
                break
            for rank in (0, 1):
                if not control.inspect(rank, launches[rank][1]["name"])["State"]["Running"]:
                    raise RuntimeError("Diagnostic engine exited during loading")
            time.sleep(5)
        else:
            raise TimeoutError("Diagnostic model did not become ready within twenty minutes")
        state["ready_epoch"] = time.time()
        save("checking_loaded_model")
        results = root / f"run-{revision}/results"
        results.mkdir(exist_ok=True)
        client_name = f"mxwave-glm53-bench-{revision}-stock"
        client = ["docker", "run", "-d", "--name", client_name, "--network", "host",
            "--cpus", "2", "--memory", "1536m", "--memory-swap", "1536m",
            "--user", f"{os.getuid()}:{os.getgid()}", "-e", "CUDA_VISIBLE_DEVICES=",
            "-e", "NVIDIA_VISIBLE_DEVICES=void", "-v", f"{control.AUDIT}:/audit:ro",
            "-v", f"{results}:/results", "--entrypoint", "/usr/bin/python3", control.IMAGE,
            "-u", f"/audit/full-model-v1/full-{revision}-src/check_full_model.py",
            "--root", "/results", "--protocol", "/audit/full-model-v1/run-v4/results/protocol.json",
            "--vocabulary", str(bound[0]["vocabulary"])]
        control.run(0, client)
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            info = control.inspect(0, client_name)["State"]
            if not info["Running"]:
                if info["ExitCode"]:
                    raise RuntimeError("Single-load diagnostic failed; retain its partial report")
                state["result"] = json.loads((results / "model-diagnostic.json").read_bytes())
                success = state["result"]["status"] == "complete"
                if not success:
                    raise RuntimeError("Diagnostic client did not complete")
                break
            time.sleep(5)
        else:
            raise TimeoutError("Single-load diagnostic exceeded fifteen minutes")
    except BaseException as exc:
        state["error"] = repr(exc)
        save("failed_pending_restoration" if state["service_paused"] else "preparation_failed")
        raise
    finally:
        if state["service_paused"]:
            save("restoring")
            state["restoration"] = control.restore(revision, path)
            state["service_paused"] = False
            save("complete" if success else "failed_restored")
        lock.close()


if __name__ == "__main__":
    main()
