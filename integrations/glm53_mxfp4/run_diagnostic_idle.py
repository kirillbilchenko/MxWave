"""Run a bounded component on Spark2 and cancel only our container if GLM wakes."""

from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Any

ROOT = Path("/home/kirya/local-spark/experiments/glm53-mxfp4-audit-2026-10-15/full-model-v1")
OLD = Path("/home/kirya/local-spark/experiments/zipserv-vllm031-2026-10-14")
IMAGE = "sha256:39a03677234ea74e35544d301bd14716c1ae9695ad8d91b16e2c76545e26e8d2"


def main() -> None:
    """Keep the component independent of SSH and preserve every interrupted attempt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--probe", choices=("mxfp8", "mhc", "moe", "checker", "dispatch",
                                           "bf16_attention", "bf16_prefill", "bf16_overlap",
                                           "bf16_kda", "bf16_small_batch", "bf16_kda_small",
                                           "bf16_vocab", "bf16_cache"),
                        required=True)
    args = parser.parse_args()
    # The vocabulary fixture independently caps CUDA allocation at 1.5 GiB.
    # Its previous measured peak was 1.16 GB; reserve 3.5 GiB host plus 2 GiB
    # GPU/headroom rather than using the generic 5 GiB host container.
    host_limit_mib = 3584 if args.probe == "bf16_vocab" else 5120
    required_available = (5.5 if args.probe == "bf16_vocab" else 6.5) * 1024**3
    if not args.revision.isalnum():
        raise ValueError("Use an alphanumeric immutable revision")
    root = ROOT / "diagnostics-v1"
    root.mkdir(exist_ok=True)
    state_path = root / f"controller-{args.revision}.json"
    if state_path.exists():
        raise FileExistsError("Preserve prior component evidence")
    if args.probe.startswith("bf16_"):
        restoration = ("restoration-bf16kernelsv2.json" if args.probe == "bf16_cache"
                       else "restoration-diagv4.json")
        restored = json.loads((ROOT / restoration).read_bytes())
        if restored.get("status") != "complete" or restored.get("health") != 200:
            raise ValueError("Restore the original GLM service before this component")
    lock = (root / "component.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    spec = importlib.util.spec_from_file_location(
        "guard", OLD / "triple-v2/src/integrations/marlin_scales/quiet_window.py")
    assert spec and spec.loader
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    state: dict[str, Any] = {"phase": "waiting_for_idle", "probe": args.probe,
        "started_epoch": time.time(), "attempts": [], "production_stopped": False}

    def save(phase: str) -> None:
        state.update(phase=phase, updated_epoch=time.time())
        temporary = state_path.with_suffix(".incomplete")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(state_path)

    active = None
    try:
        deadline = time.monotonic() + 1800
        for attempt in range(1, 7):
            previous, quiet = None, 0
            while time.monotonic() < deadline:
                now = guard.activity()
                quiet = quiet + 1 if not now["busy"] and previous == now["counters"] else 0
                previous = now["counters"]
                save("waiting_for_idle")
                if quiet >= 3:
                    available = int(guard.remote(["python3", "-S", "-c",
                        ("from pathlib import Path; print(next(int(x.split()[1])*1024 "
                        "for x in Path('/proc/meminfo').read_text().splitlines() "
                        "if x.startswith('MemAvailable:'))) ")]).strip())
                    if available >= required_available:
                        break
                time.sleep(5)
            else:
                raise TimeoutError("No component idle window within thirty minutes")
            output = root / f"{args.revision}-attempt{attempt}"
            guard.remote(["mkdir", "-p", str(output)])
            active = f"mxwave-glm53-diagnostic-{args.revision}-a{attempt}"
            source = ROOT / "diagnostics-v1" / f"src-{args.revision}"
            script_probe = "bf16_kda" if args.probe == "bf16_kda_small" else args.probe
            if args.probe == "bf16_vocab":
                script_probe = "vocab"
            command = ["docker", "run", "-d", "--name", active, "--gpus", "all",
                "--cpus", "2", "--memory", f"{host_limit_mib}m",
                "--memory-swap", f"{host_limit_mib}m",
                "--pids-limit", "256", "--network", "none", "--user", f"{os.getuid()}:{os.getgid()}",
                "-v", f"{ROOT}:/audit:ro", "-v", f"{output}:/result", "-e", "OMP_NUM_THREADS=2",
                "-e", "VLLM_CACHE_ROOT=/result/cache/vllm", "-e", "TRITON_CACHE_DIR=/result/cache/triton",
                "-e", "TILELANG_CACHE_DIR=/result/cache/tilelang", "-e", "HF_HOME=/result/cache/hf",
                "-e", f"PYTHONPATH=/audit/diagnostics-v1/{source.name}",
                "--entrypoint", "/usr/bin/python3", IMAGE, "-u",
                f"/audit/diagnostics-v1/{source.name}/diagnose_{script_probe}.py",
                "--source", "/audit/source", "--report", "/result/report.json"]
            if args.probe == "bf16_kda_small":
                command.extend(["--fused-max-tokens", "8"])
            if args.probe == "bf16_vocab":
                command.extend(["--artifacts", "/audit/vocab-v2"])
            if args.probe.startswith("bf16_"):
                extras = ["-v", f"{OLD}:/experiment:ro", "-e",
                    (f"PYTHONPATH=/audit/diagnostics-v1/{source.name}:"
                     "/experiment/triple-v2/src/integrations/zipserv"), "-e",
                    ("MXWAVE_ZIPSERV_LIBRARY=/audit/native/libmxwave_kda_zipserv.so"
                     if args.probe == "bf16_cache" else
                     "MXWAVE_ZIPSERV_LIBRARY=/experiment/n128-w4/libmxwave_zipserv.so")]
                command[command.index("--entrypoint"):command.index("--entrypoint")] = extras
            if args.probe == "bf16_cache":
                command[command.index("--entrypoint") + 1] = "/usr/bin/timeout"
                position = command.index(IMAGE) + 1
                command[position:position] = ["--signal=TERM", "--kill-after=10", "180",
                                              "/usr/bin/python3"]
            before = guard.activity()
            if before["busy"]:
                active = None
                continue
            guard.remote(command)
            entry: dict[str, Any] = {"attempt": attempt, "container": active,
                "before": before, "remote_output": str(output)}
            state["attempts"].append(entry)
            save("running")
            job_deadline = time.monotonic() + 300
            while time.monotonic() < job_deadline:
                now = guard.activity()
                if now["busy"] or now["counters"] != before["counters"]:
                    guard.remote(["docker", "stop", "--timeout", "5", active])
                    entry.update(interrupted=True, after=now)
                    active = None
                    save("interrupted")
                    break
                info = json.loads(guard.remote(["docker", "inspect", "--format",
                                               "{{json .State}}", active]))
                if not info["Running"]:
                    entry.update(exit_code=info["ExitCode"], oom=info["OOMKilled"], after=now)
                    local = root / output.name
                    local.mkdir(exist_ok=True)
                    capture = ("import subprocess,sys; r=subprocess.run(['docker','logs',sys.argv[1]],"
                               "stdout=subprocess.PIPE,stderr=subprocess.STDOUT); "
                               "sys.stdout.buffer.write(r.stdout)")
                    logs = guard.remote(["python3", "-S", "-c", capture, active])
                    (local / "stdout.log").write_text(logs)
                    read_report = ("from pathlib import Path; import sys; p=Path(sys.argv[1]); "
                                   "print(p.read_text() if p.is_file() else '{}')")
                    entry["report"] = json.loads(guard.remote([
                        "python3", "-S", "-c", read_report, str(output / "report.json")]))
                    (local / "report.json").write_text(json.dumps(entry["report"], indent=2) + "\n")
                    active = None
                    if info["ExitCode"]:
                        raise RuntimeError("Component failed; retain its report and do not widen gates")
                    save("complete_idle")
                    return
                time.sleep(2)
            else:
                raise TimeoutError("Component exceeded five-minute bound")
        raise RuntimeError("GLM interrupted all component attempts")
    except BaseException as exc:
        state["error"] = repr(exc)
        save("failed")
        raise
    finally:
        if active:
            guard.remote(["docker", "stop", "--timeout", "5", active])
        lock.close()


if __name__ == "__main__":
    main()
