"""Head-side, service-preserving qualification of the full-model adapter's MoE path."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Any

ROOT = Path("/home/kirya/local-spark/experiments/glm53-mxfp4-audit-2026-10-15")
OLD = "/home/kirya/local-spark/experiments/zipserv-vllm031-2026-10-14"
IMAGE = "sha256:39a03677234ea74e35544d301bd14716c1ae9695ad8d91b16e2c76545e26e8d2"


def main() -> None:
    """Qualify raw and aligned adapters sequentially; abort only our job on GLM activity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="v1")
    parser.add_argument("--dispatch-comparison", action="store_true")
    args = parser.parse_args()
    if not args.revision.isalnum():
        raise ValueError("Revision must contain letters and digits only")
    revision = args.revision
    spec = importlib.util.spec_from_file_location(
        "guard", OLD + "/triple-v2/src/integrations/marlin_scales/quiet_window.py")
    assert spec is not None and spec.loader is not None
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    path = ROOT / f"full-model-v1/plugin-fixture-controller-{revision}.json"
    if path.exists():
        raise FileExistsError("Do not overwrite fixture evidence")
    state: dict[str, Any] = {"phase": "waiting_for_idle", "runs": [],
                             "dispatch_comparison": args.dispatch_comparison}

    def save(phase: str) -> None:
        state.update(phase=phase, updated_epoch=time.time())
        temporary = path.with_suffix(".incomplete")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(path)

    active: str | None = None
    try:
        for mode in ("raw", "aligned"):
            deadline = time.monotonic() + 1800
            for attempt in range(1, 7):
                previous, quiet = None, 0
                while time.monotonic() < deadline:
                    now = guard.activity()
                    quiet = quiet + 1 if (not now["busy"] and previous == now["counters"]) else 0
                    previous = now["counters"]
                    save("waiting_for_idle")
                    if quiet >= 3:
                        available = int(guard.remote(["python3", "-c",
                            ("from pathlib import Path; print(next(int(x.split()[1])*1024 "
                            "for x in Path('/proc/meminfo').read_text().splitlines() "
                            "if x.startswith('MemAvailable:')))")]).strip())
                        if available >= 4.5 * 1024**3:
                            break
                    time.sleep(5)
                else:
                    raise TimeoutError("No idle window for the adapter fixture")
                relative = f"full-model-v1/plugin-fixture-{mode}-{revision}-attempt{attempt}"
                guard.remote(["mkdir", "-p", str(ROOT / relative)])
                active = f"mxwave-glm53-plugin-{mode}-{revision}-{attempt}"
                command = [
                    "docker", "run", "-d", "--name", active, "--gpus", "all",
                    "--cpus", "2", "--memory", "2560m", "--memory-swap", "2560m",
                    "--pids-limit", "256", "--network", "none", "--user",
                    f"{os.getuid()}:{os.getgid()}", "-v", f"{ROOT}:/audit:ro", "-v",
                    f"{ROOT / relative}:/result", "-v", f"{OLD}:/experiment:ro",
                    "-e", "OMP_NUM_THREADS=1", "-e",
                    f"PYTHONPATH=/audit/full-model-v1/plugin-fixture-{revision}-src",
                    "-e", "VLLM_CACHE_ROOT=/tmp/vllm", "-e", "TRITON_CACHE_DIR=/tmp/triton",
                    "-e", "HF_HOME=/tmp/hf", "-e", ("MXWAVE_MOE_RAW_LIBRARY=/audit/"
                    "full-model-v1/native/libmxwave_moe_raw.so"), "-e",
                    ("MXWAVE_MOE_ALIGNED_LIBRARY=/audit/full-model-v1/native/"
                    "libmxwave_moe_aligned.so"), "-e", ("MXWAVE_MOE_RECORD_CODEC=/experiment/"
                    "triple-v2/src/integrations/marlin_scales/records.py"),
                    "--entrypoint", "/usr/bin/python3", IMAGE, "-u",
                    f"/audit/full-model-v1/plugin-fixture-{revision}-src/benchmark_model_plugin.py",
                    "--fixture",
                    "/audit/grouped-moe-v1/fixture", "--mode", mode,
                    "--report", "/result/report.json",
                ]
                if args.dispatch_comparison:
                    command.append("--dispatch-comparison")
                before = guard.activity()
                if before["busy"]:
                    active = None
                    continue
                guard.remote(command)
                entry: dict[str, Any] = {"mode": mode, "attempt": attempt, "container": active,
                                         "before": before, "results": str(ROOT / relative)}
                state["runs"].append(entry)
                save("running")
                job_deadline = time.monotonic() + (300 if args.dispatch_comparison else 180)
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
                        active = None
                        if info["ExitCode"]:
                            raise RuntimeError(f"Adapter fixture failed: {mode}")
                        entry["accepted"] = True
                        save("mode_complete")
                        break
                    time.sleep(2)
                else:
                    raise TimeoutError("Adapter fixture exceeded its component limit")
                if entry.get("accepted"):
                    break
            else:
                raise RuntimeError("GLM interrupted all adapter-fixture attempts")
        save("complete_idle")
    except Exception as exc:
        state["error"] = repr(exc)
        save("failed")
        raise
    finally:
        if active:
            guard.remote(["docker", "stop", "--timeout", "5", active])


if __name__ == "__main__":
    main()
