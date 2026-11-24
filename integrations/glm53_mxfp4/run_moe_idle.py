"""Head-side controller: run the bounded MoE component only beside idle GLM."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Any

IMAGE = "sha256:39a03677234ea74e35544d301bd14716c1ae9695ad8d91b16e2c76545e26e8d2"
AUDIT_ROOT = "/home/kirya/local-spark/experiments/glm53-mxfp4-audit-2026-10-15"
OLD_ROOT = "/home/kirya/local-spark/experiments/zipserv-vllm031-2026-10-14"
NSIGHT_ROOT = "/opt/nvidia/nsight-compute/2025.3.1"
NSIGHT_METRICS = (
    "smsp__inst_executed.sum",
    "smsp__sass_thread_inst_executed_op_integer_pred_on.sum",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct",
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "dram__bytes_read.sum", "gpu__time_duration.sum",
)


def main() -> None:
    """Use the established metrics guard; stop only this controller's experiment container."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guard", type=Path, required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--resume-from")
    parser.add_argument("--full-layer", action="store_true")
    parser.add_argument("--profile-concentrated", action="store_true")
    parser.add_argument("--focus-regression", action="store_true")
    parser.add_argument("--candidate-library", type=Path)
    parser.add_argument("--prefill-moe-block", type=int, choices=(16, 32, 48, 64))
    parser.add_argument("--nsight-arm", choices=("raw", "candidate"))
    parser.add_argument("--nsight-projection", choices=("gate-up", "down"), default="gate-up")
    args = parser.parse_args()
    if args.profile_concentrated and (
        not args.full_layer or args.diagnostic or args.resume_from is not None
        or args.focus_regression
    ):
        parser.error("Concentrated profiling requires --full-layer without diagnostic/resume")
    if args.focus_regression and (
        not args.full_layer or args.diagnostic or args.resume_from is not None
    ):
        parser.error("Focused optimization requires --full-layer without diagnostic/resume")
    if args.candidate_library is not None and (
        args.candidate_library.is_absolute() or ".." in args.candidate_library.parts
    ):
        parser.error("Candidate library must be relative to the audit directory")
    if args.prefill_moe_block is not None and (
        not args.full_layer or args.diagnostic or args.resume_from is not None
    ):
        parser.error("Prefill tile tuning requires --full-layer without diagnostic/resume")
    if args.nsight_arm is not None and (
        not args.full_layer or args.candidate_library is None or args.diagnostic
        or args.focus_regression or args.profile_concentrated or args.resume_from is not None
        or args.prefill_moe_block is not None
    ):
        parser.error("Nsight requires --full-layer/--candidate-library without other test modes")
    if not args.namespace.replace("-", "").isalnum():
        raise ValueError("Namespace must contain only letters, digits and hyphens")
    spec = importlib.util.spec_from_file_location("mxwave_moe_idle_guard", args.guard)
    if spec is None or spec.loader is None:
        raise ValueError("Missing established GLM idle guard")
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    state_path = Path(AUDIT_ROOT) / "grouped-moe-v1" / (args.namespace + "-controller.json")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    if state_path.exists():
        raise FileExistsError("Choose a new namespace rather than overwriting evidence")
    state: dict[str, Any] = {"phase": "waiting_for_idle", "runs": []}

    def save(phase: str) -> None:
        state.update(phase=phase, updated_epoch=time.time())
        temporary = state_path.with_suffix(".incomplete")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(state_path)

    active = None
    deadline = time.monotonic() + args.timeout
    try:
        if args.full_layer:
            while time.monotonic() < deadline:
                code = (
                    "import json; from pathlib import Path; "
                    f"p=Path({AUDIT_ROOT!r})/'grouped-moe-full-layer-v1/fixture/state.json'; "
                    "print(p.read_text() if p.exists() else json.dumps({'status':'not_started'}))"
                )
                fixture_state = json.loads(guard.remote(["python3", "-c", code]))
                state["fixture_state"] = fixture_state
                if fixture_state.get("status") == "complete":
                    if fixture_state["tensors"] != 1728:
                        raise RuntimeError("Full-layer fixture has incomplete expert coverage")
                    break
                save("waiting_for_full_fixture")
                time.sleep(10)
            else:
                raise TimeoutError("Full-layer fixture did not finish before controller deadline")
        for attempt in range(1, 7):
            quiet, previous = 0, None
            while time.monotonic() < deadline:
                now = guard.activity()
                quiet = quiet + 1 if (not now["busy"] and previous is not None
                                      and now["counters"] == previous["counters"]) else 0
                previous = now
                state["last_activity"] = now
                save("waiting_for_idle")
                if quiet >= 3:
                    available = int(guard.remote(["python3", "-c", (
                        "from pathlib import Path; print(next(x.split()[1] for x in "
                        "Path('/proc/meminfo').read_text().splitlines() "
                        "if x.startswith('MemAvailable:')))")]).strip()) * 1024
                    minimum = int((5.5 if args.full_layer else 4) * 1024**3)
                    state["available_bytes"] = available
                    if available >= minimum:
                        break
                    save("waiting_for_memory")
                time.sleep(5)
            else:
                raise TimeoutError("No stable idle GLM window before controller deadline")
            relative = f"grouped-moe-v1/{args.namespace}-attempt{attempt}"
            directory = AUDIT_ROOT + "/" + relative
            guard.remote(["mkdir", "-p", directory])
            name = f"mxwave-glm53-moe-{args.namespace}-{attempt}"
            command = ["/experiment/venv/bin/python", "-u", "/audit/benchmark_moe.py",
                       "--fixture", "/audit/grouped-moe-v1/fixture", "--libraries",
                       "/audit/grouped-moe-v1/native/build", "--codec-root",
                       "/experiment/triple-v2/src", "--report", "/result/report.json",
                       "--traces", "/result/traces"]
            if args.diagnostic:
                command.append("--diagnostic")
            if args.full_layer:
                command[command.index("--fixture") + 1] = "/audit/grouped-moe-full-layer-v1/fixture"
                command.append("--full-layer")
            if args.profile_concentrated:
                command.append("--profile-concentrated")
            if args.focus_regression:
                command.append("--focus-regression")
            if args.candidate_library is not None:
                command.extend(["--candidate-library", "/audit/" + str(args.candidate_library)])
            if args.prefill_moe_block is not None:
                command.extend(["--prefill-moe-block", str(args.prefill_moe_block)])
            if args.resume_from:
                relative_resume = Path(args.resume_from)
                if relative_resume.is_absolute() or ".." in relative_resume.parts:
                    raise ValueError("Resume evidence must be relative to the audit directory")
                command.extend(["--resume-from", "/audit/" + str(relative_resume)])
            if args.nsight_arm is not None:
                command = [NSIGHT_ROOT + "/ncu", "--replay-mode", "application",
                           "--profile-from-start", "off", "--clock-control", "none",
                           "--cache-control", "none", "--launch-count", "1", "--metrics",
                           ",".join(NSIGHT_METRICS), "--export", "/result/kernel",
                           "/experiment/venv/bin/python", "-u", "/audit/profile_moe_kernel.py",
                           "--fixture", "/audit/grouped-moe-full-layer-v1/fixture",
                           "--libraries", "/audit/grouped-moe-v1/native/build",
                           "--candidate-library", "/audit/" + str(args.candidate_library),
                           "--codec-root", "/experiment/triple-v2/src", "--arm", args.nsight_arm,
                           "--projection", args.nsight_projection,
                           "--report", "/result/application.json"]
            wrapper = (
                "import os,subprocess; subprocess.run(['uv','pip','uninstall','--python',"
                "'/usr/bin/python3','--break-system-packages','flashinfer-cubin',"
                "'flashinfer-jit-cache'],check=False); "
                f"os.setgid({os.getgid()}); os.setuid({os.getuid()}); os.umask(0o077); "
                f"subprocess.run({command!r},check=True)"
            )
            if args.nsight_arm is not None:
                # The driver requires CAP_SYS_ADMIN for performance counters.
                # Keep the profiler isolated and restore result ownership on exit.
                wrapper = (
                    "import os,subprocess; subprocess.run(['uv','pip','uninstall','--python',"
                    "'/usr/bin/python3','--break-system-packages','flashinfer-cubin',"
                    "'flashinfer-jit-cache'],check=False); os.umask(0o077); "
                    f"result=subprocess.run({command!r}); "
                    "[(os.chown(os.path.join(p,n)," + str(os.getuid()) + "," + str(os.getgid())
                    + ",follow_symlinks=False)) for p,ds,fs in os.walk('/result') "
                    "for n in ds+fs]; raise SystemExit(result.returncode)"
                )
            budget = "3g" if args.full_layer else "2g"
            argv = ["docker", "run", "-d", "--name", name, "--gpus", "all",
                    "--network", "none", "--memory", budget, "--memory-swap", budget,
                    "--cpus", "2", "--pids-limit", "512",
                    "-v", AUDIT_ROOT + ":/audit:ro", "-v", OLD_ROOT + ":/experiment:ro",
                    "-v", directory + ":/result:rw"]
            if args.nsight_arm is not None:
                argv.extend(["--cap-add", "SYS_ADMIN", "-v", NSIGHT_ROOT + ":" + NSIGHT_ROOT + ":ro"])
            for key, value in {
                "PYTHONPATH": "/audit", "PYTHONDONTWRITEBYTECODE": "1",
                "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "CUDA_CACHE_PATH": "/result/cache/cuda", "TRITON_CACHE_DIR": "/result/cache/triton",
                "XDG_CACHE_HOME": "/result/cache/xdg", "VLLM_CACHE_ROOT": "/result/cache/vllm",
                "TORCHINDUCTOR_CACHE_DIR": "/result/cache/inductor",
                "HF_HOME": "/result/cache/hf",
            }.items():
                argv.extend(["-e", key + "=" + value])
            argv.extend(["--entrypoint", "/usr/bin/python3", IMAGE, "-c", wrapper])
            baseline = guard.activity()
            if baseline["busy"] or baseline["counters"] != previous["counters"]:
                continue
            record = {"container": name, "before": baseline, "available_bytes": available,
                      "results": directory}
            state["runs"].append(record)
            guard.remote(argv)
            active = name
            save("running")
            end = min(deadline, time.monotonic() + 900)
            while time.monotonic() < end:
                now = guard.activity()
                record["after"] = now
                if now["busy"] or now["counters"] != baseline["counters"]:
                    guard.remote(["docker", "stop", "--timeout", "5", name])
                    active = None
                    record["interrupted"] = True
                    break
                info = json.loads(guard.remote(["docker", "inspect", "--format",
                                                "{{json .State}}", name]))
                if not info["Running"]:
                    active = None
                    record["exit_code"] = info["ExitCode"]
                    if info["ExitCode"]:
                        raise RuntimeError(f"Grouped MoE component failed: exit {info['ExitCode']}")
                    record["quiet"] = True
                    save("complete_idle")
                    return
                save("running")
                time.sleep(2)
            else:
                raise TimeoutError("Grouped component exceeded its guarded execution window")
        raise RuntimeError("GLM interrupted six attempts; no measurement accepted")
    except BaseException as error:
        state["error"] = repr(error)
        save("failed")
        raise
    finally:
        if active is not None:
            guard.remote(["docker", "stop", "--timeout", "5", active])


if __name__ == "__main__":
    main()
