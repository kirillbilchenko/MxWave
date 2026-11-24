"""Detached, bounded two-Spark communication test that stops only its own idle-window jobs."""

from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import full_model_control as control
from benchmark_roce_pinning import review_reports


def memory_available(rank: int) -> int:
    """Read available unified memory without starting a GPU context."""
    script = ("from pathlib import Path; print(next(int(x.split()[1])*1024 "
              "for x in Path('/proc/meminfo').read_text().splitlines() "
              "if x.startswith('MemAvailable:')))")
    return int(control.run(rank, ["python3", "-S", "-c", script]).strip())


def main() -> None:
    """Wait for quiet counters, launch two capped processes and preserve interrupted attempts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--port", type=int, default=29783)
    parser.add_argument("--pause-glm", action="store_true")
    parser.add_argument("--watch", type=int)
    args = parser.parse_args()
    if not args.revision.isalnum() or not 20000 <= args.port <= 60000:
        raise ValueError("Require an immutable alphanumeric revision and a private test port")
    root = control.ROOT / f"transport-{args.revision}"
    path = control.ROOT / f"transport-controller-{args.revision}.json"
    if args.watch is not None:
        control.watch(args.revision, args.watch, path)
        return
    if path.exists():
        raise FileExistsError("Preserve prior communication evidence")
    lock = (control.ROOT / "diagnostics-v1/component.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    spec = importlib.util.spec_from_file_location("quiet_guard", control.OLD /
        "triple-v2/src/integrations/marlin_scales/quiet_window.py")
    assert spec and spec.loader
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    state: dict[str, Any] = {"phase": "preparing", "started_epoch": time.time(),
        "production_stopped": False, "service_paused": False, "attempts": [],
        "model_weights_loaded": False, "pause_authorized": args.pause_glm}
    active: list[tuple[int, str]] = []
    success = False

    def save(phase: str) -> None:
        state.update(phase=phase, updated_epoch=time.time())
        control.write(path, state)

    def stop_ours() -> None:
        for rank, name in active:
            listing = control.run(rank, ["docker", "ps", "-q", "--filter", f"name=^/{name}$"])
            if listing.strip():
                control.run(rank, ["docker", "stop", "--timeout", "5", name], timeout=40)
        active.clear()

    def terminated(signum: int, frame: Any) -> None:
        raise SystemExit(f"Signal {signum}")

    signal.signal(signal.SIGTERM, terminated)
    signal.signal(signal.SIGINT, terminated)
    try:
        save("preparing")
        originals = [control.inspect(rank, control.ORIGINAL_IDS[rank]) for rank in (0, 1)]
        if any(o["Image"] != control.IMAGE or not o["State"]["Running"] for o in originals):
            raise ValueError("Restore both original GLM ranks before this component")
        state["originals"] = [control.identity(o) for o in originals]
        hashes = [control.run(rank, ["sha256sum", str(root / "src/benchmark_roce_pinning.py")]
                              ).split()[0] for rank in (0, 1)]
        if len(set(hashes)) != 1:
            raise ValueError("Fixture source differs between hosts")
        state["source_sha256"] = hashes[0]
        deadline = time.monotonic() + 1800
        for attempt in range(1, 4):
            previous, quiet = None, 0
            while time.monotonic() < deadline:
                now = guard.activity()
                quiet = quiet + 1 if not now["busy"] and now["counters"] == previous else 0
                previous = now["counters"]
                save("waiting_for_idle")
                if quiet >= 3:
                    available = [memory_available(rank) for rank in (0, 1)]
                    state["available_memory_bytes"] = available
                    if args.pause_glm or min(available) >= 5.5 * 1024**3:
                        break
                    state["wait_reason"] = "Need 5.5 GiB free unified memory on each host"
                time.sleep(5)
            else:
                raise TimeoutError("No suitable idle window within thirty minutes")
            before = guard.activity()
            if before["busy"] or before["counters"] != previous:
                continue
            if args.pause_glm:
                watchdog = subprocess.Popen(["python3", "-u", str(Path(__file__).resolve()),
                    "--revision", args.revision, "--watch", str(os.getpid())],
                    stdout=(root / "watchdog.log").open("xb"), stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, start_new_session=True)
                state["watchdog_pid"] = watchdog.pid
                for _ in range(30):
                    ack = control.ROOT / f"watchdog-{args.revision}.json"
                    if ack.exists() and json.loads(ack.read_text()).get("ready"):
                        break
                    time.sleep(1)
                else:
                    raise RuntimeError("Independent restoration watchdog did not arm")
                state.update(service_paused=True, production_stopped=True,
                             restore_deadline_epoch=time.time() + 900)
                save("pausing_glm")
                for rank in (0, 1):
                    control.run(rank, ["docker", "stop", "--timeout", "30",
                                      control.ORIGINAL_IDS[rank]], timeout=90)
                for _ in range(30):
                    available = [memory_available(rank) for rank in (0, 1)]
                    if min(available) >= 30 * 1024**3:
                        break
                    time.sleep(2)
                else:
                    raise RuntimeError("Serving contexts did not release memory after stopping")
                state["available_memory_after_pause"] = available
            entry: dict[str, Any] = {"attempt": attempt, "before": before, "jobs": []}
            state["attempts"].append(entry)
            for rank in (1, 0):
                output = root / f"attempt{attempt}/rank{rank}"
                control.run(rank, ["mkdir", "-p", str(output)])
                name = (f"mxwave-glm53-full-{args.revision}-raw-r{rank}" if args.pause_glm else
                        f"mxwave-glm53-roce-{args.revision}-a{attempt}-r{rank}")
                active.append((rank, name))
                environment = {"PYTHONPATH": "/overlay/overlay", "GLM_ROCE_ALLREDUCE": "1",
                    "GLM_ROCE_PROXY_CPUS": "auto", "GLM_ROCE_PROXY_ISOLATE": "0",
                    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                    "GLOO_SOCKET_IFNAME": "enp1s0f0np0",
                    "B12X_ROCE_HCA": "rocep1s0f0,roceP2p1s0f0",
                    "B12X_ROCE_SPIN_LIMIT": "20000000", "XDG_CACHE_HOME": "/result/cache",
                    "B12X_COMPILE_CACHE_DIR": "/result/cache/b12x"}
                argv = ["docker", "run", "-d", "--name", name, "--gpus", "all",
                    "--network", "host", "--device", "/dev/infiniband", "--cap-add", "IPC_LOCK",
                    "--ulimit", "memlock=-1", "--memory", "6144m" if args.pause_glm else "3584m",
                    "--memory-swap", "6144m" if args.pause_glm else "3584m",
                    "--pids-limit", "256", "--user", f"{os.getuid()}:{os.getgid()}",
                    "-v", f"{root / 'src'}:/work:ro", "-v", f"{output}:/result",
                    "-v", "/home/kirya/glm53-runtime-20261010-long:/overlay:ro"]
                for key, value in environment.items():
                    argv.extend(["-e", key + "=" + value])
                argv.extend(["--entrypoint", "/usr/bin/timeout", control.IMAGE,
                    "--signal=TERM", "--kill-after=10", "300", "/usr/bin/python3", "-u",
                    "/work/benchmark_roce_pinning.py", "--rank", str(rank),
                    "--port", str(args.port), "--report", "/result/report.json"])
                control.run(rank, argv)
                entry["jobs"].append({"rank": rank, "container": name, "output": str(output),
                    "environment": environment})
                save("starting")
            job_deadline = time.monotonic() + 320
            while time.monotonic() < job_deadline:
                now = ({"production_paused": True, "epoch": time.time()} if args.pause_glm
                       else guard.activity())
                if not args.pause_glm and (now["busy"] or now["counters"] != before["counters"]):
                    entry.update(interrupted=True, after=now)
                    stop_ours()
                    save("interrupted")
                    break
                states = [json.loads(control.run(rank, ["docker", "inspect", "--format",
                                  "{{json .State}}", name])) for rank, name in active]
                if any(not s["Running"] and s["ExitCode"] for s in states):
                    entry["docker_states"] = states
                    raise RuntimeError("One fixture rank failed; retain logs and stop its peer")
                if all(not s["Running"] for s in states):
                    reports = []
                    for job in entry["jobs"]:
                        rank, name = job["rank"], job["container"]
                        local = root / f"attempt{attempt}/rank{rank}"
                        local.mkdir(parents=True, exist_ok=True)
                        capture = ("import subprocess,sys; r=subprocess.run(['docker','logs',"
                            "sys.argv[1]],stdout=subprocess.PIPE,stderr=subprocess.STDOUT); "
                            "sys.stdout.buffer.write(r.stdout)")
                        logs = control.run(rank, ["python3", "-S", "-c", capture, name])
                        (local / "stdout.log").write_text(logs)
                        report = control.remote_json(rank, Path(job["output"]) / "report.json")
                        if report["source_sha256"] != hashes[0]:
                            raise RuntimeError("Fixture loaded different source bytes")
                        control.write(local / "report.json", report)
                        reports.append(report)
                    review = review_reports(reports)
                    control.write(root / "review.json", review)
                    state["review"] = review
                    state["health"] = None if args.pause_glm else (
                        200 if control.health(8000) else None)
                    state["identities_unchanged"] = all(control.identity(
                        control.inspect(rank, control.ORIGINAL_IDS[rank])) == state["originals"][rank]
                        for rank in (0, 1))
                    if ((not args.pause_glm and state["health"] != 200)
                            or not state["identities_unchanged"]):
                        raise RuntimeError("Original service health or identity check failed")
                    entry.update(after=now, docker_states=states, accepted=True)
                    active.clear()
                    success = True
                    save("measured_pending_restoration" if args.pause_glm else "complete_idle")
                    return
                time.sleep(2)
            else:
                raise TimeoutError("Fixture exceeded its bounded runtime")
        raise RuntimeError("All three idle windows were interrupted")
    except BaseException as exc:
        state["error"] = repr(exc)
        save("failed_pending_restoration" if state["service_paused"] else "preparation_failed")
        raise
    finally:
        try:
            stop_ours()
        finally:
            if state["service_paused"]:
                save("restoring")
                state["restoration"] = control.restore(args.revision, path)
                state["service_paused"] = False
                state["health"] = state["restoration"]["health"]
                save("complete" if success else "failed_restored")
            lock.close()


if __name__ == "__main__":
    main()
