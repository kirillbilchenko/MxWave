"""Diagnostic-only candidate load with Nsight on both ranks and independent GLM restoration."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import shlex
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import full_model_control as c

NSYS_HOME = Path("/opt/nvidia/nsight-systems/2025.3.2")
NSYS = NSYS_HOME / "target-linux-sbsa-armv8/nsys"
NEW_FILES = {"run_nsys_model.py", "measure_nsys_model.py", "nsys_worker.py"}


def validate_qualified_source(bound: dict[str, Any], qualified: dict[str, Any]) -> None:
    """Require the already-qualified execution snapshot; only orchestration may differ."""
    for key in ("source_metadata", "libraries", "vocabulary"):
        if bound[key] != qualified[key]:
            raise ValueError("Source checkpoint, native libraries or vocabulary changed")
    changes = {name for name, sha in qualified["code"].items() if bound["code"].get(name) != sha}
    additions = set(bound["code"]) - set(qualified["code"])
    if changes - {"full_model_control.py"} or additions - NEW_FILES:
        raise ValueError(f"Qualified execution changed: {sorted(changes | additions)}")


def launch(rank: int, revision: str, original: dict[str, Any], vocabulary: int,
           manifest_sha: str) -> tuple[list[str], dict[str, Any]]:
    """Wrap the unchanged candidate in Nsight; leave the saved comparison runner intact."""
    argv, meta = c.engine(rank, "kda", revision, original, vocabulary, attention_mapping=True)
    command = meta["command"]
    command[command.index("--worker-extension-cls") + 1] = "nsys_worker.NsysWorker"
    command[command.index("--compilation-config") + 1] = json.dumps({
        "mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1],
        "max_cudagraph_capture_size": 1})
    command[command.index("--profiler-config") + 1] = json.dumps({
        "profiler": "cuda", "ignore_frontend": True, "detailed_trace_annotation": True})
    # Nsight/CUPTI initialization consumes memory before vLLM's free-memory guard.
    # KV stays explicitly fixed at 4 GiB; this changes no cache sizing or model path.
    command[command.index("--gpu-memory-utilization") + 1] = "0.85"
    extra = {"MXWAVE_GLM53_KDA_FUSED_MAX_TOKENS": "8", "MXWAVE_MOE_MODE": "aligned",
        "MXWAVE_GLM53_VOCAB_MODE": "packed", "MXWAVE_MOE_DISPATCH": "dispatcher",
        "MXWAVE_GLM53_VOCAB_ARTIFACTS": "/audit/full-model-v1/vocab-v2",
        "MXWAVE_GLM53_VOCAB_MANIFEST_SHA256": manifest_sha,
        "PYTHONDONTWRITEBYTECODE": "1"}
    meta["environment"].update(extra)
    for key, value in extra.items():
        old = next((i for i, token in enumerate(argv) if token.startswith(key + "=")), None)
        if old is not None:
            argv[old] = key + "=" + value
        else:
            at = argv.index("--entrypoint")
            argv[at:at] = ["-e", key + "=" + value]
    at = argv.index("--entrypoint")
    argv[at:at] = ["-v", f"{NSYS_HOME}:{NSYS_HOME}:ro"]
    meta["mounts"].append((str(NSYS_HOME), str(NSYS_HOME), "ro"))
    profiler = ["profile", "--sample=none", "--cpuctxsw=none", "--trace=cuda,nvtx,osrt",
        "--capture-range=cudaProfilerApi", "--capture-range-end=stop", "--kill=sigterm",
        "--wait=all", "--python-sampling=true", "--python-sampling-frequency=500",
        "--osrt-threshold=10000", "--osrt-backtrace-threshold=80000", "--pytorch=none",
        "--force-overwrite=false", "-o", f"/traces/nsys-r{rank}"]
    at = argv.index("--entrypoint")
    code = f"/audit/full-model-v1/full-{revision}-src"
    argv[at:] = ["--entrypoint", str(NSYS), c.IMAGE, *profiler, "/usr/bin/python3", "-u",
                 code + "/engine_boot.py", *command]
    meta["nsight"] = {"binary": str(NSYS), "arguments": profiler,
        "limitation": "No perf sampling or precise scheduler tracing; OSRT plus Python sampling"}
    return argv, meta


def cpu_cli_check(argv_meta: tuple[list[str], dict[str, Any]]) -> str:
    """Check plugin discovery, extension imports and CUDA-profiler CLI without a GPU."""
    _, meta = argv_meta
    argv = ["docker", "run", "--rm", "--network", "none", "--cpus", "2", "--memory", "3g",
        "--memory-swap", "3g", "-e", "CUDA_VISIBLE_DEVICES=", "-e", "NVIDIA_VISIBLE_DEVICES=void",
        "-e", "MXWAVE_VALIDATE_ONLY=1", "-e", "UV_CACHE_DIR=/tmp/mxwave-uv-cache"]
    for source, target, access in meta["mounts"]:
        argv.extend(["-v", f"{source}:{target}:{access}"])
    for key, value in meta["environment"].items():
        argv.extend(["-e", key + "=" + value])
    check = ("from vllm import platforms; from vllm.platforms.cpu import CpuPlatform; "
             "platforms.current_platform=CpuPlatform(); "
             "import engine_boot; from nsys_worker import NsysWorker; "
             "from combined_worker import CombinedWorker; "
             "assert issubclass(NsysWorker,CombinedWorker); engine_boot.main()")
    argv.extend(["--entrypoint", "/usr/bin/python3", c.IMAGE, "-c", check, *meta["command"]])
    return c.run(0, argv, 240)


def collect_outputs(revision: str, launches: list[tuple[list[str], dict[str, Any]]]) -> dict[str, Any]:
    """Export and bind traces after production restoration, retaining raw reports privately."""
    outcome: dict[str, Any] = {"ranks": []}
    for rank in (0, 1):
        folder = c.ROOT / f"run-{revision}/kda"
        trace = folder / "traces" / f"nsys-r{rank}.nsys-rep"
        capture = ("import subprocess,sys; r=subprocess.run(['docker','logs',sys.argv[1]],"
                   "stdout=subprocess.PIPE,stderr=subprocess.STDOUT); sys.stdout.buffer.write(r.stdout)")
        (folder / f"rank{rank}-engine.log").write_text(c.run(
            rank, ["python3", "-c", capture, launches[rank][1]["name"]], 90))
        c.run(rank, [str(NSYS), "export", "--type=sqlite", "--output",
            str(trace.with_suffix(".sqlite")), str(trace)], 300)
        script = ("import pathlib,hashlib,json,sys; p=pathlib.Path(sys.argv[1]); "
                  "print(json.dumps([{'name':str(x),'bytes':x.stat().st_size,"
                  "'sha256':hashlib.sha256(x.read_bytes()).hexdigest()} "
                  "for x in (p,p.with_suffix('.sqlite'))]))")
        records = json.loads(c.run(rank, ["python3", "-c", script, str(trace)], 120))
        outcome["ranks"].append({"rank": rank, "artifacts": records})
    destination = c.ROOT / f"run-{revision}/kda/traces-worker"
    destination.mkdir(exist_ok=False)
    c.run(0, ["rsync", "-a", "--timeout=120", "--bwlimit=65536", "-e", shlex.join(c.SSH[:-1]),
        c.SSH[-1] + ":" + str(c.ROOT / f"run-{revision}/kda/traces") + "/",
        str(destination) + "/"], 600)
    return outcome


def control(revision: str) -> None:
    """Prepare before pause, wait for idle, arm watchdog, load once and always restore."""
    code = c.ROOT / f"full-{revision}-src"
    path = c.ROOT / f"nsys-controller-{revision}.json"
    if path.exists():
        raise FileExistsError("Preserve the immutable diagnostic revision")
    lock = (c.ROOT / "diagnostics-v1/component.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state: dict[str, Any] = {"phase": "preparing", "controller_pid": os.getpid(),
        "started_epoch": time.time(), "service_paused": False, "arms": [],
        "purpose": "Candidate-only Nsight critical-path diagnostic; no stock comparison"}

    def save(phase: str) -> None:
        state.update(phase=phase, updated_epoch=time.time())
        c.write(path, state)

    def terminate(signum: int, frame: Any) -> None:
        raise SystemExit(f"Signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    save("preparing")
    success = False
    launches: list[tuple[list[str], dict[str, Any]]] = []
    try:
        previous = c.remote_json(0, c.ROOT / "full-controller-combineddispatchv1.json")
        if previous["phase"] != "complete" or previous["service_paused"]:
            raise ValueError("The qualified combined run is not complete/restored")
        bound = [c.preflight(rank, code) for rank in (0, 1)]
        if bound[0] != bound[1]:
            raise ValueError("Diagnostic sources differ between Sparks")
        validate_qualified_source(bound[0], previous["preflight"])
        state["preflight"] = bound[0]
        manifest = c.ROOT / "vocab-v2/prepared.json"
        manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
        if manifest_sha != previous["vocabulary_fixture"]["manifest_sha256"]:
            raise ValueError("Vocabulary artifact manifest differs from the qualification")
        tools = []
        for rank in (0, 1):
            if c.run(rank, ["sha256sum", str(manifest)]).split()[0] != manifest_sha:
                raise ValueError("Vocabulary manifests differ between nodes")
            tools.append({"version": c.run(rank, [str(NSYS), "--version"]).strip(),
                "binary_sha256": c.run(rank, ["sha256sum", str(NSYS)]).split()[0]})
        if tools[0] != tools[1]:
            raise ValueError("Nsight tooling differs between nodes")
        state["tooling"] = tools
        originals = [c.inspect(rank, c.ORIGINAL_IDS[rank]) for rank in (0, 1)]
        if any(o["Image"] != c.IMAGE or not o["State"]["Running"] for o in originals):
            raise ValueError("Both original GLM services must be running before the idle pause")
        state["originals"] = [c.identity(o) for o in originals]
        launches = [launch(rank, revision, originals[rank], bound[0]["vocabulary"], manifest_sha)
                    for rank in (0, 1)]
        state["launches"] = [meta for _, meta in launches]
        source_protocol = c.ROOT / "run-combineddispatchv1/results/kda/protocol.json"
        frozen = json.loads(source_protocol.read_bytes())
        cases = [case for case in frozen["timing"] if
                 (case["prompt_tokens"], case["batch"]) == (256, 4)]
        if len(cases) != 1:
            raise ValueError("The saved B4 prompt protocol is ambiguous")
        results = c.ROOT / f"run-{revision}/results"
        results.mkdir(parents=True, exist_ok=False)
        c.write(results / "protocol.json", {"prompts": cases[0]["prompts"], "tokens": 64,
            "source_protocol_sha256": hashlib.sha256(source_protocol.read_bytes()).hexdigest(),
            "stages": [["before", 5], ["captured", 1], ["after", 5]], "warmups": 2})
        state["cpu_cli"] = cpu_cli_check(launches[0])
        spec = importlib.util.spec_from_file_location("guard", c.OLD /
            "triple-v2/src/integrations/marlin_scales/quiet_window.py")
        assert spec and spec.loader
        guard = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(guard)
        deadline, prior, quiet = time.monotonic() + 3600, None, 0
        save("waiting_for_idle")
        while time.monotonic() < deadline:
            now = guard.activity()
            quiet = quiet + 1 if not now["busy"] and prior == now["counters"] else 0
            prior = now["counters"]
            if quiet >= 4:
                state["idle_before_pause"] = now
                break
            time.sleep(5)
        else:
            raise TimeoutError("No idle window within one hour; production remains running")
        watchdog = subprocess.Popen(["python3", "-u", str(code / "run_nsys_model.py"),
            "--revision", revision, "--watch", str(os.getpid())],
            stdout=(c.ROOT / f"watchdog-{revision}.log").open("xb"), stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True)
        state["watchdog_pid"] = watchdog.pid
        for _ in range(30):
            ack = c.ROOT / f"watchdog-{revision}.json"
            if ack.exists() and json.loads(ack.read_bytes()).get("ready"):
                break
            time.sleep(.5)
        else:
            raise RuntimeError("Independent restoration watchdog did not arm")
        final = guard.activity()
        if final["busy"] or final["counters"] != prior:
            raise RuntimeError("GLM became active before the pause; production remains running")
        state.update(service_paused=True, restore_deadline_epoch=time.time() + 3600)
        save("pausing_service")
        for rank in (0, 1):
            c.run(rank, ["docker", "stop", "--timeout", "30", c.ORIGINAL_IDS[rank]], 90)
        save("loading_candidate")
        for rank in (1, 0):
            c.run(rank, launches[rank][0], 120)
        deadline = time.monotonic() + 1500
        while time.monotonic() < deadline:
            if c.health(18784):
                break
            for rank in (0, 1):
                if not c.inspect(rank, launches[rank][1]["name"])["State"]["Running"]:
                    raise RuntimeError(f"Candidate rank {rank} exited during startup")
            time.sleep(5)
        else:
            raise TimeoutError("Candidate startup exceeded twenty-five minutes")
        state["candidate_ready_epoch"] = time.time()
        name = f"mxwave-glm53-bench-{revision}-kda"
        client = ["docker", "run", "-d", "--name", name, "--network", "host", "--cpus", "2",
            "--memory", "1536m", "--memory-swap", "1536m", "--user", f"{os.getuid()}:{os.getgid()}",
            "-e", "CUDA_VISIBLE_DEVICES=", "-e", "NVIDIA_VISIBLE_DEVICES=void",
            "-v", f"{code}:/src:ro", "-v", f"{results}:/results", "--entrypoint", "/usr/bin/python3",
            c.IMAGE, "-u", "/src/measure_nsys_model.py", "--root", "/results"]
        c.run(0, client)
        save("measuring_candidate")
        deadline = time.monotonic() + 1200
        while time.monotonic() < deadline:
            info = c.inspect(0, name)["State"]
            if not info["Running"]:
                if info["ExitCode"]:
                    raise RuntimeError("Candidate correctness/Nsight measurement failed")
                report = json.loads((results / "nsys-measurement.json").read_bytes())
                if report["status"] != "complete":
                    raise RuntimeError("The diagnostic client did not finish its report")
                state["measurement_sha256"] = hashlib.sha256(
                    (results / "nsys-measurement.json").read_bytes()).hexdigest()
                break
            time.sleep(5)
        else:
            raise TimeoutError("Candidate measurement exceeded twenty minutes")
        success = True
    except BaseException as exc:
        state["error"] = repr(exc)
        save("failed_pending_restoration" if state["service_paused"] else "preparation_failed")
        raise
    finally:
        if state["service_paused"]:
            save("restoring")
            state["restoration"] = c.restore(revision, path)
            state["service_paused"] = False
            save("restored_analyzing" if success else "failed_restored")
        lock.close()
    if success:
        try:
            state["capture_artifacts"] = collect_outputs(revision, launches)
            save("complete")
        except BaseException as exc:
            state["analysis_error"] = repr(exc)
            save("failed_restored")
            raise


def main() -> None:
    """Run on Spark1, or act as the independently detached restoration watchdog."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--watch", type=int)
    args = parser.parse_args()
    if not args.revision.isalnum():
        raise ValueError("Use an alphanumeric immutable revision")
    if args.watch is not None:
        c.watch(args.revision, args.watch, c.ROOT / f"nsys-controller-{args.revision}.json")
    else:
        control(args.revision)


if __name__ == "__main__":
    main()
