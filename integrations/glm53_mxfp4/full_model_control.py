"""Detached TP2 pilot with a separate restoration watchdog and exact service identities.

Run only on Spark1 after the user has authorized pausing and restoring both GLM ranks.
The preparation wait uses no GPU. The live service is paused only at an idle window.
"""

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
import urllib.request
from pathlib import Path
from typing import Any

AUDIT = Path("/home/kirya/local-spark/experiments/glm53-mxfp4-audit-2026-10-15")
ROOT = AUDIT / "full-model-v1"
OLD = Path("/home/kirya/local-spark/experiments/zipserv-vllm031-2026-10-14")
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-i",
       "/home/kirya/.ssh/id_glm_cluster", "kirya@192.168.192.2"]
IMAGE = "sha256:39a03677234ea74e35544d301bd14716c1ae9695ad8d91b16e2c76545e26e8d2"
ORIGINAL_IDS = (
    "8e0426b9a4183b8fdaa2ed2d4047924e9554b2cb37950e8f31bce52589d07e11",
    "fb147a890d8bdd2b475588935a3e1d6d7c60828f894bc980bc7b6e7a40d00224",
)
LIBRARIES = {
    "libmxwave_kda_zipserv.so": "6d51630cd075a8c0dcbb957594341fb21e42d4b4634bb1c92735275d81a70f62",
    "libmxwave_moe_raw.so": "48d9dafefed0d259134f23e7f91a30de40443d0cbb493b337667e6ce9a9d69d2",
    "libmxwave_moe_aligned.so": "53ebb8478607104723786934c96ae7dc973a9d16ff93a2ec8c93a2a653dae658",
}
METADATA = {
    "config.json": "194d290ebb0f19aedc60d24ea9e236e33c86616e460f4a108a842f6fd5b8b198",
    "model.safetensors.index.json":
        "6d1364a27eef78675390422b1b5230601817fb7460ce816951b2cd515934c85d",
}
ENV_EXACT = {
    "LD_LIBRARY_PATH", "VLLM_HOST_IP", "VLLM_DISABLED_KERNELS", "VLLM_TEST_FORCE_FP8_MARLIN",
    "VLLM_ENABLE_CUDA_COMPATIBILITY", "FLASHINFER_CUDA_ARCH_LIST", "GLM_KPOOL_FIX",
    "GLM_MAMBA_ALIGN_FIX", "GLM_ROCE_ALLREDUCE", "GLM_KDA_CONV_SPLIT",
    "GLM_FLASHKDA_PREFILL", "GLM_TRITON_MLA_PREFILL", "VLLM_WORKER_MULTIPROC_METHOD",
    "GLM_FAST_LOAD", "GLM_FAST_LOAD_THREADS", "GLM_FAST_LOAD_SLAB_MB", "GLM_FAST_LOAD_AHEAD_MB",
    "GLM_FAST_LOAD_PINNED", "GLM_FAST_LOAD_DROP_CACHE", "GLM_FAST_LOAD_VERIFY",
    "GLM_FAST_LOAD_MAX_UNITS", "GLM_FAST_LOAD_ESCAPE_S",
    "GLM_ROCE_PROXY_CPUS", "GLM_ROCE_PROXY_ISOLATE", "GLM_ROCE_PROXY_BIG",
    "GLM_ROCE_MAX_SIZE", "GLM_ROCE_GATHER_MAX_SIZE", "GLM_ROCE_REQUIRE", "GLM_ROCE_B12X_PATH",
}
TRANSPORT_PREFIXES = ("GLM_ROCE_", "B12X_", "NCCL_", "UCX_", "GLOO_")


def validate_transport_control(
    previous: list[dict[str, Any]], proposed: list[dict[str, Any]],
) -> None:
    """Reject saved TP2 controls after changing communication or host-thread settings."""
    if len(previous) != 2 or len(proposed) != 2:
        raise ValueError("Require both ranks' saved transport settings before control reuse")
    for old, new in zip(previous, proposed, strict=True):
        environments = [arm.get("environment") for arm in (old, new)]
        if any(not isinstance(env, dict) for env in environments):
            raise ValueError("Missing saved transport settings; collect a matched control")
        selected = [{key: value for key, value in env.items()
                     if key.startswith(TRANSPORT_PREFIXES) or key == "OMP_NUM_THREADS"}
                    for env in environments]
        if selected[0] != selected[1]:
            raise ValueError("Saved control transport/CPU settings differ; collect a matched control")


def run(rank: int, command: list[str], timeout: float = 90) -> str:
    """Execute explicit argv locally or on the private Spark link."""
    argv = command if rank == 0 else [*SSH, shlex.join(command)]
    result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    if result.returncode:
        # Docker command arguments may contain public settings; never include
        # complete inspect data or a production environment in an error message.
        raise RuntimeError(f"rank={rank} command={command[:2]!r} exited={result.returncode}: "
                           f"{result.stderr[-3000:]}")
    return result.stdout


def remote_json(rank: int, path: Path) -> dict[str, Any]:
    """Read only this experiment's JSON state."""
    return json.loads(run(rank, ["cat", str(path)]))


def write(path: Path, value: Any) -> None:
    """Commit an operational state update atomically."""
    temporary = path.with_suffix(".incomplete")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def inspect(rank: int, container: str) -> dict[str, Any]:
    """Read a container privately; callers retain only identities and sanitized settings."""
    return json.loads(run(rank, ["docker", "inspect", container]))[0]


def signature(value: Any) -> str:
    """Bind settings by digest without retaining sensitive environment contents."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def identity(value: dict[str, Any]) -> dict[str, Any]:
    """Record enough identity to restore the existing containers without recreating them."""
    return {"id": value["Id"], "image": value["Image"],
            "config_sha256": signature(value["Config"]),
            "host_config_sha256": signature(value["HostConfig"]),
            # Docker constructs this list from a map and returns it in a
            # different order across inspect calls, even for an idle service.
            "mounts_sha256": signature(sorted(value["Mounts"],
                                               key=lambda mount: mount["Destination"]))}


def health(port: int) -> bool:
    """Check a loopback engine health endpoint with a short timeout."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as response:
            return response.status == 200
    except (OSError, TimeoutError):
        return False


def stop_owned(revision: str) -> None:
    """Stop only experiment containers with names derived from this exact revision."""
    for rank in (0, 1):
        for mode in ("stock", "raw", "aligned", "kda"):
            names = [f"mxwave-glm53-full-{revision}-{mode}-r{rank}"]
            if rank == 0:
                names.append(f"mxwave-glm53-bench-{revision}-{mode}")
            for name in names:
                listing = run(rank, ["docker", "ps", "-q", "--filter", f"name=^/{name}$"])
                if listing.strip():
                    run(rank, ["docker", "stop", "--timeout", "10", name], timeout=45)


def restore(revision: str, state_path: Path) -> dict[str, Any]:
    """Serialize watchdog/controller restoration and verify both original configurations."""
    result_path = ROOT / f"restoration-{revision}.json"
    with (ROOT / f"restore-{revision}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(state_path.read_bytes())
        result: dict[str, Any] = {"status": "restoring", "started_epoch": time.time()}
        write(result_path, result)
        try:
            stop_owned(revision)
            restored = []
            for rank in (1, 0):
                original = inspect(rank, ORIGINAL_IDS[rank])
                if identity(original) != state["originals"][rank]:
                    raise RuntimeError("Original GLM container configuration changed")
                if not original["State"]["Running"]:
                    run(rank, ["docker", "start", ORIGINAL_IDS[rank]], timeout=120)
                restored.append({"rank": rank, **identity(inspect(rank, ORIGINAL_IDS[rank]))})
            deadline = time.monotonic() + 1200
            while time.monotonic() < deadline:
                if health(8000):
                    break
                for rank in (0, 1):
                    if not inspect(rank, ORIGINAL_IDS[rank])["State"]["Running"]:
                        raise RuntimeError("An original GLM rank exited during restoration")
                time.sleep(5)
            else:
                raise TimeoutError("Original GLM health did not return within twenty minutes")
            result.update(status="complete", health=200, containers=restored,
                          finished_epoch=time.time())
            write(result_path, result)
            return result
        except BaseException as exc:
            result.update(status="failed", error=repr(exc), updated_epoch=time.time())
            write(result_path, result)
            raise


def watch(revision: str, parent_pid: int, state_path: Path) -> None:
    """Restore on a dead controller or expired outage deadline, independent of SSH/laptop."""
    write(ROOT / f"watchdog-{revision}.json", {"pid": os.getpid(), "parent": parent_pid,
                                             "ready": True, "epoch": time.time()})
    while True:
        state = json.loads(state_path.read_bytes())
        if state.get("phase") in ("complete", "failed_restored", "preparation_failed"):
            return
        try:
            os.kill(parent_pid, 0)
            alive = True
        except ProcessLookupError:
            alive = False
        expired = time.time() > state.get("restore_deadline_epoch", float("inf"))
        if state.get("service_paused") and (not alive or expired):
            # Expiration must also terminate the controller, otherwise it could
            # start a subsequent arm after the watchdog restores production.
            if alive:
                os.kill(parent_pid, signal.SIGTERM)
            for attempt in range(3):
                try:
                    restore(revision, state_path)
                    return
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(10)
        if not alive:
            return
        time.sleep(5)


def preflight(rank: int, code: Path) -> dict[str, Any]:
    """Bind native libraries, full source metadata, and the immutable integration snapshot."""
    script = """import pathlib,json,hashlib,sys
r=pathlib.Path(sys.argv[1]); c=pathlib.Path(sys.argv[2])
expected=json.loads(sys.argv[3]); libs=json.loads(sys.argv[4])
out={'source_metadata':{},'libraries':{},'code':{}}
for name,want in expected.items():
 got=hashlib.sha256((r/'source'/name).read_bytes()).hexdigest()
 assert got==want,(name,got); out['source_metadata'][name]=got
for name,want in libs.items():
 got=hashlib.sha256((r/'native'/name).read_bytes()).hexdigest()
 assert got==want,(name,got);out['libraries'][name]=got
for p in sorted(c.rglob('*')):
 if p.is_file() and p.suffix in ('.py','.toml'):
  out['code'][str(p.relative_to(c))]=hashlib.sha256(p.read_bytes()).hexdigest()
out['vocabulary']=json.loads((r/'source/config.json').read_bytes())['text_config']['vocab_size']
print(json.dumps(out))
"""
    return json.loads(run(rank, ["python3", "-c", script, str(ROOT), str(code),
        json.dumps(METADATA), json.dumps(LIBRARIES)], timeout=120))


def engine(rank: int, mode: str, revision: str, original: dict[str, Any],
           vocabulary: int, *, attention_mapping: bool = False) -> tuple[list[str], dict[str, Any]]:
    """Construct a private engine, reusing public transport and required architecture fixes."""
    code = f"/audit/full-model-v1/full-{revision}-src"
    local = ROOT / f"run-{revision}" / mode
    run(rank, ["mkdir", "-p", str(local / "cache"), str(local / "traces")])
    environment = {}
    for entry in original["Config"]["Env"]:
        key, value = entry.split("=", 1)
        if key in ENV_EXACT or key.startswith(("NCCL_", "UCX_", "GLOO_", "B12X_")):
            if any(term in key.upper() for term in ("SECRET", "PASSWORD", "CREDENTIAL", "API_KEY")):
                raise ValueError("Unexpected secret in public transport whitelist")
            environment[key] = value
    environment.update({"PYTHONPATH": code + ":/overlay/overlay", "OMP_NUM_THREADS": "4",
        "MXWAVE_MOE_MODE": "stock" if mode == "kda" else mode,
        "MXWAVE_MOE_RAW_LIBRARY": "/audit/full-model-v1/native/libmxwave_moe_raw.so",
        "MXWAVE_MOE_ALIGNED_LIBRARY": "/audit/full-model-v1/native/libmxwave_moe_aligned.so",
        "MXWAVE_MOE_RECORD_CODEC": "/experiment/triple-v2/src/integrations/marlin_scales/records.py",
        "VLLM_SERVER_DEV_MODE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": "/cache/hf", "VLLM_CACHE_ROOT": "/cache/vllm",
        "TRITON_CACHE_DIR": "/cache/triton", "TORCHINDUCTOR_CACHE_DIR": "/cache/inductor",
        "CUDA_CACHE_PATH": "/cache/cuda", "TILELANG_CACHE_DIR": "/cache/tilelang",
        "XDG_CACHE_HOME": "/cache", "VLLM_ENGINE_READY_TIMEOUT_S": "1200"})
    if attention_mapping:
        environment["MXWAVE_GLM53_ATTENTION_MAPPING"] = "1"
    if mode == "kda":
        if not attention_mapping:
            raise ValueError("Selective KDA requires the audited attention mapping")
        environment.update({
            "MXWAVE_GLM53_KDA_MODE": "packed", "MXWAVE_GLM53_KDA_SOURCE": "/model",
            "MXWAVE_ZIPSERV_LIBRARY": "/audit/full-model-v1/native/libmxwave_kda_zipserv.so"})
    environment["PYTHONPATH"] += ":/experiment/triple-v2/src/integrations/zipserv"
    mounts = []
    for mount in original["Mounts"]:
        target = mount["Destination"]
        if target == "/overlay" or target.startswith("/usr/local/lib/python3.12/dist-packages/vllm/"):
            if mount["RW"] or not mount["Source"].startswith("/home/kirya/glm53-runtime-"):
                raise ValueError("Unexpected mutable or external architecture overlay")
            mounts.append((mount["Source"], target, "ro"))
    mounts.extend([(str(ROOT / "source"), "/model", "ro"), (str(AUDIT), "/audit", "ro"),
                   (str(OLD), "/experiment", "ro"),
                   (str(local / "cache"), "/cache", "rw"),
                   (str(local / "traces"), "/traces", "rw")])
    command = ["serve", "/model", "--served-model-name", "mxwave-glm53-pilot", "--dtype",
        "bfloat16", "--tensor-parallel-size", "2", "--nnodes", "2", "--node-rank", str(rank),
        "--master-addr", "192.168.192.1", "--master-port", "29679",
        "--distributed-executor-backend", "mp", "--max-model-len", "8192",
        "--kv-cache-dtype", "fp8_e4m3", "--kv-cache-memory-bytes", str(4 * 1024**3),
        "--gpu-memory-utilization", "0.90", "--max-num-seqs", "8",
        "--max-num-batched-tokens", "2048", "--block-size", "2304",
        "--moe-backend", "marlin", "--linear-backend", "marlin",
        "--no-enable-prefix-caching", "--enable-chunked-prefill",
        "--no-enable-flashinfer-autotune", "--disable-custom-all-reduce",
        "--chat-template", "/model/chat_template.jinja", "--reasoning-parser", "glm45",
        "--default-chat-template-kwargs", '{"reasoning_effort":"low"}',
        "--generation-config", "vllm", "--max-logprobs", str(vocabulary),
        "--worker-extension-cls", "model_plugin.ScaleWorker",
        "--compilation-config", json.dumps({"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1, 2, 4, 8], "max_cudagraph_capture_size": 8}),
        "--limit-mm-per-prompt", '{"image":0,"video":0}',
        "--profiler-config", json.dumps({"profiler": "torch", "torch_profiler_dir": "/traces",
            "torch_profiler_with_stack": False, "torch_profiler_use_gzip": True,
            "detailed_trace_annotation": True}), "--host", "127.0.0.1", "--port", "18784"]
    if rank == 1:
        command.append("--headless")
    name = f"mxwave-glm53-full-{revision}-{mode}-r{rank}"
    argv = ["docker", "run", "-d", "--name", name, "--gpus", "all", "--network", "host",
            "--ipc", "host", "--memory", "112g", "--memory-swap", "112g", "--cap-add", "IPC_LOCK",
            "--security-opt", "label=disable", "--device", "/dev/infiniband",
            "--ulimit", "memlock=-1", "--ulimit", "stack=67108864", "--ulimit", "nofile=1048576"]
    for source, target, access in mounts:
        argv.extend(["-v", f"{source}:{target}:{access}"])
    for key, value in environment.items():
        argv.extend(["-e", f"{key}={value}"])
    argv.extend(["--entrypoint", "/usr/bin/python3", IMAGE, "-u", code + "/engine_boot.py", *command])
    return argv, {"name": name, "image": IMAGE, "command": command,
                  "environment": environment, "mounts": mounts}


def control(revision: str) -> None:
    """Wait for prepared sources, pause idle GLM, measure three arms, and restore on every exit."""
    code = ROOT / f"full-{revision}-src"
    state_path = ROOT / f"full-controller-{revision}.json"
    if state_path.exists():
        raise FileExistsError("Do not overwrite a full-model run")
    state: dict[str, Any] = {"phase": "preparing", "controller_pid": os.getpid(),
                             "started_epoch": time.time(), "service_paused": False, "arms": []}

    def save(phase: str) -> None:
        state.update(phase=phase, updated_epoch=time.time())
        write(state_path, state)

    def terminate(signum: int, frame: Any) -> None:
        raise SystemExit(f"Signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    save("preparing")
    success = False
    try:
        deadline = time.monotonic() + 14400
        while time.monotonic() < deadline:
            source = remote_json(0, ROOT / "head-source-state-v2.json")
            download = remote_json(1, ROOT / "download-state.json")
            scales = remote_json(1, ROOT / "full-scale-audit.json")
            fixture = remote_json(0, ROOT / "plugin-fixture-controller-v3.json")
            for item, key in ((source, "phase"), (download, "status"), (scales, "status"),
                              (fixture, "phase")):
                if str(item[key]).startswith("failed"):
                    raise RuntimeError("A preparation gate failed; production remains running")
            state["preparation"] = {"head_bytes": source["verified_bytes"],
                "worker_bytes": download["verified_bytes"], "scale_tensors": scales["tensors"],
                "fixture": fixture["phase"]}
            save("preparing")
            if (source["phase"] == "complete" and download["status"] == "complete"
                    and scales["status"] == "complete" and fixture["phase"] == "complete_idle"):
                break
            time.sleep(20)
        else:
            raise TimeoutError("Full-model preparation deadline; production remains running")
        shards = lambda value: {x["name"]: (x["bytes"], x["sha256"])
                                for x in value["verified_shards"]}
        if (shards(source) != shards(download) or len(shards(source)) != 120
                or source["verified_bytes"] != 183429201056):
            raise ValueError("The complete verified source manifests differ across ranks")
        bound = [preflight(rank, code) for rank in (0, 1)]
        if bound[0] != bound[1]:
            raise ValueError("Source/code/native qualification differs across the two Sparks")
        state["preflight"] = bound[0]
        attention = remote_json(0, ROOT / "diagnostics-v1/attention-dispatch-v1.json")
        if (not attention["passed"] or attention["unexpected_quantized_selectors_after"] != 0
                or attention["runtime_source_sha256"]["mapping"]
                != bound[0]["code"].get("attention_mapping.py")):
            raise ValueError("Qualify the exact fused-attention mapping before pausing GLM")
        state["attention_mapping_audit"] = {
            "mapping_sha256": attention["runtime_source_sha256"]["mapping"],
            "fused_counts": attention["fused_counts"]}
        state["fixture_evidence"] = []
        for mode in ("raw", "aligned"):
            accepted = [x for x in fixture["runs"] if x["mode"] == mode and x.get("accepted")]
            if len(accepted) != 1:
                raise ValueError("Adapter fixture acceptance is ambiguous")
            proof = remote_json(1, Path(accepted[0]["results"]) / "report.json")
            for name in ("model_plugin.py", "moe_runtime.py"):
                if proof["code_sha256"][name] != bound[0]["code"][name]:
                    raise ValueError("Adapter code changed after GPU fixture qualification")
            state["fixture_evidence"].append({"mode": mode, "proof": proof})
        originals = [inspect(rank, ORIGINAL_IDS[rank]) for rank in (0, 1)]
        if any(x["Image"] != IMAGE or not x["State"]["Running"] for x in originals):
            raise ValueError("The authorized original GLM service identity or state changed")
        state["originals"] = [identity(x) for x in originals]
        launches = {mode: [engine(rank, mode, revision, originals[rank], bound[0]["vocabulary"],
                                 attention_mapping=True)
                          for rank in (0, 1)] for mode in ("stock", "raw", "aligned")}
        state["launches"] = {mode: [x[1] for x in values] for mode, values in launches.items()}
        save("waiting_for_idle")
        spec = importlib.util.spec_from_file_location("guard", str(
            OLD / "triple-v2/src/integrations/marlin_scales/quiet_window.py"))
        assert spec is not None and spec.loader is not None
        guard = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(guard)
        previous, quiet = None, 0
        while time.monotonic() < deadline:
            now = guard.activity()
            quiet = quiet + 1 if not now["busy"] and previous == now["counters"] else 0
            previous = now["counters"]
            if quiet >= 4:
                state["idle_before_pause"] = now
                break
            time.sleep(5)
        else:
            raise TimeoutError("No idle window before the preparation deadline")
        watchdog = subprocess.Popen(["python3", "-u", str(code / "full_model_control.py"),
            "--revision", revision, "--watch", str(os.getpid())],
            stdout=(ROOT / f"watchdog-{revision}.log").open("xb"), stderr=subprocess.STDOUT,
            start_new_session=True)
        state["watchdog_pid"] = watchdog.pid
        for _ in range(20):
            ack = ROOT / f"watchdog-{revision}.json"
            if ack.exists() and json.loads(ack.read_bytes()).get("ready"):
                break
            time.sleep(.5)
        else:
            raise RuntimeError("Restoration watchdog did not arm")
        if guard.activity()["busy"]:
            raise RuntimeError("GLM became active before pause; production remains running")
        state.update(service_paused=True, restore_deadline_epoch=time.time() + 7200)
        save("pausing_service")
        for rank in (0, 1):
            run(rank, ["docker", "stop", "--timeout", "30", ORIGINAL_IDS[rank]], timeout=90)
        for mode in ("stock", "raw", "aligned"):
            arm: dict[str, Any] = {"mode": mode, "started_epoch": time.time()}
            state["arms"].append(arm)
            save("starting_" + mode)
            for rank in (1, 0):
                arm[f"rank{rank}_container"] = run(rank, launches[mode][rank][0], 120).strip()
            startup_deadline = time.monotonic() + 1200
            while time.monotonic() < startup_deadline:
                if health(18784):
                    break
                for rank in (0, 1):
                    info = inspect(rank, launches[mode][rank][1]["name"])
                    if not info["State"]["Running"]:
                        raise RuntimeError(f"{mode} engine rank {rank} exited during startup")
                time.sleep(5)
            else:
                raise TimeoutError(f"{mode} engine startup deadline")
            arm["ready_epoch"] = time.time()
            save("benchmarking_" + mode)
            results = ROOT / f"run-{revision}" / "results"
            results.mkdir(exist_ok=True)
            client = ["docker", "run", "-d", "--name", f"mxwave-glm53-bench-{revision}-{mode}",
                "--network", "host", "--cpus", "2", "--memory", "1536m", "--memory-swap", "1536m",
                "--user", f"{os.getuid()}:{os.getgid()}", "-e", "CUDA_VISIBLE_DEVICES=",
                "-e", "NVIDIA_VISIBLE_DEVICES=void", "-v", f"{AUDIT}:/audit:ro", "-v",
                f"{results}:/results", "--entrypoint", "/usr/bin/python3", IMAGE, "-u",
                f"/audit/full-model-v1/full-{revision}-src/benchmark_full_model.py", "--mode", mode,
                "--root", "/results", "--vocabulary", str(bound[0]["vocabulary"])]
            run(0, client)
            client_deadline = time.monotonic() + 1800
            while time.monotonic() < client_deadline:
                info = inspect(0, f"mxwave-glm53-bench-{revision}-{mode}")
                if not info["State"]["Running"]:
                    if info["State"]["ExitCode"]:
                        raise RuntimeError(f"{mode} benchmark client exited with failure")
                    report = json.loads((results / f"{mode}-benchmark.json").read_bytes())
                    if report["status"] != "complete":
                        raise RuntimeError("Benchmark client did not record completion")
                    arm.update(status="complete", finished_epoch=time.time())
                    break
                time.sleep(5)
            else:
                raise TimeoutError(f"{mode} benchmark client deadline")
            save("measured_" + mode)
            stop_owned(revision)
        success = True
    except BaseException as exc:
        state["error"] = repr(exc)
        save("failed_pending_restoration" if state["service_paused"] else "preparation_failed")
        raise
    finally:
        if state["service_paused"]:
            save("restoring")
            state["restoration"] = restore(revision, state_path)
            state["service_paused"] = False
            save("restored_analyzing" if success else "failed_restored")
    if success:
        try:
            for mode in ("stock", "raw", "aligned"):
                local = ROOT / f"run-{revision}" / mode
                destination = local / "traces-worker"
                destination.mkdir()
                run(0, ["rsync", "-a", "--timeout=120", "--bwlimit=65536", "-e",
                        shlex.join(SSH[:-1]), SSH[-1] + ":" + str(local / "traces") + "/",
                        str(destination) + "/"], timeout=600)
                for rank in (0, 1):
                    name = launches[mode][rank][1]["name"]
                    capture = ("import subprocess,sys; r=subprocess.run(['docker','logs',sys.argv[1]],"
                               "stdout=subprocess.PIPE,stderr=subprocess.STDOUT); "
                               "sys.stdout.buffer.write(r.stdout)")
                    (local / f"rank{rank}-engine.log").write_text(run(
                        rank, ["python3", "-c", capture, name], timeout=60))
            analysis = ["docker", "run", "--rm", "--network", "none", "--cpus", "2",
                "--memory", "2g", "--memory-swap", "2g", "--user", f"{os.getuid()}:{os.getgid()}",
                "-e", "CUDA_VISIBLE_DEVICES=", "-e", "NVIDIA_VISIBLE_DEVICES=void",
                "-v", f"{code}:/src:ro", "-v", f"{ROOT / ('run-' + revision)}:/results",
                "--entrypoint", "/usr/bin/python3", IMAGE, "/src/compare_full_model.py",
                "--root", "/results"]
            state["analysis"] = run(0, analysis, timeout=600)
            save("complete")
        except BaseException as exc:
            state["analysis_error"] = repr(exc)
            save("analysis_failed_service_restored")
            raise


def main() -> None:
    """Run the controller or its independent watchdog, with immutable revision names."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--watch", type=int)
    args = parser.parse_args()
    if not args.revision.isalnum():
        raise ValueError("Revision must be alphanumeric")
    if args.watch is not None:
        watch(args.revision, args.watch, ROOT / f"full-controller-{args.revision}.json")
    else:
        control(args.revision)


if __name__ == "__main__":
    main()
