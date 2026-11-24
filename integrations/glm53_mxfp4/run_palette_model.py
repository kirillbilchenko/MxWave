"""Detached two-arm full-model storage/quality check with independent GLM restoration."""

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


def launch(
    rank: int,
    revision: str,
    label: str,
    original: dict[str, Any],
    vocabulary: int,
    manifest_sha: str,
) -> tuple[list[str], dict[str, Any]]:
    """Keep the qualified combined stack fixed while changing only its expert scale consumer."""
    argv, meta = c.engine(rank, "kda", revision, original, vocabulary, attention_mapping=True)
    stage = "raw" if label == "control" else "aligned"
    name = f"mxwave-glm53-full-{revision}-{stage}-r{rank}"
    argv[argv.index("--name") + 1] = name
    meta["name"] = name
    for command in (argv, meta["command"]):
        command[command.index("--worker-extension-cls") + 1] = (
            "palette_model_worker.PaletteModelWorker"
        )
        command[command.index("--compilation-config") + 1] = json.dumps(
            {
                "mode": 0,
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": [1],
                "max_cudagraph_capture_size": 1,
            }
        )
    extra = {
        "MXWAVE_GLM53_KDA_FUSED_MAX_TOKENS": "8",
        "MXWAVE_MOE_MODE": "aligned",
        "MXWAVE_MOE_DISPATCH": "dispatcher",
        "MXWAVE_GLM53_VOCAB_MODE": "packed",
        "MXWAVE_GLM53_VOCAB_ARTIFACTS": "/audit/full-model-v1/vocab-v2",
        "MXWAVE_GLM53_VOCAB_MANIFEST_SHA256": manifest_sha,
        "MXWAVE_PALETTE_MODEL": "1" if label == "candidate" else "0",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    if label == "candidate":
        extra["MXWAVE_MOE_ALIGNED_LIBRARY"] = (
            "/audit/full-model-v1/palette-full-v1/native/libmxwave_moe_sparse.so"
        )
    meta["environment"].update(extra)
    for key, value in extra.items():
        index = next((i for i, token in enumerate(argv) if token.startswith(key + "=")), None)
        if index is not None:
            argv[index] = key + "=" + value
        else:
            at = argv.index("--entrypoint")
            argv[at:at] = ["-e", key + "=" + value]
    return argv, meta


def cli_check(meta: dict[str, Any]) -> str:
    """Validate the actual plugin/extension/CLI before interrupting the running service."""
    argv = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--cpus",
        "2",
        "--memory",
        "3g",
        "--memory-swap",
        "3g",
        "-e",
        "CUDA_VISIBLE_DEVICES=",
        "-e",
        "NVIDIA_VISIBLE_DEVICES=void",
        "-e",
        "MXWAVE_VALIDATE_ONLY=1",
        "-e",
        "UV_CACHE_DIR=/tmp/mxwave-uv-cache",
    ]
    for source, target, access in meta["mounts"]:
        argv.extend(["-v", f"{source}:{target}:{access}"])
    for key, value in meta["environment"].items():
        argv.extend(["-e", key + "=" + value])
    check = (
        "from vllm import platforms; from vllm.platforms.cpu import CpuPlatform; "
        "platforms.current_platform=CpuPlatform(); "
        "import engine_boot; from palette_model_worker import PaletteModelWorker; "
        "from combined_worker import CombinedWorker; "
        "assert issubclass(PaletteModelWorker,CombinedWorker); engine_boot.main()"
    )
    argv.extend(["--entrypoint", "/usr/bin/python3", c.IMAGE, "-c", check, *meta["command"]])
    return c.run(0, argv, timeout=240)


def wait_container(rank: int, name: str, seconds: int) -> dict[str, Any]:
    """Wait on only this experiment's bounded container, retaining failed exit information."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        info = c.inspect(rank, name)
        if not info["State"]["Running"]:
            return info["State"]
        time.sleep(3)
    raise TimeoutError(f"Experiment container deadline: {name}")


def sandbox(root: Path, revision: str, label: str) -> None:
    """Execute generated functions only in a disposable CPU container with no network/host writes."""
    path = root / label / "real-work.json"
    report = json.loads(path.read_bytes())
    tests = {
        "slugify": """f=ns['slugify']
for text,expected in [('', ''),('  Hello, WORLD! ','hello-world'),('Crème brûlée','creme-brulee'),('a__b---c','a-b-c'),('你好',''),('A 1 / B 2','a-1-b-2')]:
 assert f(text)==expected,(text,f(text),expected)
""",
        "merge_intervals": """f=ns['merge_intervals']
for data,expected in [([],[]),([[1,3],[2,6],[8,10],[10,12]],[[1,6],[8,12]]),([[1,10],[2,3]],[[1,10]]),([[5,5],[-2,0],[0,2]],[[-2,2],[5,5]])]:
 import copy
 before=copy.deepcopy(data)
 assert f(data)==expected
 assert data==before,'Input mutated'
""",
    }
    for row in report["cases"]:
        if not row.get("sandbox_pending"):
            continue
        name = row["name"]
        if name not in tests or row["code_file"] != name + ".py":
            raise ValueError("Unexpected generated-code artifact")
        source = root / label / row["code_file"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != row["code_sha256"]:
            raise ValueError("Generated code changed before sandbox scoring")
        child = (
            "import resource\nresource.setrlimit(resource.RLIMIT_FSIZE,(1048576,1048576))\n"
            "resource.setrlimit(resource.RLIMIT_CPU,(5,5))\n"
            "ns={}\nexec(compile(open('/solution.py').read(),'/solution.py','exec'),ns)\n"
            + tests[name]
            + "\nprint('MXWAVE_TESTS_PASSED')\n"
        )
        program = (
            "import json,subprocess,sys,tempfile\n"
            "with tempfile.TemporaryFile() as f:\n"
            " try:\n"
            f"  p=subprocess.run([sys.executable,'-S','-c',{child!r}],stdout=f,stderr=f,timeout=10)\n"
            "  f.seek(0); text=f.read(32768).decode(errors='replace')\n"
            "  result={'passed':p.returncode==0 and text.rstrip().endswith('MXWAVE_TESTS_PASSED'),"
            "'exit_code':p.returncode,'output':text[-2000:]}\n"
            " except subprocess.TimeoutExpired:\n"
            "  result={'passed':False,'error':'sandbox timeout'}\n"
            " print(json.dumps(result))\n"
        )
        argv = [
            "docker",
            "run",
            "--rm",
            "--name",
            f"mxwave-sandbox-{revision}-{label}-{name}",
            "--network",
            "none",
            "--cpus",
            "1",
            "--memory",
            "256m",
            "--memory-swap",
            "256m",
            "--pids-limit",
            "64",
            "--user",
            "65534:65534",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--tmpfs",
            "/tmp:rw,nosuid,size=8m",
            "-e",
            "NVIDIA_VISIBLE_DEVICES=void",
            "-e",
            "CUDA_VISIBLE_DEVICES=",
            "-v",
            f"{source}:/solution.py:ro",
            "--entrypoint",
            "/usr/bin/timeout",
            c.IMAGE,
            "--kill-after=2",
            "20",
            "/usr/bin/python3",
            "-S",
            "-c",
            program,
        ]
        result = json.loads(c.run(0, argv, timeout=45))
        row.update(passed=result["passed"], sandbox_pending=False, sandbox=result)
        c.write(path, report)
    report["status"] = "complete"
    c.write(path, report)


def main() -> None:
    """Prepare and validate before the authorized idle pause; restore original identities on any exit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--watch", type=int)
    args = parser.parse_args()
    revision = args.revision
    if not revision.isalnum():
        raise ValueError("Use a fresh alphanumeric evidence revision")
    code = c.ROOT / f"full-{revision}-src"
    path = c.ROOT / f"palette-model-controller-{revision}.json"
    if args.watch is not None:
        c.watch(revision, args.watch, path)
        return
    if path.exists():
        raise FileExistsError("Preserve prior full-model evidence")
    lock = (c.ROOT / "diagnostics-v1/component.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state: dict[str, Any] = {
        "phase": "preparing",
        "service_paused": False,
        "started_epoch": time.time(),
        "arms": [],
        "purpose": "Full-model lossless-storage and practical/numerical qualification; no speed acceptance claim",
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
        state["originals"] = [c.identity(c.inspect(rank, c.ORIGINAL_IDS[rank])) for rank in (0, 1)]
        if any(not c.inspect(rank, c.ORIGINAL_IDS[rank])["State"]["Running"] for rank in (0, 1)):
            raise RuntimeError("Require both original GLM ranks restored before preparation")
        bound = [c.preflight(rank, code) for rank in (0, 1)]
        if bound[0] != bound[1]:
            raise ValueError("Full-model source/code differs between Sparks")
        state["preflight"] = bound[0]
        check = """import hashlib,json,sys
from pathlib import Path
r=Path(sys.argv[1]); m=r/'palette-full-v1/native/manifest.json'; data=json.loads(m.read_text())
assert data['sparse_palette_policy']=='affine_only'
p=m.parent/'libmxwave_moe_sparse.so'; h=hashlib.sha256(p.read_bytes()).hexdigest()
assert h==data['libraries_sha256'][p.name]
for stage in ('sparse','sparse-confirm'):
 p=r/'palette-full-v1/qualification'/(stage+'.json'); proof=json.loads(p.read_text())
 assert proof['status']=='passed' and proof['candidate_affine_only']
 assert sum(len(p['cases']) for p in proof['partitions'])==28
 assert proof['runtime_sha256']==hashlib.sha256((Path(sys.argv[2])/'moe_runtime.py').read_bytes()).hexdigest()
 assert all(case['current_codec_control_candidate_eager_stream_graph_bits_exact'] for p in proof['partitions'] for case in p['cases'])
print(json.dumps({'library_sha256':h,'manifest_sha256':hashlib.sha256(m.read_bytes()).hexdigest()}))
"""
        state["palette_native"] = [
            json.loads(c.run(rank, ["python3", "-S", "-c", check, str(c.ROOT), str(code)]))
            for rank in (0, 1)
        ]
        if state["palette_native"][0] != state["palette_native"][1]:
            raise ValueError("Palette native artifacts differ between nodes")
        manifest_sha = hashlib.sha256((c.ROOT / "vocab-v2/prepared.json").read_bytes()).hexdigest()
        launches = {
            label: [
                launch(
                    rank,
                    revision,
                    label,
                    c.inspect(rank, c.ORIGINAL_IDS[rank]),
                    bound[0]["vocabulary"],
                    manifest_sha,
                )
                for rank in (0, 1)
            ]
            for label in ("control", "candidate")
        }
        c.validate_transport_control(
            [x[1] for x in launches["control"]], [x[1] for x in launches["candidate"]]
        )
        state["launches"] = {label: [x[1] for x in values] for label, values in launches.items()}
        state["cpu_cli_validation"] = {
            label: cli_check(values[0][1]) for label, values in launches.items()
        }
        results = c.ROOT / f"run-{revision}/results"
        results.mkdir(parents=True, exist_ok=False)
        protocol = json.loads(
            (c.ROOT / "run-combineddispatchv1/results/kda/protocol.json").read_bytes()
        )
        if len(protocol["quality"]) != 16:
            raise ValueError("Frozen quality protocol changed")
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
                final = guard.activity()
                if not final["busy"] and final["counters"] == previous:
                    break
            time.sleep(5)
        else:
            raise TimeoutError("No idle window within thirty minutes")
        watchdog = subprocess.Popen(
            [
                "python3",
                "-u",
                str(Path(__file__).resolve()),
                "--revision",
                revision,
                "--watch",
                str(os.getpid()),
            ],
            stdout=(c.ROOT / f"watchdog-{revision}.log").open("xb"),
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        state["watchdog_pid"] = watchdog.pid
        for _ in range(30):
            ack = c.ROOT / f"watchdog-{revision}.json"
            if ack.exists() and json.loads(ack.read_text()).get("ready"):
                break
            time.sleep(1)
        else:
            raise RuntimeError("Independent restoration watchdog did not arm")
        final = guard.activity()
        if final["busy"] or final["counters"] != previous:
            raise RuntimeError("GLM became active before pause; production remains running")
        state.update(service_paused=True, restore_deadline_epoch=time.time() + 4200)
        save("pausing_glm")
        for rank in (0, 1):
            c.run(rank, ["docker", "stop", "--timeout", "30", c.ORIGINAL_IDS[rank]], timeout=90)
        for label in ("control", "candidate"):
            stage = "raw" if label == "control" else "aligned"
            arm: dict[str, Any] = {"label": label, "started_epoch": time.time()}
            state["arms"].append(arm)
            save("starting_" + label)
            for rank in (1, 0):
                c.run(rank, launches[label][rank][0], timeout=120)
            deadline = time.monotonic() + 1200
            while time.monotonic() < deadline:
                if c.health(18784):
                    break
                if any(
                    not c.inspect(rank, launches[label][rank][1]["name"])["State"]["Running"]
                    for rank in (0, 1)
                ):
                    raise RuntimeError(f"{label} engine exited during startup")
                time.sleep(5)
            else:
                raise TimeoutError(f"{label} model startup deadline")
            arm["ready_epoch"] = time.time()
            save("measuring_" + label)
            client_name = f"mxwave-glm53-bench-{revision}-{stage}"
            argv = [
                "docker",
                "run",
                "-d",
                "--name",
                client_name,
                "--network",
                "host",
                "--cpus",
                "2",
                "--memory",
                "2g",
                "--memory-swap",
                "2g",
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "-e",
                "NVIDIA_VISIBLE_DEVICES=void",
                "-e",
                "CUDA_VISIBLE_DEVICES=",
                "-e",
                f"PYTHONPATH=/audit/full-model-v1/full-{revision}-src",
                "-v",
                f"{c.AUDIT}:/audit:ro",
                "-v",
                f"{results}:/results",
                "--entrypoint",
                "/usr/bin/python3",
                c.IMAGE,
                "-u",
                f"/audit/full-model-v1/full-{revision}-src/measure_palette_model.py",
                "--root",
                "/results",
                "--label",
                label,
                "--vocabulary",
                str(bound[0]["vocabulary"]),
                "--cpu-audit",
                "/audit/full-model-v1/scale-palette-all-v1/results/screen.json",
            ]
            c.run(0, argv)
            arm["client_state"] = wait_container(0, client_name, 1200)
            if arm["client_state"]["ExitCode"]:
                raise RuntimeError(f"{label} full-model source/quality client failed")
            sandbox(results, revision, label)
            arm["status"] = "collected"
            save("finished_" + label)
            c.stop_owned(revision)
        argv = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--cpus",
            "2",
            "--memory",
            "1g",
            "--memory-swap",
            "1g",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "-e",
            "CUDA_VISIBLE_DEVICES=",
            "-e",
            "NVIDIA_VISIBLE_DEVICES=void",
            "-v",
            f"{code}:/work:ro",
            "-v",
            f"{results}:/results",
            "--entrypoint",
            "/usr/bin/python3",
            c.IMAGE,
            "/work/analyze_palette_model.py",
            "--root",
            "/results",
        ]
        c.run(0, argv, timeout=180)
        state["review"] = json.loads((results / "palette-model-review.json").read_bytes())
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
