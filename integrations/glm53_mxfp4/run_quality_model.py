"""Detached stock versus U8 combined quality evaluation with independent GLM restoration."""

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

PREPARED_NAME: str | None = None
CLIENT_SCRIPT = "measure_quality_model.py"
ANALYZE_SCRIPT = "analyze_quality_model.py"
CLIENT_MEMORY = "2g"
CLIENT_WAIT_SECONDS = 2400
OUTAGE_SECONDS = 8400
ENTRY_SCRIPT = Path(__file__).resolve()
SANDBOX_TASKS = True


def check_prepared(
    code: Path, results: Path, bound: list[dict[str, Any]], name: str,
) -> dict[str, Any]:
    """Require both source bindings and every prospective protocol before pausing a service."""
    prepared = json.loads((c.ROOT / name).read_bytes())
    if prepared["bindings"] != bound:
        raise ValueError("Prepared paired-run source differs from both current nodes")
    for filename, digest in prepared["protocol_files_sha256"].items():
        if Path(filename).name != filename:
            raise ValueError("Prepared protocols must be local files")
        if hashlib.sha256((results / filename).read_bytes()).hexdigest() != digest:
            raise ValueError("Frozen paired-run protocol changed: " + filename)
    if not {"protocol.json", "quality-protocol.json", "graph-quality-plan.json"}.issubset(
        prepared["protocol_files_sha256"]
    ):
        raise ValueError("Require every prospective graph-quality protocol")
    return prepared


def launch(
    rank: int,
    revision: str,
    label: str,
    original: dict[str, Any],
    vocabulary: int,
    manifest_sha: str,
) -> tuple[list[str], dict[str, Any]]:
    """Compare stock arithmetic with the qualified U8 combined candidate on identical weights."""
    argv, meta = c.engine(
        rank,
        "kda" if label == "candidate" else "stock",
        revision,
        original,
        vocabulary,
        attention_mapping=True,
    )
    stage = "raw" if label == "control" else "aligned"
    name = f"mxwave-glm53-full-{revision}-{stage}-r{rank}"
    argv[argv.index("--name") + 1] = name
    meta["name"] = name
    for command in (argv, meta["command"]):
        command[command.index("--worker-extension-cls") + 1] = "quality_worker.QualityWorker"
        command[command.index("--compilation-config") + 1] = json.dumps(
            {
                "mode": 0,
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": [1],
                "max_cudagraph_capture_size": 1,
            }
        )
    for command in (argv, meta["command"]):
        command.extend(["--tool-call-parser", "glm47", "--enable-auto-tool-choice", "--seed", "42"])
    extra = {
        "MXWAVE_GLM53_KDA_FUSED_MAX_TOKENS": "8",
        "MXWAVE_MOE_MODE": "aligned" if label == "candidate" else "stock",
        "MXWAVE_MOE_DISPATCH": "dispatcher",
        "MXWAVE_GLM53_VOCAB_MODE": "packed" if label == "candidate" else "stock",
        "MXWAVE_GLM53_VOCAB_ARTIFACTS": "/audit/full-model-v1/vocab-v2",
        "MXWAVE_GLM53_VOCAB_MANIFEST_SHA256": manifest_sha,
        "MXWAVE_PALETTE_MODEL": "1" if label == "candidate" else "0",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    if label == "candidate":
        extra["MXWAVE_MOE_ALIGNED_LIBRARY"] = (
            "/audit/full-model-v1/quality-best-v1-native/libmxwave_moe_sparse.so"
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
        "import engine_boot; from quality_worker import QualityWorker; "
        "from quality_worker import runtime_metadata; import msgspec; "
        "msgspec.msgpack.encode(runtime_metadata(False)); "
        "from vllm.parser import ParserManager; "
        "assert '--enable-auto-tool-choice' in __import__('sys').argv; "
        "assert ParserManager.get_parser(tool_parser_name='glm47',reasoning_parser_name='glm45',"
        "enable_auto_tools=True).tool_parser_cls is not None; "
        "from combined_worker import CombinedWorker; "
        "assert issubclass(QualityWorker,CombinedWorker); "
        "import importlib,sys; "
        "extension=sys.argv[sys.argv.index('--worker-extension-cls')+1]; "
        "module,name=extension.rsplit('.',1); "
        "assert issubclass(getattr(importlib.import_module(module),name),QualityWorker); "
        "engine_boot.main()"
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
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--require-stable-repeat", action="store_true")
    parser.add_argument("--task-repeats", type=int, choices=(5,))
    args = parser.parse_args()
    if args.task_repeats and (args.profile or args.require_stable_repeat):
        raise ValueError("Practical-repeat diagnostics use the saved numerical/speed evidence")
    revision = args.revision
    if not revision.isalnum():
        raise ValueError("Use a fresh alphanumeric evidence revision")
    code = c.ROOT / f"full-{revision}-src"
    path = c.ROOT / f"quality-model-controller-{revision}.json"
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
        "purpose": (
            "Five fixed practical-task rounds per arm; diagnostic only; prior strict verdict preserved"
            if args.task_repeats
            else "Stock versus best U8 combined quality; repeatability, held-out loss and tasks"
        ),
        "task_repeats": args.task_repeats,
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
r=Path(sys.argv[1]); code=Path(sys.argv[2]); n=r/'quality-best-v1-native'
m=n/'manifest.json'; data=json.loads(m.read_bytes())
assert data['sparse_palette_policy']=='affine_only' and data['sparse_shared_stage']
p=n/'libmxwave_moe_sparse.so'; h=hashlib.sha256(p.read_bytes()).hexdigest()
assert h=='9f5d96dcb0762d7e5ed5436ca296f70d590e73797d40522e4d8c7d6f3902f5f0'
assert h==data['libraries_sha256'][p.name]
binding=json.loads((n/'qualification-binding.json').read_bytes())
assert binding['native_sha256']==h
comparison=n/'comparison-manifest.json'
assert hashlib.sha256(comparison.read_bytes()).hexdigest()==binding['comparison_manifest_sha256']
assert json.loads(comparison.read_bytes())['libraries_sha256']['libmxwave_moe_sparse.so']==h
for name,sha in binding['source_sha256'].items():
 assert hashlib.sha256((code/name).read_bytes()).hexdigest()==sha,name
for name,sha in binding['proof_sha256'].items():
 p=n/name; assert hashlib.sha256(p.read_bytes()).hexdigest()==sha
 proof=json.loads(p.read_bytes()); assert proof['status']=='passed'
 assert proof['candidate_shared_stage'] and proof['candidate_affine_only']
 assert proof['runtime_sha256']==binding['source_sha256']['moe_runtime.py']
 assert proof['codec_sha256']==binding['source_sha256']['scale_tiles.py']
 assert proof['build_manifest_sha256']==binding['comparison_manifest_sha256']
 assert sum(len(part['cases']) for part in proof['partitions'])==28
 assert all(case['current_codec_control_candidate_eager_stream_graph_bits_exact']
            for part in proof['partitions'] for case in part['cases'])
d=r/'quality-best-v1-data'; manifest=json.loads((d/'dataset-manifest.json').read_bytes())
assert manifest['status']=='prepared'
assert hashlib.sha256((d/'wikitext-test.txt').read_bytes()).hexdigest()==manifest['text_sha256']
assert hashlib.sha256((d/'gsm8k-test.jsonl').read_bytes()).hexdigest()==manifest['files']['gsm8k-test.jsonl']['sha256']
assert 'quality_worker:install' in (code/'engine_plugin/pyproject.toml').read_text()
print(json.dumps({'library_sha256':h,'binding_sha256':hashlib.sha256((n/'qualification-binding.json').read_bytes()).hexdigest(),'dataset_manifest_sha256':hashlib.sha256((d/'dataset-manifest.json').read_bytes()).hexdigest()}))
"""
        state["candidate_native"] = [
            json.loads(c.run(rank, ["python3", "-S", "-c", check, str(c.ROOT), str(code)]))
            for rank in (0, 1)
        ]
        if state["candidate_native"][0] != state["candidate_native"][1]:
            raise ValueError("Candidate native artifacts/datasets differ between nodes")
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
        if PREPARED_NAME is not None:
            state["prepared"] = check_prepared(code, results, bound, PREPARED_NAME)
        else:
            results.mkdir(parents=True, exist_ok=False)
            protocol = json.loads(
                (c.ROOT / "run-combineddispatchv1/results/kda/protocol.json").read_bytes()
            )
            if len(protocol["quality"]) != 16:
                raise ValueError("Frozen quality protocol changed")
            c.write(results / "protocol.json", protocol)
        if args.task_repeats:
            frozen = c.ROOT / "run-stablequalityv1/results/quality-protocol.json"
            raw = frozen.read_bytes()
            suite = json.loads(raw)
            if len(suite["tasks"]) != 34 or len({r["name"] for r in suite["tasks"]}) != 34:
                raise ValueError("Preserve all 34 practical cases from the completed experiment")
            (results / "quality-protocol.json").write_bytes(raw)
            source_audit = (c.ROOT / "scale-palette-all-v1/results/screen.json").read_bytes()
            (results / "task-repeat-source-audit.json").write_bytes(source_audit)
            c.write(results / "task-repeat-plan.json", {
                "version": 1, "rounds_per_arm": args.task_repeats,
                "arms_in_order": ["control", "candidate"],
                "quality_protocol_sha256": hashlib.sha256(raw).hexdigest(),
                "source_audit_sha256": hashlib.sha256(source_audit).hexdigest(),
                "tasks_per_round": 34, "seed": 42,
                "task_warmup_rounds": 0,
                "startup_warmup": "Engine initialization/capture; first task round retained",
                "all_rounds_reported": True, "stop_or_selection_based_on_scores": False,
                "qualification_attempted": False,
                "prior_experiment": "stablequalityv1", "prior_quality_qualified": False,
                "frozen_before_model_launch_epoch": time.time(),
            })
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
                str(ENTRY_SCRIPT),
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
        state.update(service_paused=True, restore_deadline_epoch=time.time() + OUTAGE_SECONDS)
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
                CLIENT_MEMORY,
                "--memory-swap",
                CLIENT_MEMORY,
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
                f"/audit/full-model-v1/full-{revision}-src/"
                + ("measure_task_repeats.py" if args.task_repeats else CLIENT_SCRIPT),
                "--root",
                "/results",
                "--label",
                label,
                "--vocabulary",
                str(bound[0]["vocabulary"]),
                "--data",
                "/audit/full-model-v1/quality-best-v1-data",
                "--cpu-audit",
                "/audit/full-model-v1/scale-palette-all-v1/results/screen.json",
            ]
            if args.profile:
                argv.append("--profile")
            if args.require_stable_repeat:
                argv.append("--require-stable-repeat")
            if args.task_repeats:
                argv.extend(["--task-repeats", str(args.task_repeats)])
            c.run(0, argv)
            arm["client_state"] = wait_container(0, client_name, CLIENT_WAIT_SECONDS)
            if arm["client_state"]["ExitCode"]:
                raise RuntimeError(f"{label} full-model source/quality client failed")
            if SANDBOX_TASKS:
                for task_label in (
                    [f"{label}-r{i}" for i in range(args.task_repeats)]
                    if args.task_repeats else [label]
                ):
                    sandbox(results, revision, task_label)
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
            "/work/analyze_task_repeats.py" if args.task_repeats
            else "/work/" + ANALYZE_SCRIPT,
            "--root",
            "/results",
        ]
        c.run(0, argv, timeout=180)
        state["review"] = json.loads((results / "quality-model-review.json").read_bytes())
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
