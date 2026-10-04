"""Run frozen MTP diagnostics and matched-target context checks, restoring Spark serving."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from run_corrected_controls import IMAGE, _available_memory, _checkpoint_hash, _hash_file


def _source_digest() -> str:
    root = Path(__file__).resolve().parent.parent
    scripts = (
        "evaluate_mtp_followup.py",
        "mtp_task_screen.py",
        "run_mtp_followup.py",
        "evaluate_fast_pilots.py",
        "analyze_kl_prefixes.py",
        "evaluate_next_token_kl.py",
        "evaluate_quantization_controls.py",
        "run_corrected_controls.py",
    )
    files = sorted((root / "mxwave").rglob("*.py")) + [root / "scripts" / s for s in scripts]
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def _docker(*arguments: str) -> str:
    return subprocess.check_output(["docker", *arguments], text=True).strip()


def _job(root: Path, name: str, command: list[str], model: Path | None, resume: bool) -> None:
    expected = ["1800", "python3", "scripts/evaluate_mtp_followup.py", *command]
    invocation = [
        "run",
        "-d",
        "--init",
        "--name",
        name,
        "--network",
        "none",
        "--ipc=private",
        "--shm-size=1g",
        f"--memory={'108g' if model else '4g'}",
        f"--memory-swap={'108g' if model else '4g'}",
        "-e",
        "OMP_NUM_THREADS=4",
        "-e",
        "PYTHONUNBUFFERED=1",
        "-e",
        "HF_HUB_OFFLINE=1",
        "-e",
        "PYTHONPATH=/work/runtime",
        "-e",
        "VLLM_CACHE_ROOT=/cache/vllm",
        "-e",
        "FLASHINFER_WORKSPACE_BASE=/cache/flashinfer",
        "-e",
        "TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor",
        "-e",
        "TRITON_CACHE_DIR=/cache/triton",
    ]
    for directory in ("validation", "runtime", "measurements"):
        invocation += [
            "--mount",
            f"type=bind,src={root / directory},dst=/work/{directory}"
            + (",readonly" if directory != "measurements" else ""),
        ]
    invocation += [
        "--mount",
        "type=bind,src="
        + str(root.parent / "mxwave-corrected-controls-2026-10-03/runtime-cache")
        + ",dst=/cache",
        "--workdir",
        "/work/runtime",
        "--entrypoint",
        "timeout",
    ]
    if model is not None:
        invocation += ["--gpus", "all", "--mount", f"type=bind,src={model},dst=/model,readonly"]
    invocation += [IMAGE, *expected]
    print(f"START {name}", flush=True)
    inspection = subprocess.run(
        ["docker", "inspect", name], capture_output=True, text=True, check=False
    )
    if inspection.returncode:
        _docker(*invocation)
    else:
        previous = json.loads(inspection.stdout)[0]
        if (
            not resume
            or previous["State"]["Running"]
            or previous["Image"] != IMAGE
            or previous["Config"]["Cmd"] != expected
        ):
            raise ValueError("Existing container cannot be reused")
        if previous["State"]["ExitCode"] == 0 and not previous["State"]["OOMKilled"]:
            print(f"REUSE {name}", flush=True)
            return
        _docker("start", name)
    deadline = time.monotonic() + 1800
    try:
        while True:
            state = json.loads(_docker("inspect", name))[0]["State"]
            if not state["Running"]:
                if state["ExitCode"] or state["OOMKilled"]:
                    raise RuntimeError(f"Job failed: {name}: {state}")
                break
            if _available_memory() < 16 * 1024**3 or time.monotonic() >= deadline:
                raise RuntimeError(f"Memory/time stop gate: {name}")
            time.sleep(5)
    finally:
        state = json.loads(_docker("inspect", name))[0]["State"]
        if state["Running"]:
            _docker("stop", "--timeout", "10", name)
        with (root / f"{name}.log").open("w") as stream:
            subprocess.run(["docker", "logs", name], stdout=stream, stderr=stream, check=False)
    print(f"DONE {name}", flush=True)


def _interrupt(signum: int, frame: Any) -> None:
    raise KeyboardInterrupt(f"Signal {signum}")


def main() -> None:
    """Rehash real checkpoints, lock GPU work and restore the prior service on all exits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spark-root", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--pause-serving-container", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.experiment_root
    protocol = json.loads((root / "validation/protocol.json").read_text())
    if protocol["executed_source_sha256"] != _source_digest():
        raise ValueError("Frozen executed sources changed")
    (root / "measurements").mkdir(exist_ok=True)
    identity_path = root / "run-identity.json"
    if identity_path.exists() and not args.resume:
        raise FileExistsError("Run already exists; use --resume with unchanged inputs")
    original = args.spark_root / "experiments/mxwave-corrected-controls-2026-10-03"
    models = {
        "h64": original / "checkpoints/h64",
        "bf16": args.spark_root / "models/qwen3.8-27b-bf16/huggingface",
    }
    identities = {}
    for label, model in models.items():
        identities[label] = _checkpoint_hash(model)
        expected = json.loads((original / "measurements" / f"{label}.json").read_text())
        if identities[label] != expected["checkpoint_sha256"]:
            raise ValueError("Actual checkpoint bytes changed")
    identity = {
        "checkpoint_sha256": identities,
        "runtime_image": IMAGE,
        "executed_source_sha256": _source_digest(),
        "runner_sha256": _hash_file(Path(__file__)),
        "protocol_sha256": _hash_file(root / "validation/protocol.json"),
    }
    if identity_path.exists():
        previous = json.loads(identity_path.read_text())
        if identity != previous:
            raise ValueError("Cannot resume a changed identity")
    else:
        identity_path.write_text(json.dumps(identity, indent=2) + "\n")
    signal.signal(signal.SIGTERM, _interrupt)
    signal.signal(signal.SIGINT, _interrupt)
    with (root.parent / ".mxwave-gpu.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        was_running = json.loads(_docker("inspect", args.pause_serving_container))[0]["State"][
            "Running"
        ]
        try:
            if was_running:
                _docker("stop", "--timeout", "30", args.pause_serving_container)
            for label, profile, mtp, phase in (
                ("h64", "graphs", 0, "diagnostics"),
                ("h64", "graphs", 2, "diagnostics"),
                ("h64", "eager", 0, "diagnostics"),
                ("h64", "eager", 2, "diagnostics"),
                ("h64", "graphs", 0, "long"),
                ("bf16", "graphs", 0, "long"),
            ):
                _job(
                    root,
                    f"{root.name}-{label}-{profile}-mtp{mtp}-{phase}",
                    [
                        "collect",
                        "--model",
                        "/model",
                        "--label",
                        label,
                        "--profile",
                        profile,
                        "--mtp-tokens",
                        str(mtp),
                        "--phase",
                        phase,
                        "--checkpoint-sha256",
                        identities[label],
                        "--protocol",
                        "/work/validation/protocol.json",
                        "--output-dir",
                        "/work/measurements",
                    ],
                    models[label],
                    args.resume,
                )
            _job(
                root,
                f"{root.name}-compare",
                [
                    "compare",
                    "--protocol",
                    "/work/validation/protocol.json",
                    "--output-dir",
                    "/work/measurements",
                    "--output",
                    "/work/measurements/comparison.json",
                ],
                None,
                args.resume,
            )
        finally:
            if was_running:
                _docker("start", args.pause_serving_container)
                print(f"RESTORED {args.pause_serving_container}", flush=True)


if __name__ == "__main__":
    main()
