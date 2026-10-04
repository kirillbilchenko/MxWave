"""Run existing-checkpoint pilots serially and restore the paused Spark service."""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from run_corrected_controls import IMAGE, _available_memory, _checkpoint_hash, _hash_file


def _docker(*arguments: str) -> str:
    return subprocess.check_output(["docker", *arguments], text=True).strip()


def _run_job(
    root: Path, name: str, command: list[str], model: Path | None = None, *, resume: bool = False
) -> None:
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
        "--memory=108g",
        "--memory-swap=108g",
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
        "--mount",
        f"type=bind,src={root},dst=/work",
        "--mount",
        f"type=bind,src={root.parent / 'mxwave-corrected-controls-2026-10-03/runtime-cache'},dst=/cache",
        "--workdir",
        "/work/runtime",
        "--entrypoint",
        "timeout",
    ]
    if model is not None:
        invocation += ["--gpus", "all", "--mount", f"type=bind,src={model},dst=/model,readonly"]
    invocation += [IMAGE, "1800", "python3", "scripts/evaluate_fast_pilots.py", *command]
    print(f"START {name}", flush=True)
    inspection = subprocess.run(
        ["docker", "inspect", name], capture_output=True, text=True, check=False
    )
    if inspection.returncode:
        _docker(*invocation)
    else:
        prior = json.loads(inspection.stdout)[0]
        expected_command = ["1800", "python3", "scripts/evaluate_fast_pilots.py", *command]
        if (
            not resume
            or prior["State"]["Running"]
            or prior["Image"] != IMAGE
            or prior["Config"]["Cmd"] != expected_command
        ):
            raise ValueError(f"Existing pilot container cannot be reused: {name}")
        if prior["State"]["ExitCode"] == 0 and not prior["State"]["OOMKilled"]:
            print(f"REUSE completed {name}", flush=True)
            return
        _docker("start", name)
    deadline = time.monotonic() + 1800
    try:
        while True:
            state = json.loads(_docker("inspect", name))[0]["State"]
            if not state["Running"]:
                if state["ExitCode"] or state["OOMKilled"]:
                    raise RuntimeError(f"Pilot failed: {name}: {state}")
                break
            if _available_memory() < 16 * 1024**3 or time.monotonic() >= deadline:
                raise RuntimeError(f"Pilot reached memory/time stop gate: {name}")
            time.sleep(5)
    finally:
        state = json.loads(_docker("inspect", name))[0]["State"]
        if state["Running"]:
            _docker("stop", "--timeout", "10", name)
        with (root / f"{name}.log").open("w") as stream:
            subprocess.run(["docker", "logs", name], stdout=stream, stderr=stream, check=False)
    print(f"DONE {name}", flush=True)


def _interrupt(signum: int, frame: Any) -> None:
    raise KeyboardInterrupt(f"Received signal {signum}")


def main() -> None:
    """Verify real weight bytes, run bounded pilot jobs, and restore prior serving state."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spark-root", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--pause-serving-container", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.experiment_root
    if not (root / "validation/protocol.json").is_file():
        raise FileNotFoundError("Freeze the pilot protocol before running")
    identity_path = root / "run-identity.json"
    if identity_path.exists() and not args.resume:
        raise FileExistsError("Refusing to overwrite a pilot run")
    (root / "measurements").mkdir(exist_ok=True)
    original = args.spark_root / "experiments/mxwave-corrected-controls-2026-10-03"
    models = {
        "bf16": args.spark_root / "models/qwen3.8-27b-bf16/huggingface",
        "h64": original / "checkpoints/h64",
    }
    identities = {}
    for label, model in models.items():
        # Rehash actual source bytes rather than trusting the retained manifest.
        identities[label] = _checkpoint_hash(model)
        expected = json.loads((original / "measurements" / f"{label}.json").read_text())
        if identities[label] != expected["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint changed since corrected controls: {label}")
    record = {
        "checkpoint_sha256": identities,
        "runtime_image": IMAGE,
        "runner_sha256": _hash_file(Path(__file__)),
        "protocol_sha256": _hash_file(root / "validation/protocol.json"),
    }
    if identity_path.exists():
        previous = json.loads(identity_path.read_text())
        for key in ("checkpoint_sha256", "runtime_image", "protocol_sha256"):
            if previous[key] != record[key]:
                raise ValueError(f"Cannot resume a changed pilot identity: {key}")
        record["parent_run_identity_sha256"] = _hash_file(identity_path)
        identity_path = root / f"resume-identity-{time.time_ns()}.json"
    identity_path.write_text(json.dumps(record, indent=2) + "\n")
    signal.signal(signal.SIGTERM, _interrupt)
    signal.signal(signal.SIGINT, _interrupt)
    was_running = json.loads(_docker("inspect", args.pause_serving_container))[0]["State"][
        "Running"
    ]
    try:
        if was_running:
            _docker("stop", "--timeout", "30", args.pause_serving_container)
        for label, mtp, quality, serving in (
            ("h64", 0, True, True),
            ("h64", 2, False, True),
            ("bf16", 0, True, False),
        ):
            _run_job(
                root,
                f"{root.name}-{label}-mtp{mtp}",
                [
                    "collect",
                    "--model",
                    "/model",
                    "--label",
                    label,
                    "--mtp-tokens",
                    str(mtp),
                    "--checkpoint-sha256",
                    identities[label],
                    "--protocol",
                    "/work/validation/protocol.json",
                    "--output-dir",
                    "/work/measurements",
                    *(["--long-context"] if quality else []),
                    *(["--serving"] if serving else []),
                ],
                models[label],
                resume=args.resume,
            )
        _run_job(
            root,
            f"{root.name}-compare",
            [
                "compare",
                "--protocol",
                "/work/validation/protocol.json",
                "--output-dir",
                "/work/measurements",
                "--output",
                "/work/pilot-comparison.json",
            ],
            resume=args.resume,
        )
    finally:
        if was_running:
            _docker("start", args.pause_serving_container)
            print(f"RESTORED {args.pause_serving_container}", flush=True)


if __name__ == "__main__":
    main()
