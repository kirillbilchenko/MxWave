"""Run the frozen Spark control experiment serially, with bounded jobs and cleanup."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

IMAGE = "sha256:785fd756b4ae1cacda7612cb859dbf227330e8ced9a6fdea42e9379128bebce0"


def _docker(*arguments: str) -> str:
    return subprocess.check_output(["docker", *arguments], text=True).strip()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _protocol_path(root: Path) -> Path:
    paths = [
        path
        for path in (root / "validation/protocol.json", root / "validation/protocol.json.gz")
        if path.is_file()
    ]
    if not paths:
        raise FileNotFoundError("Freeze validation/protocol.json or protocol.json.gz first")
    if len(paths) != 1:
        raise ValueError("Both plain and compressed protocols exist; select a single artifact")
    return paths[0]


def _available_memory() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("Cannot read MemAvailable")


def _job(root: Path, name: str, mounts: list[str], command: list[str], *, memory: str) -> None:
    print(f"START {name}", flush=True)
    inspection = subprocess.run(
        ["docker", "inspect", name], capture_output=True, text=True, check=False
    )
    if inspection.returncode:
        invocation = [
            "run",
            "-d",
            "--init",
            "--name",
            name,
            "--gpus",
            "all",
            "--ipc=private",
            "--shm-size=1g",
            f"--memory={memory}",
            f"--memory-swap={memory}",
            "-e",
            "OMP_NUM_THREADS=4",
            "-e",
            "PYTHONUNBUFFERED=1",
            "-e",
            "HF_HUB_OFFLINE=1",
            "-e",
            "PYTHONPATH=/work/runtime-quality-v1",
            "-e",
            "VLLM_CACHE_ROOT=/work/runtime-cache/vllm",
            "-e",
            "FLASHINFER_WORKSPACE_BASE=/work/runtime-cache/flashinfer",
            "-e",
            "TORCHINDUCTOR_CACHE_DIR=/work/runtime-cache/torchinductor",
            "-e",
            "TRITON_CACHE_DIR=/work/runtime-cache/triton",
            "--mount",
            f"type=bind,src={root},dst=/work",
            *mounts,
            "--workdir",
            "/work/runtime-quality-v1",
            "--entrypoint",
            "timeout",
            IMAGE,
            "1800",
            "python3",
            *command,
        ]
        _docker(*invocation)
    deadline = time.monotonic() + 1800
    try:
        while True:
            state = json.loads(_docker("inspect", name))[0]["State"]
            if not state["Running"]:
                if state["ExitCode"] or state["OOMKilled"]:
                    raise RuntimeError(
                        f"{name} failed: exit={state['ExitCode']}, OOM={state['OOMKilled']}"
                    )
                break
            available = _available_memory()
            if available < 16 * 1024**3 or time.monotonic() > deadline:
                _docker("stop", "--timeout", "10", name)
                raise RuntimeError(
                    f"{name} reached its memory/time stop gate; available={available}"
                )
            time.sleep(5)
    finally:
        with (root / f"{name}.log").open("w") as stream:
            subprocess.run(["docker", "logs", name], stdout=stream, stderr=stream, check=False)
    print(f"DONE {name}", flush=True)


def _checkpoint_hash(model: Path, shard_records: dict | None = None) -> str:
    if shard_records is None:
        shard_records = {
            path.name: {"bytes": path.stat().st_size, "sha256": _hash_file(path)}
            for path in sorted(model.glob("*.safetensors"))
        }
    index = model / "model.safetensors.index.json"
    return _canonical_hash(
        {
            "config_sha256": _hash_file(model / "config.json"),
            "index_sha256": _hash_file(index) if index.exists() else None,
            "shards": shard_records,
        }
    )


def main() -> None:
    """Run complete controls while restoring any explicitly paused serving container."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spark-root", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--pause-serving-container")
    args = parser.parse_args()
    root = args.experiment_root
    protocol = _protocol_path(root)
    source = args.spark_root / "models/qwen3.8-27b-bf16/huggingface"
    calibration = args.spark_root / "calibration"
    model_mount = ["--mount", f"type=bind,src={source},dst=/source,readonly"]
    cal_mount = ["--mount", f"type=bind,src={calibration},dst=/calibration,readonly"]
    serving_was_running = False
    if args.pause_serving_container:
        serving_was_running = json.loads(_docker("inspect", args.pause_serving_container))[0][
            "State"
        ]["Running"]
        if serving_was_running:
            _docker("stop", "--timeout", "30", args.pause_serving_container)
    try:
        source_identity = None
        # Evaluate the first corrected control early, then keep all GPU work serial.
        for case in ("rtn", "mse", "norm", "h64"):
            name = f"mxwave-corrected-{case}-convert-noswap-20261003"
            conversion_report = root / "measurements" / f"{case}-conversion.json"
            model = root / "checkpoints" / case
            if conversion_report.is_file():
                measurement = json.loads(conversion_report.read_text())
                manifest_path = model / "mxwave-manifest.json"
                if measurement["case"] != case or (
                    measurement["manifest_sha256"] != _hash_file(manifest_path)
                ):
                    raise ValueError("Completed conversion report does not match its checkpoint")
                manifest = json.loads(manifest_path.read_text())
                for filename, record in manifest["shard_integrity"]["per_shard"].items():
                    path = model / filename
                    if path.stat().st_size != record["bytes"] or (
                        _hash_file(path) != record["sha256"]
                    ):
                        raise ValueError(f"Completed shard integrity mismatch: {filename}")
                print(f"REUSE verified completed {case} conversion", flush=True)
            else:
                _job(
                    root,
                    name,
                    model_mount + cal_mount,
                    [
                        "scripts/convert_quantization_control.py",
                        "--case",
                        case,
                        "--model-dir",
                        "/source",
                        "--output-dir",
                        f"/work/checkpoints/{case}",
                        "--calibration",
                        "/calibration/qwen3.8-27b-64x512.safetensors",
                        "--report",
                        f"/work/measurements/{case}-conversion.json",
                        *(["--resume"] if case == "rtn" else []),
                    ],
                    memory="96g",
                )
            identity = json.loads((model / "mxwave-run.json").read_text())
            if source_identity is None:
                source_identity = identity["source"]
                signature = _checkpoint_hash(source, source_identity["shards"])
                _collect(root, "bf16", source, signature)
            elif identity["source"] != source_identity:
                raise ValueError("Source changed between fixed controls")
            manifest = json.loads((model / "mxwave-manifest.json").read_text())
            _collect(
                root, case, model, _checkpoint_hash(model, manifest["shard_integrity"]["per_shard"])
            )
        amd = args.spark_root / "models/qwen3.8-27b-amd-awq-mxfp4/huggingface"
        _collect(root, "amd", amd, _checkpoint_hash(amd))
        _job(
            root,
            "mxwave-corrected-compare-20261003",
            [],
            [
                "scripts/evaluate_quantization_controls.py",
                "compare",
                "--protocol",
                f"/work/{protocol.relative_to(root).as_posix()}",
                "--output-dir",
                "/work/measurements",
                "--output",
                "/work/quality-comparison.json",
            ],
            memory="12g",
        )
        print("COMPLETE corrected controls", flush=True)
    finally:
        if serving_was_running:
            _docker("start", args.pause_serving_container)
            print("RESTORED serving container", flush=True)


def _collect(root: Path, label: str, model: Path, signature: str) -> None:
    _job(
        root,
        f"mxwave-corrected-{label}-measure-20261003",
        [
            "--mount",
            f"type=bind,src={model},dst=/model,readonly",
        ],
        [
            "scripts/evaluate_quantization_controls.py",
            "collect",
            "--model",
            "/model",
            "--label",
            label,
            "--protocol",
            f"/work/{_protocol_path(root).relative_to(root).as_posix()}",
            "--output-dir",
            "/work/measurements",
            "--checkpoint-sha256",
            signature,
        ],
        memory="108g",
    )


if __name__ == "__main__":
    main()
