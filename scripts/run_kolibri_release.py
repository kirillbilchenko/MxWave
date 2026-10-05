"""Run the frozen Kolibri quality screen on Spark when conversion has completed."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

IMAGE = "sha256:785fd756b4ae1cacda7612cb859dbf227330e8ced9a6fdea42e9379128bebce0"
SERVING_CONTAINER = "local-spark-qwen3-8-27b-mxwave-ollama-model-1"


def _docker_state(name: str) -> dict[str, Any]:
    result = subprocess.run(
        ["docker", "inspect", "--format", "{{json .State}}", name],
        check=True, text=True, capture_output=True,
    )
    return json.loads(result.stdout)


def main() -> None:
    """Wait for valid artifacts, measure the two models sequentially, and restore serving."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    started = time.time()
    while not (root / "conversion-result.json").exists():
        state = _docker_state("mxwave-kolibri1-convert")
        if not state["Running"]:
            raise RuntimeError(f"Conversion stopped without a completion marker: {state}")
        print("Waiting for full conversion", flush=True)
        time.sleep(30)
    for filename in ("conversion-result.json", "smoke-result.json", "fp8-smoke-result.json"):
        record = json.loads((root / filename).read_text())
        if record.get("status") not in ("complete", "passed"):
            raise RuntimeError(f"Prerequisite failed: {filename}")
    if not (root / "reference-fp8-download.json").is_file():
        raise RuntimeError("The official FP8 reference download is not complete")
    if not (root / "validation/protocol.json").is_file():
        raise RuntimeError("The quality protocol has not been frozen")
    logs = root / "qualification-logs"
    logs.mkdir(exist_ok=True)
    if not (root / "payload-verification.json").exists():
        with (logs / "payload-verification.log").open("a") as stream:
            subprocess.run([
                "docker", "run", "--rm", "--memory", "4g", "--cpus", "4",
                "-e", "PYTHONPATH=/work/runtime", "-v", f"{root}:/work",
                "--entrypoint", "python3", IMAGE,
                "-u", "/work/runtime/scripts/verify_kolibri_checkpoint.py",
            ], check=True, stdout=stream, stderr=subprocess.STDOUT)
    state = _docker_state(SERVING_CONTAINER)
    restore_serving = bool(state["Running"] and not state["Paused"])
    try:
        if restore_serving:
            subprocess.run(["docker", "stop", "--time", "30", SERVING_CONTAINER], check=True)
        for label in ("fp8", "mse"):
            if (root / f"measurements/{label}.json").exists():
                continue
            command = [
                "docker", "run", "--rm", "--name", f"mxwave-kolibri1-evaluate-{label}",
                "--gpus", "all", "--memory", "105g", "--cpus", "12",
                "-e", "PYTHONPATH=/work/vendor:/work/runtime:/work/runtime/scripts",
                "-e", "VLLM_USE_DEEP_GEMM=0", "-e", "VLLM_USE_DEEP_GEMM_E8M0=0",
                "-v", f"{root}:/work", "--entrypoint", "python3", IMAGE,
                "-u", "/work/runtime/scripts/evaluate_kolibri_release.py", "collect",
                "--label", label,
            ]
            print(f"Starting isolated {label} measurements", flush=True)
            with (logs / f"{label}.log").open("a") as stream:
                subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
        with (logs / "comparison.log").open("a") as stream:
            subprocess.run([
                "docker", "run", "--rm", "--memory", "4g", "--cpus", "4",
                "-e", "PYTHONPATH=/work/vendor:/work/runtime/scripts", "-v", f"{root}:/work",
                "--entrypoint", "python3", IMAGE,
                "-u", "/work/runtime/scripts/evaluate_kolibri_release.py", "compare",
            ], check=True, stdout=stream, stderr=subprocess.STDOUT)
        quality = json.loads((root / "quality-comparison.json").read_text())
        result = {"status": quality["status"], "seconds_including_wait": time.time() - started,
                  "quality_comparison": str(root / "quality-comparison.json"),
                  "published": False}
        (root / "qualification-result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    finally:
        if restore_serving:
            subprocess.run(["docker", "start", SERVING_CONTAINER], check=True)


if __name__ == "__main__":
    main()
