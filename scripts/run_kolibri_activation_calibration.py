"""Collect full routed Kolibri calibration in isolation and restore normal Spark serving."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from run_kolibri_release import IMAGE, SERVING_CONTAINER, _docker_state


def main() -> None:
    """Run the validated observer against the official FP8 teacher, without publishing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if json.loads((root / "activation-calibration/smoke-result.json").read_text())["status"] != "passed":
        raise RuntimeError("Routed activation observer must pass its parity probe first")
    if not (root / "activation-calibration/protocol.json").is_file():
        raise RuntimeError("Separate activation calibration prompts have not been frozen")
    state = _docker_state(SERVING_CONTAINER)
    restore = bool(state["Running"] and not state["Paused"])
    name, active = "mxwave-kolibri1-activation-calibrate", False
    try:
        if restore:
            subprocess.run(["docker", "stop", "--time", "30", SERVING_CONTAINER], check=True)
        active = True
        with (root / "qualification-logs/activation-calibrate.log").open("w") as log:
            subprocess.run([
                "docker", "run", "--rm", "--name", name, "--gpus", "all",
                "--memory", "105g", "--cpus", "12",
                "-e", "PYTHONPATH=/work/vendor:/work/runtime:/work/runtime/scripts:/work",
                "-e", "VLLM_USE_DEEP_GEMM=0", "-e", "VLLM_USE_DEEP_GEMM_E8M0=0",
                "-e", "VLLM_ALLOW_INSECURE_SERIALIZATION=1",
                "-v", f"{root}:/work", "--entrypoint", "python3", IMAGE,
                "-u", "/work/calibrate_kolibri_activations.py", "collect",
            ], check=True, stdout=log, stderr=subprocess.STDOUT, timeout=2400)
        active = False
        result = json.loads((root / "activation-calibration/result.json").read_text())
        print(json.dumps(result), flush=True)
    finally:
        if active:
            subprocess.run(["docker", "stop", "--time", "30", name], check=False)
        if restore:
            subprocess.run(["docker", "start", SERVING_CONTAINER], check=True)


if __name__ == "__main__":
    main()
