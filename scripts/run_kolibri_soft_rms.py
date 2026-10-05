"""Measure one fixed softened-RMS recipe without another full checkpoint on disk."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

from run_kolibri_release import IMAGE, SERVING_CONTAINER, _docker_state


def main() -> None:
    """Probe the compact loader, convert all observed experts, and run the frozen screen."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    logs = root / "qualification-logs"
    logs.mkdir(exist_ok=True)
    started = time.monotonic()

    def run(name, script, arguments, memory, cpus, *, inference=False, timeout=5400):
        command = [
            "docker",
            "run",
            "--rm",
            "--name",
            name,
            "--gpus",
            "all",
            "--memory",
            memory,
            "--cpus",
            str(cpus),
        ]
        if not inference:
            command.extend(["--memory-swap", "16g"])
        command.extend(
            [
                "-e",
                "PYTHONPATH="
                + ("/work/vendor:" if inference else "")
                + "/work/runtime:/work/runtime/scripts",
                "-e",
                "VLLM_USE_DEEP_GEMM=0",
                "-e",
                "VLLM_USE_DEEP_GEMM_E8M0=0",
                "-v",
                f"{root}:/work",
                "--entrypoint",
                "python3",
                IMAGE,
                "-u",
                f"/work/runtime/scripts/{script}",
                *arguments,
            ]
        )
        print(f"Starting {name}", flush=True)
        with (logs / f"{name}.log").open("w") as stream:
            subprocess.run(
                command, check=True, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout
            )

    def convert(smoke):
        trial = "smoke" if smoke else "strength025"
        if (root / f"soft-rms-trials/{trial}/conversion-result.json").is_file():
            return
        run(
            f"mxwave-kolibri1-soft-rms-{trial}",
            "retune_kolibri_soft_rms.py",
            [
                "--source",
                "/work/smoke-source" if smoke else "/work/source-bf16",
                "--baseline",
                "/work/smoke-mixed" if smoke else "/work/checkpoint-mixed",
                "--statistics",
                "/work/activation-calibration/"
                + ("smoke-rms.safetensors" if smoke else "routed-rms.safetensors"),
                "--counts",
                "/work/activation-calibration/"
                + ("smoke-rms.json" if smoke else "routed-rms.json"),
                "--output",
                f"/work/soft-rms-trials/{trial}",
            ],
            "12g",
            4,
        )

    convert(True)
    if not (root / "soft-rms-trials/smoke/smoke-result.json").is_file():
        run(
            "mxwave-kolibri1-soft-rms-loader-probe",
            "evaluate_kolibri_soft_rms.py",
            [
                "smoke",
                "--specification",
                "/work/soft-rms-trials/smoke/trial.json",
            ],
            "12g",
            4,
            inference=True,
            timeout=600,
        )
    convert(False)
    trial = root / "soft-rms-trials/strength025"
    state = _docker_state(SERVING_CONTAINER)
    restore = bool(state["Running"] and not state["Paused"])
    name = "mxwave-kolibri1-soft-rms-evaluate"
    try:
        if restore:
            subprocess.run(["docker", "stop", "--time", "30", SERVING_CONTAINER], check=True)
        run(
            name,
            "evaluate_kolibri_soft_rms.py",
            [
                "collect",
                "--specification",
                "/work/soft-rms-trials/strength025/trial.json",
            ],
            "105g",
            12,
            inference=True,
            timeout=1800,
        )
        measurement = json.loads((trial / "evaluation/measurements/mse.json").read_text())
        quality = json.loads((trial / "evaluation/quality-comparison.json").read_text())
        baseline = json.loads((root / "mixed-result.json").read_text())
        conversion = json.loads((trial / "conversion-result.json").read_text())
        result = {
            "status": quality["status"],
            "seconds": time.monotonic() - started,
            "mean_next_token_kl": quality["mean_next_token_kl"],
            "unweighted_mean_next_token_kl": baseline["mean_next_token_kl"],
            "kl_reduction_fraction": 1
            - quality["mean_next_token_kl"] / baseline["mean_next_token_kl"],
            "candidate_ppl": quality["domains"]["all"]["candidate_ppl"],
            "ppl_ratio": quality["domains"]["all"]["ppl_ratio"],
            "behavior_passed": all(row["passed"] for row in measurement["behavior"]),
            "median_output_tokens_per_second": {
                str(c): statistics.median(
                    row["aggregate_output_tokens_per_second"]
                    for row in measurement["throughput"]
                    if row["concurrency"] == c
                )
                for c in (1, 4)
            },
            "trial_specification_sha256": measurement["trial_specification_sha256"],
            "recipe": measurement["recipe"],
            "conversion": conversion,
            "standalone_checkpoint": False,
            "published": False,
        }
        (root / "soft-rms-result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    finally:
        subprocess.run(
            ["docker", "stop", "--time", "30", name],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if restore:
            subprocess.run(["docker", "start", SERVING_CONTAINER], check=True)


if __name__ == "__main__":
    main()
