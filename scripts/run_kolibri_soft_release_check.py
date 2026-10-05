"""Check the exported soft-RMS checkpoint with the stock mixed-format vLLM loader."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
import subprocess
import time
from pathlib import Path

from run_kolibri_release import IMAGE, SERVING_CONTAINER, _docker_state


def main() -> None:
    """Wait for export, collect the unchanged full screen, and restore normal serving."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    marker = root / "checkpoint-soft-rms-export.json"
    deadline = time.monotonic() + 3600
    while not marker.exists():
        if (
            time.monotonic() > deadline
            or not _docker_state("mxwave-kolibri1-soft-rms-export")["Running"]
        ):
            raise RuntimeError("Soft-RMS export did not finish successfully")
        time.sleep(10)
    export = json.loads(marker.read_text())
    if (
        export["status"] != "complete"
        or export["verified_payloads"] != 116303
        or export["shards"] != 32
    ):
        raise ValueError("Incomplete standalone soft-RMS export")
    manifest = root / "checkpoint-soft-rms/mxwave-manifest.json"
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != export["manifest_sha256"]:
        raise ValueError("Exported soft-RMS manifest changed")
    view = root / "soft-rms-standalone-check"
    if (view / "mixed-evaluation/measurements/mse.json").exists():
        raise FileExistsError("Refusing to overwrite a completed standalone qualification")
    (view / "validation").mkdir(parents=True, exist_ok=True)
    (view / "measurements").mkdir(exist_ok=True)
    for name, target in (
        ("checkpoint-mixed", "checkpoint-soft-rms"),
        ("source-bf16", "source-bf16"),
        ("reference-fp8", "reference-fp8"),
        ("smoke-mixed", "smoke-mixed"),
    ):
        link = view / name
        if link.is_symlink() and link.readlink().is_absolute():
            if link.resolve() != (root / target).resolve():
                raise ValueError(f"Unexpected existing checkpoint link: {link}")
            link.unlink()
        if not link.exists():
            link.symlink_to(Path("..") / target, target_is_directory=True)
    for name in ("validation/protocol.json", "mixed-smoke-result.json"):
        shutil.copyfile(root / name, view / name)
    for suffix in ("json", "safetensors"):
        link = view / f"measurements/fp8.{suffix}"
        if link.is_symlink() and link.readlink().is_absolute():
            if link.resolve() != (root / f"measurements/fp8.{suffix}").resolve():
                raise ValueError(f"Unexpected existing measurement link: {link}")
            link.unlink()
        if not link.exists():
            link.symlink_to(Path("../../measurements") / f"fp8.{suffix}")
    state = _docker_state(SERVING_CONTAINER)
    restore = bool(state["Running"] and not state["Paused"])
    name = "mxwave-kolibri1-soft-rms-standalone-evaluate"
    active = False
    started = time.monotonic()
    try:
        if restore:
            subprocess.run(["docker", "stop", "--timeout", "30", SERVING_CONTAINER], check=True)
        for script, arguments, inference in (
            ("release_kolibri_file_cache.py", ["--root", "/work"], False),
            (
                "evaluate_kolibri_mixed.py",
                ["collect", "--root", "/work/soft-rms-standalone-check"],
                True,
            ),
        ):
            command = [
                "docker",
                "run",
                "--rm",
                "--name",
                name if inference else name + "-cache",
                "--memory",
                "105g" if inference else "512m",
                "--cpus",
                "12" if inference else "1",
            ]
            if inference:
                command += ["--gpus", "all"]
            command += [
                "-e",
                "PYTHONPATH=/work/vendor:/work/runtime:/work/runtime/scripts",
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
            active = inference
            with (
                root / "qualification-logs" / f"{name}-{'quality' if inference else 'cache'}.log"
            ).open("w") as stream:
                subprocess.run(
                    command, check=True, stdout=stream, stderr=subprocess.STDOUT, timeout=1800
                )
            active = False
        directory = view / "mixed-evaluation"
        measurement = json.loads((directory / "measurements/mse.json").read_text())
        quality = json.loads((directory / "quality-comparison.json").read_text())
        if measurement["checkpoint_manifest_sha256"] != export["manifest_sha256"]:
            raise ValueError("Standalone measurements do not identify the exported checkpoint")
        result = {
            "status": quality["status"],
            "seconds": time.monotonic() - started,
            "checkpoint_manifest_sha256": export["manifest_sha256"],
            "trial_specification_sha256": export["trial_specification_sha256"],
            "candidate_ppl": quality["domains"]["all"]["candidate_ppl"],
            "ppl_ratio": quality["domains"]["all"]["ppl_ratio"],
            "mean_next_token_kl": quality["mean_next_token_kl"],
            "behavior_passed": all(row["passed"] for row in measurement["behavior"]),
            "median_output_tokens_per_second": {
                str(c): statistics.median(
                    row["aggregate_output_tokens_per_second"]
                    for row in measurement["throughput"]
                    if row["concurrency"] == c
                )
                for c in (1, 4)
            },
            "source": "Stock vLLM mixed-format loader; exact exported soft-RMS payloads",
            "published": False,
        }
        (root / "soft-rms-release-result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    finally:
        if active:
            subprocess.run(["docker", "stop", "--timeout", "30", name], check=False)
        if restore:
            subprocess.run(["docker", "start", SERVING_CONTAINER], check=True)


if __name__ == "__main__":
    main()
