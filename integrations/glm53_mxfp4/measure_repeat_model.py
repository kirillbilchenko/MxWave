"""One reference load: small policy ablation, activation fixtures and real-input GEMM replays."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from benchmark_full_model import request, write
from check_full_model import rpc
from decode_quality import collect, compare
from measure_quality_model import source_check
from quality_suite import BASE


def main() -> None:
    """Retain normal-repeat evidence before intrusive probes; never qualify or time a candidate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--vocabulary", required=True, type=int)
    args = parser.parse_args()
    path = args.root / "reference-repeat-measurement.json"
    if path.exists():
        raise FileExistsError("Preserve previous repeat evidence")
    report: dict[str, Any] = {
        "status": "running", "started_epoch": time.time(), "policies": [], "captures": [],
        "throughput_benchmark_run": False, "quality_qualification_attempted": False,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limits": [
            "This new diagnostic launch pins cuBLAS workspace before CUDA initialization.",
            "Default/strict policy comparisons share that workspace setting.",
            "Custom kernels may ignore the PyTorch determinism policy.",
            "Snapshots and fixed-input replays are intrusive and cannot qualify normal execution.",
            "No original quality floor or failed frozen result is changed.",
        ],
    }

    def save(phase: str) -> None:
        report.update(phase=phase, updated_epoch=time.time())
        write(path, report)

    try:
        audit = json.loads(Path(
            "/audit/full-model-v1/scale-palette-all-v1/results/screen.json"
        ).read_bytes())
        report["source_before"] = source_check("control", audit)
        save("source_scales_verified")
        report["weights"] = rpc(BASE, "mxwave_checkpoint_check")
        if any(row["status"] != "passed" for row in report["weights"]):
            raise RuntimeError("Selected loaded weights disagree with source")
        report["warmup"] = rpc(BASE, "mxwave_mhc_warmup", [2048])
        report["eager"] = rpc(BASE, "mxwave_set_eager", [True])
        if any(row["status"] != "complete" for row in report["eager"]):
            raise RuntimeError("Actual eager dispatch was not selected")
        protocol = json.loads((args.root / "protocol.json").read_bytes())
        prompts = [protocol.get("quality_prefix_token_ids", []) + ids
                   for ids in protocol["quality"][:2]]
        if len(prompts) != 2 or any(len(p) != 258 for p in prompts):
            raise ValueError("The frozen diagnostic contexts changed")
        for enabled in (False, True):
            name = "strict" if enabled else "default"
            policy: dict[str, Any] = {
                "name": name, "runtime": rpc(BASE, "mxwave_repeat_policy", [enabled]),
                "runs": [], "comparisons": [],
            }
            report["policies"].append(policy)
            save(name + "_warmup")
            request(BASE, "/v1/completions", {
                "model": "mxwave-glm53-pilot", "prompt": prompts[0], "max_tokens": 4,
                "temperature": 0, "seed": 42, "ignore_eos": True,
            })
            for attempt in range(5):
                save(name + f"_repeat_{attempt}")
                current = collect(BASE, prompts, args.root, f"{name}-{attempt}", args.vocabulary)
                policy["runs"].append(current)
                if attempt:
                    policy["comparisons"].append(compare(policy["runs"][0], current, args.root))
                save(name + f"_finished_{attempt}")
            if enabled:
                continue
            # Capture only the first frozen prefill at the first three MoE layers.
            for attempt in range(3):
                label = f"default-{attempt}"
                capture: dict[str, Any] = {"label": label}
                report["captures"].append(capture)
                save("capturing_" + label)
                capture["armed"] = rpc(BASE, "mxwave_repeat_capture", [label, True])
                if any(row["status"] != "armed" for row in capture["armed"]):
                    raise RuntimeError("Both ranks must arm the bounded capture")
                try:
                    capture["response"] = request(BASE, "/v1/completions", {
                        "model": "mxwave-glm53-pilot", "prompt": prompts[0], "max_tokens": 4,
                        "min_tokens": 4, "temperature": 0, "seed": 42, "ignore_eos": True,
                        "return_token_ids": True,
                    })
                finally:
                    capture["workers"] = rpc(BASE, "mxwave_repeat_capture", [label, False])
                if any(row["status"] != "complete" for row in capture["workers"]):
                    raise RuntimeError("Bounded capture is incomplete; preserve fixtures")
                capture["replays"] = rpc(BASE, "mxwave_repeat_replay")
                save("captured_" + label)
        report["policy_restored"] = rpc(BASE, "mxwave_repeat_policy", [False])
        report["source_after"] = source_check("control", audit)
        report.update(status="complete", finished_epoch=time.time())
        save("complete")
    except BaseException as exc:
        report.update(status="failed", error=repr(exc), finished_epoch=time.time())
        save("failed")
        raise


if __name__ == "__main__":
    main()
