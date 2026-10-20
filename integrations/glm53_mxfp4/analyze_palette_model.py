"""Review storage exactness separately from full-model arithmetic and practical tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from benchmark_full_model import write
from decode_quality import compare


def bitwise(reference: dict[str, Any], other: dict[str, Any]) -> bool:
    """Require matching conditioning and hashes for every retained vocabulary distribution."""
    return all(
        a["prompt_sha256"] == b["prompt_sha256"]
        and a["token_ids"] == b["token_ids"]
        and all(
            x["sha256"] == y["sha256"]
            and x["conditioning_token_ids"] == y["conditioning_token_ids"]
            for x, y in zip(a["steps"], b["steps"], strict=True)
        )
        for a, b in zip(reference["contexts"], other["contexts"], strict=True)
    )


def numerical_gates(repeat: dict[str, Any], candidate: dict[str, Any]) -> tuple[bool, bool]:
    """Keep the existing repeat-stability floor even when the candidate matches the first run."""
    stable = bool(
        repeat["all_decode_conditioning_matches"]
        and repeat["mean_decode_kl"] is not None
        and repeat["mean_decode_kl"] <= 1e-5
        and abs(repeat["mean_decode_nll_difference"]) <= 1e-4
    )
    passed = (
        stable
        and candidate["all_decode_conditioning_matches"]
        and (candidate["mean_decode_kl"] <= max(1e-7, 5 * repeat["mean_decode_kl"]))
        and (
            abs(candidate["mean_decode_nll_difference"])
            <= max(1e-5, 5 * abs(repeat["mean_decode_nll_difference"]))
        )
    )
    return stable, bool(passed)


def review(root: Path) -> dict[str, Any]:
    """Keep an unstable repeat control unqualified and never infer accuracy from source hashes."""
    reports = {
        label: json.loads((root / (label + "-decode-quality.json")).read_bytes())
        for label in ("control", "control-repeat", "candidate", "candidate-repeat")
    }
    aa = compare(reports["control"], reports["control-repeat"], root)
    ab = compare(reports["control"], reports["candidate"], root)
    bb = compare(reports["candidate"], reports["candidate-repeat"], root)
    raw_equal = {
        "control_repeat": bitwise(reports["control"], reports["control-repeat"]),
        "candidate": bitwise(reports["control"], reports["candidate"]),
        "candidate_repeat": bitwise(reports["candidate"], reports["candidate-repeat"]),
    }
    measurements = {
        label: json.loads((root / (label + "-measurement.json")).read_bytes())
        for label in ("control", "candidate")
    }
    storage = True
    allocations = {}
    tasks = {}
    for label, report in measurements.items():
        before = report["complete_scale_gpu_source_checks"]
        after = report["gpu_source_checks_after_inference"]
        if len(before) != 2 or len(after) != 2 or {r["tp_rank"] for r in before} != {0, 1}:
            raise ValueError("Missing full-model GPU source checks")
        storage &= report["all_source_checks_passed"] and report["graph_eager_graph_chats_passed"]
        for values in (before, after):
            storage &= all(
                r["status"] == "passed"
                and r["scale_groups"] == 84
                and r["compressed_groups_gpu_source_bytes_exact"]
                for r in values
            )
        allocations[label] = sum(r["allocation"]["cuda_allocated_bytes"] for r in after)
        task = json.loads((root / label / "real-work.json").read_bytes())
        if (
            task["status"] != "complete"
            or len(task["cases"]) != 6
            or {c["name"] for c in task["cases"]}
            != {"json", "constraints", "slugify", "merge_intervals", "retrieval", "tool"}
        ):
            raise ValueError("Practical task scoring is incomplete")
        tasks[label] = [{"name": c["name"], "passed": bool(c["passed"])} for c in task["cases"]]
    stable, numerical = numerical_gates(aa, ab)
    return {
        "status": "complete",
        "scale_storage_lossless": bool(storage),
        "control_repeat_stable": bool(stable),
        "numerical_window_gate_passed": bool(numerical),
        "full_logprob_arrays_bitwise_equal": raw_equal,
        "end_to_end_bitwise_losslessness_demonstrated": bool(storage and all(raw_equal.values())),
        "practical_tasks": tasks,
        "all_practical_tasks_passed": all(c["passed"] for rows in tasks.values() for c in rows),
        "allocated_cuda_bytes_after_inference": allocations,
        "additional_cuda_saving_bytes": allocations["control"] - allocations["candidate"],
        "actual_decode_comparisons": {
            "control_repeat": aa,
            "candidate": ab,
            "candidate_repeat": bb,
        },
        "limits": [
            "Reference is the same already quantized MXFP4 checkpoint and existing codec.",
            "Source storage exactness is distinct from identical GEMM rounding or log-probabilities.",
            "Six authored tasks and sixteen short quality windows are not a standard accuracy benchmark.",
            "This qualification diagnostic does not resolve the component prefill latency regression.",
            "Do not relax the repeat-control stability floor or claim release qualification from these checks.",
        ],
    }


def main() -> None:
    """Write a hash-bound review after both engines finish and code sandboxes are scored."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    value = review(args.root)
    value["analysis_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    value["inputs"] = {
        str(p.relative_to(args.root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in args.root.rglob("*.json")
        if p.name != "palette-model-review.json"
    }
    write(args.root / "palette-model-review.json", value)


if __name__ == "__main__":
    main()
