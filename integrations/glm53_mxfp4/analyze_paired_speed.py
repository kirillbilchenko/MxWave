"""Recompute paired serving throughput and client latency from every retained request."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from benchmark_full_model import write


def timing_metrics(measured: dict[str, Any], batch_size: int) -> dict[str, float | None]:
    """Reject incomplete streams and recompute latency/TPS without using profile durations."""
    requests = measured["requests"]
    if len(requests) != batch_size or measured["wall_seconds"] <= 0:
        raise ValueError("Incomplete timing group")
    firsts, latencies, decode_rates, middle = [], [], [], []
    for row in requests:
        arrivals = row["arrivals"]
        if (len(row["token_ids"]) != 64 or not arrivals
                or sum(a["tokens"] for a in arrivals) != 64
                or any(a["tokens"] <= 0 for a in arrivals)):
            raise ValueError("Require every one of the 64 generated tokens")
        times = [a["seconds"] for a in arrivals]
        if (times != sorted(times) or times[0] <= 0 or times[-1] <= times[0]
                or row["elapsed_seconds"] < times[-1]
                or row["elapsed_seconds"] > measured["wall_seconds"]):
            raise ValueError("Invalid stream arrival timing")
        firsts.append(times[0])
        latencies.append(row["elapsed_seconds"])
        decode_rates.append((64 - arrivals[0]["tokens"]) / (times[-1] - times[0]))
        if len(arrivals) == 64 and all(a["tokens"] == 1 for a in arrivals):
            middle.append(statistics.median(times[i] - times[i - 1] for i in range(9, 56)))
    tps = batch_size * 64 / measured["wall_seconds"]
    ttft = statistics.median(firsts)
    if (not math.isclose(tps, measured["aggregate_tokens_per_second"], rel_tol=1e-12)
            or not math.isclose(ttft, measured["median_ttft_seconds"], rel_tol=1e-12)):
        raise ValueError("Stored group summary differs from raw arrivals")
    return {"aggregate_tokens_per_second": tps, "median_ttft_seconds": ttft,
            "median_request_seconds": statistics.median(latencies),
            "median_request_decode_tokens_per_second": statistics.median(decode_rates),
            "middle_single_token_streams": float(len(middle)),
            "median_middle_interarrival_seconds": statistics.median(middle) if middle else None}


def review(root: Path) -> dict[str, Any]:
    """Keep all six workload cases, output-ID differences and historical qualification failures."""
    plan_path = root / "paired-speed-plan.json"
    plan = json.loads(plan_path.read_bytes())
    audit = json.loads((root / "source-audit.json").read_bytes())
    arms: dict[str, Any] = {}
    medians: dict[str, Any] = {}
    allocations = {}
    source_groups = 0
    for label in ("control", "candidate"):
        arm = json.loads((root / (label + "-speed.json")).read_bytes())
        if (arm["status"] != "complete" or arm["qualification_attempted"]
                or arm["plan_sha256"] != hashlib.sha256(plan_path.read_bytes()).hexdigest()
                or len(arm["trials"]) != 30):
            raise ValueError("Incomplete or changed speed measurement")
        for phase in ("source_before", "source_after"):
            if len(arm[phase]) != 2:
                raise ValueError("Missing source rank")
            for row in arm[phase]:
                expected = Counter((tuple(c["shape"]), c["raw_sha256"]) for c in audit["cases"]
                                   if c["rank"] == row["tp_rank"])
                actual = Counter((tuple(c["shape"]), c["sha256"])
                                 for c in row["source_scale_hashes"])
                if (row["status"] != "passed" or expected != actual
                        or row["candidate"] != (label == "candidate")):
                    raise ValueError("Source integrity failed")
                source_groups += sum(actual.values())
        per_case = {}
        for batch_size in (1, 4, 8):
            for length in (256, 2048):
                trials = [r for r in arm["trials"]
                          if (r["batch"], r["prompt_tokens"]) == (batch_size, length)]
                if sorted(r["trial"] for r in trials) != list(range(5)):
                    raise ValueError("Missing or repeated timing trial")
                values = [timing_metrics(r["measurement"], batch_size) for r in trials]
                key = f"C{batch_size}-P{length}"
                per_case[key] = {}
                for name in values[0]:
                    available = [v[name] for v in values if v[name] is not None]
                    per_case[key][name] = {
                        "median": statistics.median(available) if available else None,
                        "min": min(available) if available else None,
                        "max": max(available) if available else None,
                        "available_trials": len(available)}
        medians[label] = per_case
        arms[label] = arm
        allocations[label] = sum(row["allocation"]["cuda_allocated_bytes"]
                                 for row in arm["source_after"])
    comparisons = {}
    for key in medians["control"]:
        left, right = medians["control"][key], medians["candidate"][key]
        changes = {name: (right[name]["median"] / left[name]["median"] - 1) * 100
                   for name in left if left[name]["median"] not in (None, 0)
                   and right[name]["median"] is not None}
        comparisons[key] = {"candidate_change_percent": changes}
    ids = []
    for left in arms["control"]["trials"]:
        right = next(r for r in arms["candidate"]["trials"]
                     if (r["trial"], r["batch"], r["prompt_tokens"])
                     == (left["trial"], left["batch"], left["prompt_tokens"]))
        matches = [a["token_ids"] == b["token_ids"] for a, b in zip(
            left["measurement"]["requests"], right["measurement"]["requests"], strict=True)]
        ids.append({"trial": left["trial"], "batch": left["batch"],
                    "prompt_tokens": left["prompt_tokens"], "matching_sequences": sum(matches),
                    "sequences": len(matches)})
    return {"status": "complete", "quality_qualified": False, "qualification_attempted": False,
        "previous_failed_qualifications_preserved": True,
        "source_scale_groups_verified": source_groups, "trials_per_arm": 30,
        "timed_output_tokens_per_arm": 8320, "workloads": medians, "comparisons": comparisons,
        "generated_sequence_agreement": ids, "allocated_cuda_bytes_after_inference": allocations,
        "combined_cuda_saving_bytes": allocations["control"] - allocations["candidate"],
        "profile_pair_complete": all(arms[k]["profile"]["status"] == "complete" for k in arms),
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "limits": plan["limits"]}


def main() -> None:
    """Write a review without promoting the experimental model or its unresolved numerical gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = review(args.root)
    result["analysis_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write(args.root / "quality-model-review.json", result)


if __name__ == "__main__":
    main()
