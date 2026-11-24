"""Measure practical repeat variability while preserving the preceding strict failure."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Any

from analyze_quality_model import task_scores
from benchmark_full_model import write


def summarize_rounds(rounds: list[list[dict[str, Any]]]) -> dict[str, Any]:
    """Retain every round and both directions of every within-arm pass-loss comparison."""
    if len(rounds) != 5 or any(len(rows) != 34 for rows in rounds):
        raise ValueError("Exactly five complete 34-case rounds are required")
    keys = [{row["name"] for row in rows} for rows in rounds]
    if any(names != keys[0] or len(names) != 34 for names in keys):
        raise ValueError("Practical repeat case sets differ")
    indexed = [{row["name"]: row for row in rows} for rows in rounds]
    differences = []
    for a, b in itertools.combinations(range(5), 2):
        differences.append({
            "round_a": a, "round_b": b,
            "a_pass_b_fail": [name for name in sorted(keys[0])
                              if indexed[a][name]["passed"] and not indexed[b][name]["passed"]],
            "b_pass_a_fail": [name for name in sorted(keys[0])
                              if indexed[b][name]["passed"] and not indexed[a][name]["passed"]],
            "different_answer_hashes": [name for name in sorted(keys[0])
                                        if indexed[a][name]["answer_sha256"]
                                        != indexed[b][name]["answer_sha256"]],
        })
    return {
        "strict_passes_by_round": [sum(r["passed"] for r in rows) for rows in rounds],
        "completed_inference_by_round": [sum(r["inference_completed"] for r in rows)
                                         for rows in rounds],
        "within_arm_pairs": differences,
        "per_case": [{
            "name": name, "kind": indexed[0][name]["kind"],
            "strict_passes": sum(row[name]["passed"] for row in indexed),
            "distinct_answer_hashes": len({row[name]["answer_sha256"] for row in indexed}),
            "code_functional_passes": sum(row[name]["semantic_passed"] for row in indexed)
            if indexed[0][name]["kind"] == "code" else None,
        } for name in sorted(keys[0])],
    }


def completed_scores(root: Path, label: str, protocol: dict[str, Any]) -> list[dict[str, Any]]:
    """Derive API completion from raw replies when a frozen scorer predates that field."""
    rows = task_scores(root, label, protocol)
    raw = json.loads((root / label / "real-work.json").read_bytes())
    replies = {row["name"]: isinstance(row.get("response"), dict) for row in raw["cases"]}
    if len(replies) != len(rows) or set(replies) != {row["name"] for row in rows}:
        raise ValueError("Raw response and scored task sets differ")
    for row in rows:
        completed = replies[row["name"]]
        if "inference_completed" in row and row["inference_completed"] != completed:
            raise ValueError("Scored completion flag disagrees with the raw API reply")
        row["inference_completed"] = completed
    return rows


def review(root: Path) -> dict[str, Any]:
    """Report diagnostic observations without recalibrating a numerical or practical gate."""
    plan_raw = (root / "task-repeat-plan.json").read_bytes()
    plan = json.loads(plan_raw)
    raw = (root / "quality-protocol.json").read_bytes()
    protocol = json.loads(raw)
    if plan["rounds_per_arm"] != 5 or plan["qualification_attempted"]:
        raise ValueError("Keep the predeclared diagnostic scope")
    if hashlib.sha256(raw).hexdigest() != plan["quality_protocol_sha256"]:
        raise ValueError("Original practical tasks changed")
    audit_raw = (root / "task-repeat-source-audit.json").read_bytes()
    audit = json.loads(audit_raw)
    if hashlib.sha256(audit_raw).hexdigest() != plan["source_audit_sha256"]:
        raise ValueError("Independent checkpoint scale audit changed")
    if audit["status"] != "passed" or len(audit["cases"]) != 168:
        raise ValueError("Incomplete independent source audit")
    expected = {
        rank: Counter((tuple(c["shape"]), c["raw_sha256"])
                      for c in audit["cases"] if c["rank"] == rank)
        for rank in (0, 1)
    }
    arms, measurements = {}, {}
    for label in ("control", "candidate"):
        measurement = json.loads((root / (label + "-quality-measurement.json")).read_bytes())
        if (
            measurement["status"] != "collected_pending_code_sandbox"
            or measurement["predeclared_rounds"] != 5
            or measurement["quality_protocol_sha256"] != plan["quality_protocol_sha256"]
            or measurement["plan_sha256"] != hashlib.sha256(plan_raw).hexdigest()
            or len(measurement["rounds"]) != 5
        ):
            raise ValueError("Incomplete or changed practical-repeat collection")
        for phase in ("source_before", "source_after"):
            source = measurement[phase]
            if {row["tp_rank"] for row in source} != {0, 1} or len(source) != 2:
                raise ValueError("Missing TP2 source checks")
            if not all(row["status"] == "passed" and row["scale_groups"] == 84
                       and row["stable_dispatch"]["installed"] for row in source):
                raise ValueError("Loaded source or stable-dispatch check failed")
            for row in source:
                actual = Counter((tuple(c["shape"]), c["sha256"])
                                 for c in row["source_scale_hashes"])
                if actual != expected[row["tp_rank"]] or row["candidate"] != (label == "candidate"):
                    raise ValueError("Source scale bytes or declared arm changed")
                if label == "candidate" and row["native_sha256"] != (
                    "9f5d96dcb0762d7e5ed5436ca296f70d590e73797d40522e4d8c7d6f3902f5f0"
                ):
                    raise ValueError("Require the qualified v9 U8 library")
        rounds = [completed_scores(root, f"{label}-r{i}", protocol) for i in range(5)]
        arms[label] = {**summarize_rounds(rounds), "scores_by_round": rounds}
        measurements[label] = measurement
    return {
        "status": "complete", "analysis_version": 2,
        "experiment_type": "practical_repeat_diagnostic",
        "qualification_attempted": False, "quality_qualified": False,
        "prior_experiment": "stablequalityv1", "prior_strict_verdict_preserved": True,
        "rounds_per_arm": 5, "tasks_per_round": 34, "arms": arms,
        "plan_sha256": hashlib.sha256(plan_raw).hexdigest(),
        "quality_protocol_sha256": plan["quality_protocol_sha256"],
        "allocated_cuda_bytes_after_inference": {
            label: sum(r["allocation"]["cuda_allocated_bytes"] for r in m["source_after"])
            for label, m in measurements.items()
        },
        "limits": [
            "All five rounds are retained; no best round is selected.",
            "Tasks and sampler seed are unchanged; temporal repetitions are not independent samples.",
            "No throughput, new perplexity, numerical-floor or release qualification claim.",
            "Baseline pass losses would demonstrate task instability, not prove harmless adapter drift.",
            "The earlier strict failure cannot be converted into a pass by this diagnostic.",
        ],
    }


def main() -> None:
    """Write a complete diagnostic review after isolated code sandboxes are scored."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = review(args.root)
    result["analysis_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["inputs"] = {
        str(p.relative_to(args.root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in args.root.rglob("*.json") if p.name != "quality-model-review.json"
    }
    write(args.root / "quality-model-review.json", result)


if __name__ == "__main__":
    main()
