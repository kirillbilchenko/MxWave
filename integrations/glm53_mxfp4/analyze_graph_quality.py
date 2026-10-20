"""Recompute every prospective quality gate for the matched graph-four model pair."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from analyze_quality_model import heldout_values, task_scores
from benchmark_full_model import write
from decode_quality import compare
from measure_graph_quality import numerical_passed, validate_protocol


def review(root: Path) -> dict[str, Any]:
    """Require all fixed repeats, exact source coverage and the unchanged strict task contract."""
    protocol = json.loads((root / "quality-protocol.json").read_bytes())
    plan = json.loads((root / "graph-quality-plan.json").read_bytes())
    validate_protocol(plan, protocol)
    proto_sha = hashlib.sha256((root / "quality-protocol.json").read_bytes()).hexdigest()
    plan_sha = hashlib.sha256((root / "graph-quality-plan.json").read_bytes()).hexdigest()
    measurements = {label: json.loads((root / (label + "-quality-measurement.json")).read_bytes())
                    for label in ("control", "candidate")}
    comparisons = []
    storage = True
    source: dict[int, Any] = {}
    allocations, losses = {}, {}
    for label, arm in measurements.items():
        if (arm["status"] != "collected_pending_code_sandbox"
                or arm["quality_protocol_sha256"] != proto_sha or arm["plan_sha256"] != plan_sha
                or len(arm["numerical"]) != 5 or len(arm["heldout"]) != 3):
            raise ValueError("Incomplete paired quality evidence")
        for phase in ("source_before", "source_after"):
            if {r["tp_rank"] for r in arm[phase]} != {0, 1}:
                raise ValueError("Incomplete source rank coverage")
            for row in arm[phase]:
                storage &= row["status"] == "passed" and row["scale_groups"] == 84
                counts = Counter((tuple(r["shape"]), r["sha256"])
                                 for r in row["source_scale_hashes"])
                if label == "control" and phase == "source_before":
                    source[row["tp_rank"]] = counts
                else:
                    storage &= source[row["tp_rank"]] == counts
        storage &= all(row["status"] == "passed" for row in arm["loaded_weight_checks"])
        allocations[label] = sum(row["allocation"]["cuda_allocated_bytes"]
                                 for row in arm["source_after"])
        reports = []
        for index, leg in enumerate(arm["numerical"]):
            decoded = json.loads((root / f"graph-{label}-r{index}-decode-quality.json").read_bytes())
            if decoded != leg["decode"] or not all(r["proof"]["passed"]
                                                   for r in leg["batch_proofs"]):
                raise ValueError("Retained numerical or controlled admission evidence changed")
            reports.append(decoded)
            if index:
                comparisons.append({"kind": label + "_repeat", "index": index,
                                    **compare(reports[0], decoded, root)})
            if label == "candidate":
                reference = json.loads(
                    (root / f"graph-control-r{index}-decode-quality.json").read_bytes())
                comparisons.append({"kind": "paired_cross_arm", "index": index,
                                    **compare(reference, decoded, root)})
        losses[label] = [heldout_values(root, f"{label}-r{i}", protocol) for i in range(3)]
    numerical = all(numerical_passed(row, protocol["strict_control_stability_gate"])
                    for row in comparisons)
    loss_deltas = [losses["candidate"][i]["mean_nll"] - losses["control"][i]["mean_nll"]
                   for i in range(3)]
    repeat_deltas = {label: [row["mean_nll"] - rows[0]["mean_nll"] for row in rows[1:]]
                     for label, rows in losses.items()}
    loss_passed = (all(abs(x) <= 1e-5 for x in loss_deltas)
                   and all(abs(x) <= 1e-4 for rows in repeat_deltas.values() for x in rows))
    tasks = {label: task_scores(root, label, protocol) for label in measurements}
    candidate = {row["name"]: row for row in tasks["candidate"]}
    lost = [row["name"] for row in tasks["control"]
            if row["passed"] and not candidate[row["name"]]["passed"]]
    gained = [row["name"] for row in tasks["control"]
              if not row["passed"] and candidate[row["name"]]["passed"]]
    completed = all(row["inference_completed"] for rows in tasks.values() for row in rows)
    timing = []
    for count in (1, 4):
        arms = {label: next(row for row in arm["timing"] if row["batch"] == count)
                for label, arm in measurements.items()}
        for value in arms.values():
            if len(value["trials"]) != 5:
                raise ValueError("Every timing trial must be retained")
        reference, experiment = (arms[label] for label in ("control", "candidate"))
        timing.append({"batch": count, "prompt_tokens": 256, "arms": arms,
            "throughput_change_percent": 100 * (experiment["median_aggregate_tokens_per_second"]
                / reference["median_aggregate_tokens_per_second"] - 1),
            "ttft_change_percent": 100 * (experiment["median_ttft_seconds"]
                / reference["median_ttft_seconds"] - 1)})
    return {"status": "complete", "quality_qualified": bool(
        storage and numerical and loss_passed and completed and not lost),
        "scale_storage_lossless": bool(storage), "numerical_gate_passed": numerical,
        "numerical_comparisons": comparisons, "heldout": {"measurements": losses,
            "paired_nll_deltas": loss_deltas, "repeat_nll_deltas": repeat_deltas,
            "absolute_candidate_nll_limit": 1e-5, "gate_passed": loss_passed},
        "practical": {"scores": tasks, "strict_totals": {
            label: sum(row["passed"] for row in rows) for label, rows in tasks.items()},
            "total_cases_per_arm": 110, "reference_passed_candidate_failed": lost,
            "reference_failed_candidate_passed": gained,
            "all_inference_requests_completed": completed, "no_reference_pass_lost": not lost},
        "allocated_cuda_bytes_after_inference": allocations,
        "combined_cuda_saving_bytes": allocations["control"] - allocations["candidate"],
        "timing": timing, "quality_protocol_sha256": proto_sha, "plan_sha256": plan_sha,
        "end_to_end_bitwise_losslessness_demonstrated": False,
        "previous_failed_qualifications_preserved": True,
        "limits": protocol["limits"] + [
            "Batch-one practical tasks and batch-four decode are tested; not every batch shape.",
            "Two serial model loads cannot remove all time/order effects from performance.",
            "Exact storage reconstruction does not imply identical floating-point arithmetic."]}


def main() -> None:
    """Retain a failed quality gate as a result, without choosing different repeats or floors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = review(args.root)
    result["analysis_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write(args.root / "quality-model-review.json", result)


if __name__ == "__main__":
    main()
