"""Measure one arm of the prospective matched graph-four quality protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

from benchmark_full_model import batch, request, write
from check_full_model import SANITY, assess_chat, rpc
from decode_quality import compare
from measure_cohort_model_dispatch import BASE, cohort_request, quality_leg, snapshot
from measure_quality_model import source_check
from quality_suite import heldout, tasks


def numerical_passed(value: dict[str, Any], gate: dict[str, float]) -> bool:
    """Apply the original limits to all real-KV decode steps and their exact conditioning."""
    return bool(value["expected_decode_steps"] == value["decode_steps"] == 48
                and value["all_decode_conditioning_matches"] and value["all_generated_ids_match"]
                and value["mean_decode_kl"] is not None
                and value["mean_decode_kl"] <= gate["mean_kl"]
                and abs(value["mean_decode_nll_difference"]) <= gate["absolute_mean_nll"])


def validate_protocol(plan: dict[str, Any], suite: dict[str, Any]) -> None:
    """Reject an incomplete or relaxed prospective protocol before sending inference requests."""
    if (plan["capture_sizes"] != [1, 4] or plan["arms_in_order"] != ["control", "candidate"]
            or plan["numerical_repeats"] != 5 or plan["heldout_repeats"] != 3
            or plan["admission_grace_seconds"] != 1.0 or plan["timing_trials"] != 5
            or plan["timing_tokens"] != 64 or not plan["stop_on_numerical_failure"]
            or not plan["previous_failed_gates_preserved"]
            or suite["strict_control_stability_gate"]
            != {"mean_kl": 1e-5, "absolute_mean_nll": 1e-4}
            or suite["heldout_gate"] != {"absolute_mean_nll": 1e-5,
                "reference_repeat_max_absolute_mean_nll": 1e-4}
            or len(suite["wiki_windows"]) != 100 or suite["wiki_scored_tokens"] != 102400
            or len(suite["tasks"]) != 110 or len({r["name"] for r in suite["tasks"]}) != 110
            or suite["heldout_repeats"] != 3 or suite["graph_four_numerical_repeats"] != 5):
        raise ValueError("Frozen graph-quality protocol changed")


def main() -> None:
    """Check exact source storage, fixed repeats, held-out loss, tasks and separate profiles."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--label", choices=("control", "candidate"), required=True)
    parser.add_argument("--vocabulary", type=int, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cpu-audit", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    root, label = args.root, args.label
    path = root / (label + "-quality-measurement.json")
    if path.exists():
        raise FileExistsError("Never overwrite an earlier paired quality measurement")
    suite = json.loads((root / "quality-protocol.json").read_bytes())
    plan = json.loads((root / "graph-quality-plan.json").read_bytes())
    validate_protocol(plan, suite)
    proto = json.loads((root / "protocol.json").read_bytes())
    prompts = [proto.get("quality_prefix_token_ids", []) + ids for ids in proto["quality"]]
    if len(prompts) != 16 or any(len(ids) != 258 for ids in prompts):
        raise ValueError("Preserve the original complete quality contexts")
    report: dict[str, Any] = {"status": "running", "label": label, "started_epoch": time.time(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "quality_protocol_sha256": hashlib.sha256(
            (root / "quality-protocol.json").read_bytes()).hexdigest(),
        "plan_sha256": hashlib.sha256((root / "graph-quality-plan.json").read_bytes()).hexdigest(),
        "numerical": [], "comparisons": [], "heldout": [], "timing": [],
        "previous_failed_gates_preserved": True, "direct_wrapper_used": False}

    def save(phase: str) -> None:
        report.update(phase=phase, updated_epoch=time.time())
        write(path, report)

    def retain_comparison(kind: str, left: dict[str, Any], right: dict[str, Any]) -> None:
        value = compare(left, right, root)
        passed = numerical_passed(value, suite["strict_control_stability_gate"])
        report["comparisons"].append({"kind": kind, "reference": left["label"],
            "candidate": right["label"], "comparison": value, "passed": passed})
        save("numerical_" + kind)
        if not passed:
            raise RuntimeError("Unchanged numerical limit failed; skip longer quality work")

    save("preparing")
    try:
        report["source_before"] = source_check(label, json.loads(args.cpu_audit.read_bytes()))
        report["loaded_weight_checks"] = rpc(BASE, "mxwave_checkpoint_check")
        if any(row["status"] != "passed" for row in report["loaded_weight_checks"]):
            raise RuntimeError("Loaded source tensor check failed")
        report["mhc_warmup"] = rpc(BASE, "mxwave_mhc_warmup", [2048])
        report["graph_selection"] = rpc(BASE, "mxwave_set_eager", [False])
        observed = snapshot(BASE)
        if (any(row["status"] != "complete" or row["eager"]
                for row in report["graph_selection"])
                or any(not row["graphs_enabled"] or any(
                    not row["graph_modes"][size].endswith("FULL") for size in ("1", "4"))
                       for row in observed)):
            raise RuntimeError("Both workers must select retained graphs for batches one and four")
        report["initial_snapshot"] = observed
        report["sanity"] = []
        for prompt, expected in SANITY:
            response = request(BASE, "/v1/chat/completions", {
                "model": "mxwave-glm53-pilot", "messages": [{"role": "user", "content": prompt}],
                "chat_template_kwargs": {"reasoning_effort": "low"}, "temperature": 0,
                "seed": 42, "max_tokens": 256})
            report["sanity"].append({"prompt": prompt,
                **assess_chat(response["choices"][0], expected)})
            save("sanity")
        if not all(row["passed"] for row in report["sanity"]):
            raise RuntimeError("Natural chat sanity failed before the longer quality sample")
        for index in range(5):
            save(f"numerical_repeat_{index}")
            leg = quality_leg(BASE, prompts, root, f"graph-{label}-r{index}",
                              args.vocabulary, 1.0)
            report["numerical"].append(leg)
            save(f"collected_numerical_repeat_{index}")
            if index:
                retain_comparison("self_repeat", report["numerical"][0]["decode"], leg["decode"])
            if label == "candidate":
                reference = json.loads(
                    (root / f"graph-control-r{index}-decode-quality.json").read_bytes())
                retain_comparison("paired_cross_arm", reference, leg["decode"])
        report["numerical_gate_passed"] = True
        # Timing uses normal independent HTTP streams, with no scheduler pause or profiling.
        for count in (1, 4):
            case = next(row for row in proto["timing"]
                        if row["batch"] == count and row["prompt_tokens"] == 256)
            save(f"timing_batch_{count}")
            for _ in range(2):
                batch(BASE, case["prompts"], 16)
            result: dict[str, Any] = {"batch": count, "prompt_tokens": 256, "trials": []}
            report["timing"].append(result)
            for _ in range(5):
                result["trials"].append(batch(BASE, case["prompts"], 64))
                save(f"timing_batch_{count}")
            result["median_aggregate_tokens_per_second"] = statistics.median(
                row["aggregate_tokens_per_second"] for row in result["trials"])
            result["median_ttft_seconds"] = statistics.median(
                row["median_ttft_seconds"] for row in result["trials"])
        for index in range(3):
            save(f"heldout_repeat_{index}")
            value = heldout(root, suite, f"{label}-r{index}")
            report["heldout"].append(value)
            save(f"collected_heldout_repeat_{index}")
            if abs(value["mean_nll"] - report["heldout"][0]["mean_nll"]) > 1e-4:
                raise RuntimeError("Unchanged held-out repeat limit failed; skip task sample")
            if label == "candidate":
                reference = json.loads((root / f"control-r{index}-heldout.json").read_bytes())
                if abs(value["mean_nll"] - reference["mean_nll"]) > 1e-5:
                    raise RuntimeError("Prospective paired held-out limit failed; skip task sample")
        task_root = root / label
        task_root.mkdir()
        save("practical_tasks")
        report["practical"] = tasks(task_root, suite)
        report["source_after"] = source_check(label, json.loads(args.cpu_audit.read_bytes()))
        if args.profile:
            save("profile_after_scoring")
            request(BASE, "/start_profile", {})
            try:
                response, receipt = cohort_request(BASE, {
                    "model": "mxwave-glm53-pilot", "prompt": prompts[:4], "max_tokens": 4,
                    "min_tokens": 4, "temperature": 0, "seed": 42, "ignore_eos": True,
                    "return_token_ids": True}, True, 1.0)
            finally:
                request(BASE, "/stop_profile", {}, timeout=240)
            report["profile"] = {"response": response, "receipt": receipt,
                "after_all_unprofiled_measurements": True}
            save("profile_collected")
            if not receipt["proof"]["passed"]:
                raise RuntimeError("Keep mismatched profile cohort and fail without retry")
        report.update(status="collected_pending_code_sandbox", finished_epoch=time.time())
        save("collected")
    except BaseException as exc:
        report.update(status="failed", error=repr(exc), finished_epoch=time.time())
        save("failed")
        raise


if __name__ == "__main__":
    main()
