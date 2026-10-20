"""Verify and summarize a restored native A/B/A development comparison."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from .drafter_fp8_roundtrip import digest


def geometric_mean(values: list[float]) -> float:
    """Average positive paired ratios without weighting slow prompts more heavily."""
    return math.exp(statistics.mean(math.log(value) for value in values))


def metric(counters: dict[str, float], name: str) -> float:
    """Find one aggregate Prometheus series, rejecting ambiguous engine counts."""
    matches = [value for key, value in counters.items()
               if key.split("{", 1)[0] == "vllm:spec_decode_" + name + "_total"]
    if len(matches) != 1:
        raise ValueError(f"Expected one aggregate speculative counter: {name}")
    return matches[0]


def summarize_requests(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Require fixed outputs/counters and summarize ordinary end-to-end streams."""
    if len(rows) != 6 or sorted(row["trial"] for row in rows) != list(range(6)):
        raise ValueError("Expected six trials for each prompt")
    first = rows[0]
    for row in rows:
        result = row["result"]
        if (result["token_ids"] != first["result"]["token_ids"]
                or row["counter_delta"] != first["counter_delta"]
                or len(result["token_ids"]) != 256
                or result["usage"]["completion_tokens"] != 256):
            raise ValueError("Within-load output, budget or speculative repeat failed")
    latencies = [row["result"]["latency_seconds"] for row in rows]
    counters = first["counter_delta"]
    rounds = metric(counters, "num_drafts")
    drafts = metric(counters, "num_draft_tokens")
    accepted = metric(counters, "num_accepted_tokens")
    if drafts != 7 * rounds:
        raise ValueError("Native draft length differs from the fixed seven-token policy")
    return {
        "trials": 6, "median_end_to_end_tps": statistics.median(256 / x for x in latencies),
        "median_ttft_seconds": statistics.median(r["result"]["ttft_seconds"] for r in rows),
        "latency_relative_range": (max(latencies) - min(latencies)) / statistics.median(latencies),
        "draft_rounds": rounds, "draft_tokens": drafts, "accepted_tokens": accepted,
        "accepted_per_draft_call": accepted / rounds, "acceptance_fraction": accepted / drafts,
        "repeat_tokens_and_counters_exact": True,
    }


def review(root: Path, output: Path) -> dict[str, Any]:
    """Recompute comparisons after verified restoration; never qualify final serving speed."""
    if output.exists():
        raise FileExistsError(output)
    state = json.loads((root / "controller.json").read_bytes())
    if state["phase"] != "complete_restored" or state["service_paused"]:
        raise ValueError("All measurements and production restoration must complete")
    manifest = json.loads((root / "evidence-manifest.json").read_bytes())
    for row in manifest["files"]:
        path = root / row["path"]
        if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError(f"Archived evidence differs: {row['path']}")
    plan = json.loads((root / "plan.json").read_bytes())
    prepared = json.loads((root / "prepared.json").read_bytes())
    if digest(root / "plan.json") != prepared["plan_sha256"]:
        raise ValueError("Prepared plan differs")
    if not state["restoration"]["originals_running"] or not state["restoration"]["health_200"]:
        raise ValueError("Original production restoration not healthy")
    models = state["restoration"]["models"]["data"]
    if not any(r["id"] == "glm-5.3-flash" and r["max_model_len"] == 500000 for r in models):
        raise ValueError("Restored model identity or context differs")
    reports = {arm: json.loads((root / f"{arm}-measurement.json").read_bytes())
               for arm in plan["arms"]}
    for arm, report in reports.items():
        if report["status"] != "complete" or len(report["cases"]) != 48:
            raise ValueError(f"Incomplete native measurement: {arm}")
        for rank in (0, 1):
            inventory = json.loads((root / f"{arm}-r{rank}" /
                                    f"draft-runtime-weights-r{rank}.json").read_bytes())
            expected = json.loads((root / f"expected-runtime-r{rank}.json").read_bytes())
            expected["model.fc.weight"]["sha256"] = plan["context_payloads"][plan["variants"][arm]]
            if inventory != expected:
                raise ValueError("Effective native weights differ from the selected export")
    cases = []
    for case in plan["calibration_cases"]:
        rows = {arm: [r for r in report["cases"] if r["case"] == case["id"]]
                for arm, report in reports.items()}
        arms = {arm: summarize_requests(value) for arm, value in rows.items()}
        same_tokens = all(value[0]["result"]["token_ids"] == rows["a1"][0]["result"]["token_ids"]
                          for value in rows.values())
        baseline_counters_equal = rows["a1"][0]["counter_delta"] == rows["a2"][0]["counter_delta"]
        baseline_tps = math.sqrt(
            arms["a1"]["median_end_to_end_tps"] * arms["a2"]["median_end_to_end_tps"],
        )
        baseline_ttft = math.sqrt(
            arms["a1"]["median_ttft_seconds"] * arms["a2"]["median_ttft_seconds"],
        )
        cases.append({
            "id": case["id"], "domain": case["domain"], "arms": arms,
            "cross_arm_tokens_identical": same_tokens,
            "baseline_repeat_counters_equal": baseline_counters_equal,
            "candidate_tps_ratio": arms["b1"]["median_end_to_end_tps"] / baseline_tps,
            "candidate_ttft_ratio": arms["b1"]["median_ttft_seconds"] / baseline_ttft,
            "candidate_rounds_ratio": arms["b1"]["draft_rounds"] / arms["a1"]["draft_rounds"],
            "baseline_load_tps_ratio": arms["a2"]["median_end_to_end_tps"]
            / arms["a1"]["median_end_to_end_tps"],
        })
    domains = {}
    for domain in ("code", "chat", "explanation"):
        selected = [row for row in cases if row["domain"] == domain]
        domains[domain] = {
            "prompts": len(selected),
            **{key: geometric_mean([row[key] for row in selected]) for key in (
                "candidate_tps_ratio", "candidate_ttft_ratio", "candidate_rounds_ratio",
                "baseline_load_tps_ratio",
            )},
        }
    stable = all(abs(row["baseline_load_tps_ratio"] - 1) <= 0.02 for row in cases)
    result = {
        "status": "verified", "source_sha256": digest(Path(__file__)),
        "plan_sha256": digest(root / "plan.json"), "payloads_verified": len(manifest["files"]),
        "native_reload_verified": True, "domains": domains, "cases": cases,
        "all_cross_arm_tokens_identical": all(r["cross_arm_tokens_identical"] for r in cases),
        "baseline_repeat_counters_equal": all(r["baseline_repeat_counters_equal"] for r in cases),
        "all_prompt_baseline_load_controls_within_two_percent": stable,
        "production_restored": True, "sealed_test_opened": False,
        "final_serving_qualification_pass": False,
        "limits": [
            "Eight reused validation prompts, one candidate load, and two baseline loads.",
            "Six trials within a load are not six independent model loads or new prompts.",
            "Per-call counters may include post-cap work from the final verification.",
            "No final 8% speed claim, confidence interval, C8/C16 or long-context qualification.",
        ],
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    """Review retained native results without accessing sealed prompts or live services."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = review(args.root, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, indent=2))


if __name__ == "__main__":
    main()
