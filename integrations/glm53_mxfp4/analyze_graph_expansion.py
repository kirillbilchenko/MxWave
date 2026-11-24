"""Recompute graph-expansion timings, IDs, source scope and strict C8 numerical comparisons."""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

from analyze_paired_speed import timing_metrics
from analyze_prefill_admission_model import forward_histogram, numerical_passed
from benchmark_full_model import write
from decode_quality import compare


def analyze(root: Path) -> dict[str, Any]:
    """Preserve all samples and qualification failures; describe policy speed independently."""
    out = root / "graph-expansion-model-review.json"
    if out.exists():
        raise FileExistsError("Preserve the first graph-expansion analysis")
    plan = json.loads((root / "graph-expansion-plan.json").read_bytes())
    path = root / "graph-expansion-measurement.json"
    report = json.loads(path.read_bytes())
    if report["status"] != "complete" or len(report["trials"]) != 72 or len(report["warmups"]) != 24:
        raise ValueError("Incomplete graph-expansion timing grid")
    expected = {(b, p, i, mode) for b, p in plan["cases"] for i in range(6)
                for mode in ("baseline", "expanded")}
    seen = set()
    rows = []
    for trial in report["trials"]:
        key = (trial["batch"], trial["prompt_tokens"], trial["trial"], trial["policy"])
        if key not in expected or key in seen:
            raise ValueError("Unexpected or duplicate graph-expansion timing trial")
        seen.add(key)
        measured = timing_metrics(trial["measurement"], trial["batch"])
        histogram = forward_histogram(trial["before"], trial["after"])
        for state in trial["before"] + trial["after"]:
            d = state["descriptors"]
            if (state["policy"] != trial["policy"] or not state["graphs_enabled"]
                    or state["physical_capture_sizes"] != [1, 2, 4, 8]
                    or state["max_num_seqs"] != 8 or not state["speculative_config_absent"]
                    or d["2"]["num_tokens"] != (4 if key[3] == "baseline" else 2)
                    or d["8"]["mode"] != ("NONE" if key[3] == "baseline" else "FULL")):
                raise ValueError("Timed graph policy differs from its actual descriptor snapshot")
        rows.append({"batch": key[0], "prompt_tokens": key[1], "trial": key[2], "policy": key[3],
                     "eager_forward_histogram": histogram, **measured})
    if seen != expected:
        raise ValueError("Missing graph-expansion timing trials")
    workloads = []
    for count, length in plan["cases"]:
        values = {}
        for mode in ("baseline", "expanded"):
            selected = [r for r in rows if (r["batch"], r["prompt_tokens"], r["policy"])
                        == (count, length, mode)]
            values[mode] = {"median_tps": statistics.median(r["aggregate_tokens_per_second"]
                                                            for r in selected),
                "median_ttft_seconds": statistics.median(r["median_ttft_seconds"] for r in selected),
                "min_tps": min(r["aggregate_tokens_per_second"] for r in selected),
                "max_tps": max(r["aggregate_tokens_per_second"] for r in selected)}
        ids_equal = True
        for trial in range(6):
            arms = [next(r for r in report["trials"] if (r["batch"], r["prompt_tokens"], r["trial"],
                    r["policy"]) == (count, length, trial, mode)) for mode in ("baseline", "expanded")]
            ids_equal &= all(a["token_ids"] == b["token_ids"] for a, b in zip(
                arms[0]["measurement"]["requests"], arms[1]["measurement"]["requests"], strict=True))
        workloads.append({"batch": count, "prompt_tokens": length, **values,
            "tps_ratio": values["expanded"]["median_tps"] / values["baseline"]["median_tps"],
            "ttft_ratio": values["expanded"]["median_ttft_seconds"] /
                          values["baseline"]["median_ttft_seconds"],
            "all_timed_ids_match": ids_equal})
    numerical = []
    for first, second in (("baseline-a", "baseline-b"), ("expanded-a", "expanded-b"),
                          ("baseline-a", "expanded-a"), ("baseline-b", "expanded-b")):
        a, b = [json.loads((root / (name + "-decode-quality.json")).read_bytes())
                for name in (first, second)]
        result = compare(a, b, root)
        numerical.append({"reference": first, "candidate": second,
                          "passed": numerical_passed(result), **result})
    source_ok = all(len(report[key]) == 2 and all(r["status"] == "passed" for r in report[key])
                    for key in ("source_before", "source_after", "loaded_checks"))
    lookup = {(r["batch"], r["prompt_tokens"]): r for r in workloads}
    speed_passed = (lookup[8, 256]["tps_ratio"] >= plan["primary_c8_min_tps_ratio"]
        and all(r["tps_ratio"] >= plan["control_min_tps_ratio"] for r in workloads)
        and all(lookup[c, 256]["ttft_ratio"] <= plan["control_max_ttft_ratio"] for c in (1, 4)))
    result = {"status": "complete", "source_checks_passed": source_ok,
        "timed_trial_groups": len(rows), "all_trials_retained": True, "workloads": workloads,
        "trials": rows, "observed_speed_gate_passed": speed_passed,
        "numerical_comparisons": numerical, "numerical_gate_passed": all(r["passed"] for r in numerical),
        "profiles_completed": len(report["profiles"]) == 4 and all(
            p["status"] == "complete" and p["after_all_unprofiled_timing"] for p in report["profiles"]),
        "actual_graph_replay_proof_pending_trace_verification": True,
        "quality_qualified": False, "previous_failed_gates_preserved": True,
        "plan_sha256": hashlib.sha256((root / "graph-expansion-plan.json").read_bytes()).hexdigest(),
        "measurement_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limits": ["Both policies share all four captured allocations and one resident model.",
            "C2 already uses padded C4 graphs in the baseline; exact C2 changes padding work.",
            "C8 changes eager to graph; actual launch proof requires the separate both-rank traces.",
            "No MTP or admission wait; fixed eight-request limit means C16 remains untested.",
            "Strict numerical/reference-repeat and practical-format failures remain open."]}
    write(out, result)
    return result
