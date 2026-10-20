"""Review exactness, occupancy limits and repeated warm/cache-pressure N8 component results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

PRIMARY = ("kda-input-l0-r0", "kda-output-l0-r0", "head-r0")
TOKENS = (0, 1, 2, 3, 4, 7, 8, 9, 16, 32, 128)


def timing_summaries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recompute paired/order medians from raw positive samples instead of trusting summaries."""
    result = []
    for reference in ("original", "control"):
        for mode in ("warm", "evicted"):
            selected = [r for r in records if r["reference"] == reference and r["mode"] == mode]
            if len(selected) != 8:
                raise ValueError("Require eight paired blocks per reference/cache condition")
            pairs = []
            for record in selected:
                medians = {}
                for arm in (reference, "compact"):
                    samples = record[arm + "_samples_us"]
                    if (len(samples) != 12 or any(not isinstance(v, (int, float))
                            or not math.isfinite(v) or v <= 0 for v in samples)):
                        raise ValueError("Require twelve finite positive timing samples")
                    medians[arm] = statistics.median(samples)
                pairs.append((record["order"][0], medians["compact"] / medians[reference], medians))
            result.append({"reference": reference, "mode": mode,
                "reference_median_us": statistics.median(p[2][reference] for p in pairs),
                "compact_median_us": statistics.median(p[2]["compact"] for p in pairs),
                "median_paired_ratio": statistics.median(p[1] for p in pairs),
                "order_ratios": {first: statistics.median(p[1] for p in pairs if p[0] == first)
                                 for first in (reference, "compact")}})
    return result


def review(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Require complete independent processes before accepting an input-kernel lead."""
    if len(reports) != 2 or {r.get("confirmation") for r in reports} != {False, True}:
        raise ValueError("Require the main and fresh confirmation processes")
    for field in ("code_sha256", "original_library_sha256", "runtime_sha256",
                  "build_manifest_sha256", "fixture_manifest_sha256"):
        if len({r.get(field) for r in reports}) != 1 or not reports[0].get(field):
            raise ValueError(f"Changed or missing provenance: {field}")
    candidates = {r.get("candidate", "compact") for r in reports}
    if len(candidates) != 1:
        raise ValueError("Cannot compare different kernel candidates as confirmation")
    expected = {f"kda-{kind}-l{layer}-r{rank}" for kind in ("input", "output")
                for layer in (0, 22, 44) for rank in (0, 1)} | {"head-r0", "head-r1"}
    sessions = []
    input_lead = True
    input_nonregression = True
    library_nonregression = True
    for report in sorted(reports, key=lambda r: r["confirmation"]):
        cases = report["cases"]
        if (report.get("status") != "passed" or len(cases) != 14
                or {c["name"] for c in cases} != expected):
            raise ValueError("Incomplete fourteen-fixture qualification")
        summary: dict[str, Any] = {"confirmation": report["confirmation"], "cases": [],
            "cuda_peak_bytes": report["cuda_peak_bytes"], "seconds": report["seconds"]}
        for case in cases:
            if (set(case["weight_checks"]) != {"original", "control", "compact"}
                    or not all(case["weight_checks"].values())
                    or len(case["arithmetic"]) != len(TOKENS)
                    or {r["tokens"] for r in case["arithmetic"]} != set(TOKENS)):
                raise ValueError("Missing decoded weight or arithmetic coverage")
            for record in case["arithmetic"]:
                for kind in ("random", "basis", "stress"):
                    if (set(record[kind]) != {"original", "control", "compact"}
                            or not all(record[kind].values())):
                        raise ValueError("A candidate or control changed arithmetic bits")
            graphs = case["graphs"]
            graph_tokens = {1, 4, 8} if case["name"] in PRIMARY else {4}
            if ({g["tokens"] for g in graphs} != graph_tokens or len(graphs) != len(graph_tokens)
                    or any(not g["bits_exact"] or not g["nondefault_stream"]
                           or g["replay_allocation_growth_bytes"] for g in graphs)):
                raise ValueError("Incomplete stream/graph/replay qualification")
            a, b = case["resources"]["control"], case["resources"]["compact"]
            saving = report.get("expected_shared_saving_bytes", 2048)
            if saving not in (0, 2048) or a["dynamic_shared_bytes"] - b["dynamic_shared_bytes"] != saving:
                raise ValueError("Unexpected native resource change")
            item = {k: case[k] for k in ("name", "logical_shape", "source_weight_sha256",
                                         "artifact_sha256", "resources")}
            item["timings"] = []
            if case["name"] in PRIMARY:
                if len(case["timings"]) != 3 or {r["tokens"] for r in case["timings"]} != {1, 4, 8}:
                    raise ValueError("Missing primary timing sizes")
                for timing in case["timings"]:
                    records = timing["records"]
                    if (len(records) != 32 or {(r["reference"], r["block"], r["mode"])
                            for r in records} != {(ref, block, mode)
                                for ref in ("original", "control") for block in range(8)
                                for mode in ("warm", "evicted")}):
                        raise ValueError("Incomplete cache/reference timing blocks")
                    for record in records:
                        ref = record["reference"]
                        if (record["order"] != ([ref, "compact"] if record["block"] % 2 == 0
                                                else ["compact", ref])
                                or any(len(record[arm + "_samples_us"]) != 12
                                       for arm in (ref, "compact"))):
                            raise ValueError("Missing timing order or per-call samples")
                    recomputed = timing_summaries(records)
                    for result in recomputed:
                        ratios = [result["median_paired_ratio"], *result["order_ratios"].values()]
                        if max(ratios) > 1.02:
                            library_nonregression = False
                            if case["name"].startswith("kda-input"):
                                input_nonregression = False
                        if case["name"].startswith("kda-input") and timing["tokens"] == 4:
                            input_lead &= max(ratios) <= .97
                    item["timings"].append({"tokens": timing["tokens"],
                                             "summaries": recomputed})
            summary["cases"].append(item)
        sessions.append(summary)
    return {"status": "complete", "candidate": next(iter(candidates)),
        "correctness_passed": True, "cases_per_process": 14,
        "exact_activation_cases_per_process": 14 * len(TOKENS) * 3 * 3,
        "input_m4_lead_gate_passed": bool(input_lead),
        "input_nonregression_gate_passed": input_nonregression,
        "whole_library_nonregression_gate_passed": library_nonregression,
        "input_full_model_followup_gate_passed": bool(input_lead and input_nonregression),
        "whole_library_full_model_followup_gate_passed": bool(input_lead and library_nonregression),
        "declared_gates": {"m4_input_maximum_ratio_both_cache_conditions_and_orders": .97,
                           "maximum_primary_regression_ratio": 1.02,
                           "references": ["original", "matched_compiler_control"]},
        "sessions": sessions,
        "limits": ["Components only; no new full-model throughput, quality or memory-saving result.",
            "Fresh process confirmation shares one service pause; not a separate day or machine.",
            "Occupancy values are CUDA resource limits, not measured achieved occupancy.",
            "At most an input-only follow-up is justified if other matrices regress."]}


def main() -> None:
    """Write a compact input-hash-bound review without initializing Torch or CUDA."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs=2, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = review([json.loads(p.read_bytes()) for p in args.reports])
    result["inputs"] = {p.name: {"bytes": p.stat().st_size,
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in args.reports}
    result["analysis_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
