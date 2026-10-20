"""Recompute C1 timing, metric deltas and unchanged numerical gates from retained pilot evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

from benchmark_full_model import write
from decode_quality import compare
from mtp_metrics import delta


def numerical_passed(value: dict[str, Any]) -> bool:
    """Require all 48 actual decode comparisons, matching conditioning and the existing floors."""
    kl, nll = value["mean_decode_kl"], value["mean_decode_nll_difference"]
    return (value["expected_decode_steps"] == value["decode_steps"] == 48
        and value["all_decode_conditioning_matches"] and value["all_generated_ids_match"]
        and kl is not None and nll is not None and math.isfinite(kl) and math.isfinite(nll)
        and 0 <= kl <= 1e-5 and abs(nll) <= 1e-4)


def timed_rows(arm: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    """Check every fixed-length timed request and recompute throughput from its wall clock."""
    if (arm["status"] != "complete" or len(arm["trials"]) != 10
            or len(arm["warmups"]) != 4):
        raise ValueError("Incomplete MTP pilot timing")
    rows: dict[int, list[dict[str, Any]]] = {256: [], 2048: []}
    seen = set()
    for trial in arm["trials"]:
        key = (trial["trial"], trial["prompt_tokens"])
        if key in seen or key[0] not in range(5) or key[1] not in rows:
            raise ValueError("Missing or duplicated MTP timing trial")
        seen.add(key)
        measured = trial["measurement"]
        if len(measured["requests"]) != 1:
            raise ValueError("This pilot compares C1 only")
        request = measured["requests"][0]
        if (len(request["token_ids"]) != 64
                or sum(a["tokens"] for a in request["arrivals"]) != 64
                or not 0 < request["ttft_seconds"] <= request["elapsed_seconds"]
                <= measured["wall_seconds"]):
            raise ValueError("Incomplete or invalid fixed-token API measurement")
        tps = 64 / measured["wall_seconds"]
        if not math.isclose(tps, measured["aggregate_tokens_per_second"], rel_tol=1e-12):
            raise ValueError("Retained throughput differs from its actual wall time")
        rows[key[1]].append({"trial": key[0], "tokens_per_second": tps,
            "ttft_seconds": request["ttft_seconds"], "wall_seconds": measured["wall_seconds"],
            "decode_tokens_per_second": request["decode_tokens_per_second"],
            "stream_chunks": len(request["arrivals"]), "token_ids": request["token_ids"]})
    return {n: sorted(values, key=lambda r: r["trial"]) for n, values in rows.items()}


def main() -> None:
    """Preserve speed observations even if baseline repeats or MTP numerical comparisons fail."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads((args.root / "mtp-pilot-plan.json").read_bytes())
    if plan["mean_decode_kl_floor"] != 1e-5 or plan["abs_mean_decode_nll_floor"] != 1e-4:
        raise ValueError("Numerical floors changed")
    arms = {label: json.loads((args.root / (label + "-mtp.json")).read_bytes())
            for label in ("control", "candidate")}
    timing = {label: timed_rows(arm) for label, arm in arms.items()}
    comparisons = []
    for first, second in (("control-a", "control-b"), ("candidate-a", "candidate-b"),
                          ("control-a", "candidate-a"), ("control-b", "candidate-b")):
        reports = [json.loads((args.root / (label + "-decode-quality.json")).read_bytes())
                   for label in (first, second)]
        result = compare(*reports, args.root)
        comparisons.append({"reference": first, "candidate": second,
                            "passed": numerical_passed(result), **result})
    workloads = []
    for length in (256, 2048):
        values = {}
        for label in arms:
            values[label] = {"median_tokens_per_second": statistics.median(
                r["tokens_per_second"] for r in timing[label][length]),
                "median_ttft_seconds": statistics.median(r["ttft_seconds"]
                    for r in timing[label][length]), "trials": timing[label][length]}
        left, right = values["control"], values["candidate"]
        workloads.append({"concurrency": 1, "prompt_tokens": length, **values,
            "tps_change_percent": 100 * (right["median_tokens_per_second"] /
                                          left["median_tokens_per_second"] - 1),
            "ttft_change_percent": 100 * (right["median_ttft_seconds"] /
                                           left["median_ttft_seconds"] - 1),
            "all_timed_ids_match": all(a["token_ids"] == b["token_ids"] for a, b in
                zip(timing["control"][length], timing["candidate"][length], strict=True))})
    metric_deltas = {}
    for label, arm in arms.items():
        texts = [(args.root / f"{label}-metrics-{phase}-timing.txt").read_text()
                 for phase in ("before", "after")]
        metric_deltas[label] = delta(*texts)
        if metric_deltas[label] != arm["timed_speculative_counter_delta"]:
            raise ValueError("Speculative deltas differ from retained native counter exposition")
    source_ok = all(len(arm[key]) == 2 and all(r["status"] == "passed" for r in arm[key])
                    for arm in arms.values() for key in
                    ("source_before", "source_after", "base_loaded_checks", "draft_loaded_checks"))
    result = {"status": "complete", "experiment": "native MTP depth-two C1 pilot",
        "source_checks_passed": source_ok, "workloads": workloads,
        "timed_speculative_counter_deltas": metric_deltas, "numerical_comparisons": comparisons,
        "pilot_numerical_passed": source_ok and all(c["passed"] for c in comparisons)
                                  and all(c["all_timed_ids_match"] for c in workloads),
        "quality_qualified": False, "previous_failed_gates_preserved": True,
        "raw_reports_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(args.root.glob("*.json")) if p.name != "quality-model-review.json"},
        "limits": ["Two sequential loads and five trials per prompt; clocks/order effects remain.",
            "MTP stream chunks can contain several tokens; no per-token GPU time is inferred.",
            "Numerical controls compare live decode only; open full-model MLA cause remains.",
            "No PPL, broad task/GSM8K rerun or release qualification in this pilot."]}
    write(args.root / "quality-model-review.json", result)
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("workloads", "numerical_comparisons", "raw_reports_sha256")}))


if __name__ == "__main__":
    main()
