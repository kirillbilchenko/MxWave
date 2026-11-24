"""Offline matched quality, retained allocation, request timing and trace review for KDA."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from compare_full_model import profile, quality, quality_gate


def main() -> None:
    """Compare one immutable stock control and selective candidate after service restoration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root
    reports = {mode: json.loads((root / mode / f"{mode}-benchmark.json").read_bytes())
               for mode in ("stock", "kda")}
    if any(r["status"] != "complete" for r in reports.values()):
        raise ValueError("Both complete arms are required")
    if len({r["protocol_sha256"] for r in reports.values()}) != 1:
        raise ValueError("The paired protocols differ")

    def prefixed(mode: str, name: str) -> dict[str, Any]:
        value = copy.deepcopy(reports[mode][name])
        for context in value["contexts"]:
            context["next_logprobs_file"] = mode + "/" + context["next_logprobs_file"]
        return value

    repeat = quality(prefixed("stock", "quality"), prefixed("stock", "quality_repeat"), root)
    candidate = quality(prefixed("stock", "quality"), prefixed("kda", "quality"), root)
    gates = quality_gate(repeat, candidate)
    diagnostic_limits = {
        "mean_kl": max(1e-7, 5 * repeat["mean_kl"]) if repeat["mean_kl"] is not None else None,
        "absolute_mean_nll_difference": max(1e-5, 5 * abs(repeat["mean_nll_difference"])),
        "scope": "Repeat-relative diagnostic only; does not override the strict stability gate"}
    allocations = {mode: sum(x["cuda_allocated_bytes"]
                            for x in r["allocation_after"]["results"])
                   for mode, r in reports.items()}
    saved = allocations["stock"] - allocations["kda"]
    limits = {row["kda"].get("fused_max_tokens", 1)
              for row in reports["kda"]["allocation_after"]["results"]}
    if len(limits) != 1:
        raise ValueError("The two ranks used different fused token limits")
    fused_max_tokens = limits.pop()
    cases = []
    for left, right in zip(reports["stock"]["cases"], reports["kda"]["cases"], strict=True):
        if (left["batch"], left["prompt_tokens"]) != (right["batch"], right["prompt_tokens"]):
            raise ValueError("Timing cases differ")
        cases.append({"batch": left["batch"], "prompt_tokens": left["prompt_tokens"],
                      "stock_tokens_per_second": left["median_aggregate_tokens_per_second"],
                      "kda_tokens_per_second": right["median_aggregate_tokens_per_second"],
                      "throughput_ratio": right["median_aggregate_tokens_per_second"] /
                                          left["median_aggregate_tokens_per_second"],
                      "stock_ttft": left["median_ttft_seconds"],
                      "kda_ttft": right["median_ttft_seconds"]})
    traces = []
    for mode in reports:
        for folder in (root.parent / mode / "traces", root.parent / mode / "traces-worker"):
            for p in sorted(folder.rglob("*.trace.json*")):
                traces.append({"mode": mode, **profile(p)})
    result = {"status": "complete", "protocol_sha256": reports["stock"]["protocol_sha256"],
              "stock_repeat": repeat, "candidate_quality": candidate, "quality_gates": gates,
              "repeat_relative_diagnostic_limits": diagnostic_limits,
              "allocation_bytes": allocations, "saved_cuda_bytes_global": saved,
              "saved_allocation_fraction": saved / allocations["stock"], "timing": cases,
              "profiles": traces,
              "fused_max_tokens": fused_max_tokens,
              "limits": ["Authored paired diagnostic windows; not a standard model evaluation.",
                         "Same checkpoint, TP2, KV policy and single-token graph configuration.",
                         (f"No DFlash; fused execution up to {fused_max_tokens} tokens; "
                          "larger batches/prefill use the dense fallback."),
                         "Full-vocabulary windows evaluate prefill, not fused decode arithmetic.",
                         "Kernel trace totals overlap and are separate from timing medians."]}
    # Retain the input report hashes without copying giant trace arrays into the state.
    result["report_sha256"] = {mode: hashlib.sha256(
        (root / mode / f"{mode}-benchmark.json").read_bytes()).hexdigest() for mode in reports}
    destination = args.output or root / "kda-comparison.json"
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "quality_passed": gates["passed"],
                      "saved_cuda_bytes_global": saved}))


if __name__ == "__main__":
    main()
