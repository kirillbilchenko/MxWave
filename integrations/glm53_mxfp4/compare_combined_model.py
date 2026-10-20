"""Review combined allocation, saved timing, prefill diagnostics and actual decode quality."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import runpy
import sys
from pathlib import Path
from typing import Any

from compare_full_model import quality_gate
from decode_quality import compare


def main() -> None:
    """Apply the same predeclared control stability floor to real decode distributions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    base = root / "combined-prefill-review.json"
    sys.argv = [str(Path(__file__).with_name("compare_kda_model.py")), "--root", str(root),
                "--output", str(base)]
    runpy.run_path(sys.argv[0], run_name="__main__")
    result = json.loads(base.read_bytes())

    def read(mode: str, label: str) -> dict[str, Any]:
        report = json.loads((root / mode / (label + "-decode-quality.json")).read_bytes())
        value = copy.deepcopy(report)
        for context in value["contexts"]:
            for step in context["steps"]:
                step["file"] = mode + "/" + step["file"]
        return value

    reference = read("stock", "stock")
    repeat = compare(reference, read("stock", "stock-repeat"), root)
    candidate = compare(reference, read("kda", "kda"), root)

    def gate_input(value: dict[str, Any]) -> dict[str, Any]:
        return {"mean_kl": value["mean_decode_kl"],
                "mean_nll_difference": value["mean_decode_nll_difference"]
                    if value["mean_decode_nll_difference"] is not None else float("inf"),
                "all_greedy_tokens_exact": value["all_generated_ids_match"]}

    gates = quality_gate(gate_input(repeat), gate_input(candidate))
    if (not repeat["all_decode_conditioning_matches"]
            or not candidate["all_decode_conditioning_matches"]
            or candidate["expected_decode_steps"] != 48):
        gates.update(passed=False, status="unqualified_changed_conditioning")
    result["actual_decode"] = {"stock_repeat": repeat, "candidate": candidate, "gates": gates,
        "protocol": {"contexts": 16, "generated_tokens": 4, "decode_steps_per_context": 3,
            "prefill_step_excluded_from_decode_kl": True,
            "comparison_requires_identical_generated_prefix": True}}
    result["candidate"] = "expert-scales + KDA-8 + compressed head + row-addressable embedding"
    result["stock_timing_reused"] = True
    reports = {mode: json.loads((root / mode / (mode + "-benchmark.json")).read_bytes())
               for mode in ("stock", "kda")}
    components = []
    for rank, row in enumerate(reports["kda"]["allocation_after"]["results"]):
        kda, vocab = row["kda"], row["vocabulary"]
        components.append({"rank": rank, "expert_scale_raw_bytes": row["scale_raw_bytes"],
            "expert_scale_retained_bytes": row["scale_retained_bytes"],
            "expert_gemm_scratch_bytes": row["scratch_bytes"],
            "expert_scale_layers": row["layers_seen"],
            "kda_raw_bytes": kda["raw_bytes"], "kda_packed_bytes": kda["packed_bytes"],
            "kda_pool_bytes": kda["pool_bytes"], "kda_layers": kda["resident_layers"],
            "vocabulary_raw_bytes": vocab["raw_bytes"],
            "vocabulary_retained_bytes": vocab["retained_bytes"],
            "vocabulary_layers": vocab["resident_layers"],
            "vocabulary_dense_parameters_alive": vocab["dense_parameter_references_alive"]})
    result["components"] = components
    for row in result["timing"]:
        row["combined_tokens_per_second"] = row.pop("kda_tokens_per_second")
        row["combined_ttft"] = row.pop("kda_ttft")
    result["limits"] = [
        "One imported checkpoint, TP2, fixed KV policy, graph size one; no DFlash or MTP.",
        "Saved stock throughput from the identical model/settings; sequential, not interleaved.",
        "Stock repeat and candidate include actual fused autoregressive decode steps.",
        "Weights and scale decoding are exact; changed GEMM arithmetic needs quality checks.",
        "Four-bit expert codes and other MXFP8/BF16 tensors remain in their existing formats.",
        "Authored diagnostic contexts and three simple chats do not replace a standard evaluation.",
    ]
    result["decode_report_sha256"] = {f"{mode}/{label}": hashlib.sha256(
        (root / mode / (label + "-decode-quality.json")).read_bytes()).hexdigest()
        for mode, label in (("stock", "stock"), ("stock", "stock-repeat"), ("kda", "kda"))}
    destination = root / "combined-comparison.json"
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "complete", "saved_cuda_bytes_global":
        result["saved_cuda_bytes_global"], "decode_quality": gates["status"]}))


if __name__ == "__main__":
    main()
