"""Review one-load source checks and graph/eager evidence without running inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from compare_full_model import quality


def analyze(root: Path) -> dict[str, Any]:
    """Bind the retained report and describe exactly which diagnostic gates completed."""
    path = root / "model-diagnostic.json"
    raw = path.read_bytes()
    report = json.loads(raw)
    result: dict[str, Any] = {
        "source_report_sha256": hashlib.sha256(raw).hexdigest(),
        "source_status": report["status"],
        "error": report.get("error"),
        "loaded_weights": [],
        "phases": [],
        "comparisons": {},
        "compression_tested": False,
        "throughput_measured": False,
        "model_quality_qualified": False,
        "limits": [
            "Selected source values and two next-token distributions per dispatch mode.",
            "Three simple chats are a sanity gate, not an evaluation benchmark.",
            "This stock-only diagnostic cannot qualify a compression candidate.",
        ],
    }
    for worker in report.get("loaded_weight_checks", []):
        records = worker["records"]
        result["loaded_weights"].append({
            "rank": worker["rank"], "status": worker["status"],
            "expert_layers": worker["expert_layers"],
            "mxfp8_linears": worker["mxfp8_linears"],
            "compared_tensors": len(records),
            "mismatches": [row for row in records if not row["bits_exact"]],
            "limits": worker["limits"],
        })
        if "attention" in worker:
            attention = worker["attention"]
            result["loaded_weights"][-1]["attention"] = {
                "status": attention["status"],
                "expected_fused_layers": attention.get("expected_fused_layers"),
                "seen_fused_layers": attention.get("seen_fused_layers"),
                "compared_tensors": len(attention.get("records", [])),
                "mismatches": [row for row in attention.get("records", [])
                               if not row["bits_exact"]],
                "bad_modules": [row for row in attention.get("modules", [])
                                if not row.get("passed")],
                "error": attention.get("error"),
            }
    result["mhc_warmup"] = report.get("mhc_warmup", [])
    phases = {phase["label"]: phase for phase in report["phases"]}
    for phase in phases.values():
        result["phases"].append({
            "label": phase["label"], "dispatch": phase["dispatch"],
            "chat_checks": phase["chat_checks"],
            "sanity_passed": phase.get("sanity_passed", False),
            "mean_nll": phase.get("quality", {}).get("mean_nll"),
        })
    for left, right in (("graphs_before", "graphs_after"), ("graphs_before", "eager"),
                        ("eager", "graphs_after")):
        if all("quality" in phases.get(label, {}) for label in (left, right)):
            result["comparisons"][f"{left}_to_{right}"] = quality(
                phases[left]["quality"], phases[right]["quality"], root)
    if report["status"] != "complete":
        result["decision"] = "diagnostic_incomplete"
    elif not all(phase["sanity_passed"] for phase in result["phases"]):
        result["decision"] = "stock_sanity_gate_failed"
    else:
        result["decision"] = "diagnostic_sanity_passed_only"
    return result


def main() -> None:
    """Write a compact review after the original GLM service has been restored."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve prior diagnostic reviews")
    result = analyze(args.root)
    result["analysis_code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    temporary = args.output.with_suffix(".incomplete")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"decision": result["decision"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
