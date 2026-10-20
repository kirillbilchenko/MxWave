"""Locate the earliest differing observed operation in repeated long-prompt prefill."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

from .drafter_capture_pairing import file_digest
from .review_drafter_long_repeat import compare_runs, load_run


def compare_traces(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    """Compare complete explicit operands, distinguishing a result from its input drift."""
    for trace in (first, second):
        if (trace["logical_rows"] != 3973 or trace["layers"] != list(range(6))
                or not trace["full_tensors_hashed"] or not trace["scratch_contents_excluded"]
                or not trace["unused_routing_tails_excluded"]):
            raise ValueError("Unexpected or incomplete prefill observation policy")
    left, right = first["events"], second["events"]
    if ([e["name"] for e in left] != [e["name"] for e in right]
            or len({e["name"] for e in left}) != len(left)):
        raise ValueError("Observed operation ordering or multiplicity differs")
    equal: dict[str, bool] = {}
    changes = []
    for a, b in zip(left, right, strict=True):
        if a["category"] != b["category"]:
            raise ValueError("Observation category changed")
        fields_a = {t["path"]: t for t in a["tensors"]}
        fields_b = {t["path"]: t for t in b["tensors"]}
        if set(fields_a) != set(fields_b):
            raise ValueError("Observed operand structure changed")
        changed = [path for path in fields_a if fields_a[path] != fields_b[path]]
        equal[a["name"]] = not changed
        if changed:
            input_name = a["name"].removesuffix("/output") + "/input"
            changes.append({"name": a["name"], "category": a["category"],
                            "different_fields": changed,
                            "matching_explicit_inputs": equal.get(input_name)
                            if a["name"].endswith("/output") else None})
    return {"events": len(left), "different_observations": len(changes),
            "first_different_observation": changes[0] if changes else None,
            "first_different_result": next((e for e in changes if e["category"] == "result"), None),
            "changes": changes}


def review(root: Path, output: Path) -> dict[str, Any]:
    """Validate restored evidence and report observed boundaries without assuming a kernel cause."""
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((root / "evidence-manifest.json").read_bytes())
    for row in manifest["files"]:
        path = root / row["path"]
        if (not path.resolve().is_relative_to(root.resolve()) or path.stat().st_size != row["bytes"]
                or file_digest(path) != row["sha256"]):
            raise ValueError("Retained diagnostic evidence changed")
    state = json.loads((root / "controller.json").read_bytes())
    if state["phase"] != "complete_restored" or state["service_paused"] or not state["restoration"]["health_200"]:
        raise ValueError("Require a completed diagnostic and restored production")
    plan = json.loads((root / "plan.json").read_bytes())
    measured = json.loads((root / "ct-measurement.json").read_bytes())
    if plan["revision"] != "drafter-prefill-boundary-v1" or measured["status"] != "complete":
        raise ValueError("Unexpected or incomplete diagnostic")
    case = plan["calibration_cases"][0]
    captures = [r for r in measured["cases"] if r["policy"].startswith("capture")]
    if len(captures) != 4 or len(measured["cases"]) != 6:
        raise ValueError("Require four captures plus native before/after controls")
    results = []
    for rank in (0, 1):
        runs = [load_run(root, case, row, rank) for row in captures]
        traces = []
        for row in captures:
            filename = f"context-{case['id']}-{row['policy']}-0000-r{rank}.json"
            traces.append(json.loads((root / f"ct-r{rank}" / filename).read_bytes())["prefill_boundaries"])
        for a, b in itertools.combinations(range(4), 2):
            results.append({"rank": rank, "policies": [captures[a]["policy"], captures[b]["policy"]],
                            "boundaries": compare_traces(traces[a], traces[b]),
                            "features": compare_runs(runs[a], runs[b], len(case["prompt_ids"]))})
    report = {
        "status": "reviewed", "diagnostic_only": True, "training_eligible": False,
        "plan_sha256": file_digest(root / "plan.json"), "source_sha256": file_digest(Path(__file__)),
        "production_restored": True, "sealed_test_opened": False,
        "repeat_gate_passed": measured["summary"]["repeat_gate_passed"], "pairs": results,
        "limits": ["Matching explicit operands do not establish matching hidden cache or kernel state.",
                   "CPU copies synchronize eager execution and may affect overlap or scheduling.",
                   "An exact observed trace cannot qualify the unobserved runtime.",
                   "The diagnostic makes no serving-speed claim."],
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    """Independently review the retained per-operation prefill digests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = review(args.evidence, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "pairs"}, indent=2))


if __name__ == "__main__":
    main()
