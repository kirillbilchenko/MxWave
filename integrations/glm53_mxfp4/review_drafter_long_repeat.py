"""Locate target-feature versus draft-output drift in repeated long-prompt captures."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from .audit_drafter_feature_export_numpy import arrays
from .drafter_capture_pairing import file_digest

LAYERS = (5, 14, 24, 33, 42)


def common_prefix(first: list[int], second: list[int]) -> int:
    """Count equal response tokens before the first branch, preserving zero-based positions."""
    return next((i for i, (a, b) in enumerate(zip(first, second, strict=True)) if a != b),
                len(first))


def difference(first: np.ndarray, second: np.ndarray) -> dict[str, Any]:
    """Compare raw captured bits without converting their original floating-point formats."""
    if first.shape != second.shape or first.dtype != second.dtype:
        return {"exact": False, "shape_or_dtype_differs": True,
                "shapes": [list(first.shape), list(second.shape)]}
    raw_dtype = np.dtype(f"u{first.dtype.itemsize}")
    changed = first.view(raw_dtype) != second.view(raw_dtype)
    count = int(np.count_nonzero(changed))
    position = np.argwhere(changed)[0].tolist() if count else None
    return {"exact": count == 0, "different_elements": count, "first_index": position}


def load_run(root: Path, case: dict[str, Any], measured: dict[str, Any], rank: int) -> dict[str, Any]:
    """Validate contiguous accepted history and retain position-indexed raw feature views."""
    label = case["id"] + "-" + measured["policy"]
    paths = sorted((root / f"ct-r{rank}").glob(f"context-{label}-*-r{rank}.safetensors"))
    expected = case["prompt_ids"] + measured["result"]["token_ids"]
    needed = len(expected) - 1
    segments, sampled, frames, logits = [], [], [], {}
    accepted = 0
    for index, path in enumerate(paths):
        meta = json.loads(path.with_suffix(".json").read_bytes())
        if (meta["case"] != label or meta["rank"] != rank or meta["index"] != index
                or meta["draft_graph"] != [1, 8] or file_digest(path) != meta["payload_sha256"]):
            raise ValueError("Captured frame identity, order or digest differs")
        value = arrays(path)
        ids = value["target_input_ids"].reshape(-1)
        positions = value["target_positions"].reshape(-1)
        start = int(positions[0])
        end = int(value["draft_query_positions"].reshape(-1)[0])
        produced = int(value["target_num_sampled"].reshape(-1)[0])
        output = value["target_sampled_tokens"].reshape(-1)[:produced].tolist()
        valid = end - start
        if (not 1 <= produced <= 8 or start != accepted or not 1 <= valid <= len(ids)
                or positions[:valid].tolist() != list(range(start, end))):
            raise ValueError("Noncontiguous accepted frame or invalid sampled count")
        if index == 0:
            if valid != len(case["prompt_ids"]) or produced != 1 or meta["target_num_draft_tokens"] != 0:
                raise ValueError("Missing complete initial prefill")
        elif (len(ids) != 8 or valid != produced or meta["target_num_draft_tokens"] != 7
              or ids[0] != sampled[-1] or ids[1:valid].tolist() != output[:valid - 1]
              or int(value["target_num_rejected"].reshape(-1)[0]) != 8 - valid):
            raise ValueError("Verification does not follow the fixed-seven accepted prefix")
        limit = min(end, needed)
        if ids[:max(0, limit - start)].tolist() != expected[start:limit]:
            raise ValueError("Captured accepted inputs differ from the API response")
        if limit > start:
            segments.append((start, limit, value))
        sampled.extend(output)
        accepted = end
        indices = value["target_logits_indices"].reshape(-1).tolist()
        scores = value["target_logits"]
        if len(indices) != len(scores):
            raise ValueError("Cannot map native target logits to input positions")
        for row, offset in enumerate(indices):
            if 0 <= offset < valid and start + offset < needed:
                logits[start + offset] = scores[row]
        frames.append(value)
    if not frames or accepted < needed or sampled[:len(measured["result"]["token_ids"])] != measured["result"]["token_ids"]:
        raise ValueError("Capture is incomplete or disagrees with native sampling")
    return {"segments": segments, "frames": frames, "logits": logits,
            "response": measured["result"]["token_ids"], "needed": needed}


def compare_runs(first: dict[str, Any], second: dict[str, Any], prompt_length: int) -> dict[str, Any]:
    """Compare only rows with equal accepted histories, independent of speculative grouping."""
    common = common_prefix(first["response"], second["response"])
    limit = min(prompt_length + common, first["needed"], second["needed"])
    features = []
    for layer in LAYERS:
        count, first_position = 0, None
        i = j = 0
        while i < len(first["segments"]) and j < len(second["segments"]):
            a0, a1, a = first["segments"][i]
            b0, b1, b = second["segments"][j]
            start, stop = max(a0, b0), min(a1, b1, limit)
            if start < stop:
                left = a[f"aux_layer_{layer}"][start - a0:stop - a0]
                right = b[f"aux_layer_{layer}"][start - b0:stop - b0]
                changed = np.any(left != right, axis=1)
                count += int(np.count_nonzero(left != right))
                if first_position is None and np.any(changed):
                    first_position = start + int(np.flatnonzero(changed)[0])
            if a1 <= b1:
                i += 1
            if b1 <= a1:
                j += 1
        features.append({"layer": layer, "different_elements": count,
                         "first_different_input_position": first_position})
    prefill = {
        f"aux_layer_{layer}": difference(first["frames"][0][f"aux_layer_{layer}"],
                                         second["frames"][0][f"aux_layer_{layer}"])
        for layer in LAYERS
    }
    initial_draft = {
        key: difference(first["frames"][0][key], second["frames"][0][key])
        for key in ("draft_query_ids", "draft_query_positions", "request_seed",
                    "anchor_before_propose", "combined_target_hidden", "draft_hidden",
                    "draft_ids", "draft_values", "draft_scores", "returned_proposal")
    }
    positions = sorted(set(first["logits"]) & set(second["logits"]))
    changed_logits = []
    for position in positions:
        if position >= limit:
            continue
        result = difference(first["logits"][position], second["logits"][position])
        if not result["exact"]:
            changed_logits.append({"input_position": position, **result})
    return {
        "shared_response_prefix_tokens": common, "equal_history_feature_rows": limit,
        "prefill_features": prefill, "all_prefill_features_exact": all(r["exact"] for r in prefill.values()),
        "shared_history_features": features, "initial_draft": initial_draft,
        "shared_decision_positions": sum(p < limit for p in positions),
        "different_decision_logit_rows": len(changed_logits),
        "first_different_decision_logits": changed_logits[0] if changed_logits else None,
    }


def review(root: Path, output: Path) -> dict[str, Any]:
    """Review restored evidence without promoting a failed repeat or opening sealed data."""
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
        raise ValueError("The complete diagnostic must restore production first")
    plan = json.loads((root / "plan.json").read_bytes())
    measured = json.loads((root / "ct-measurement.json").read_bytes())
    if plan["revision"] != "drafter-long-repeat-v1" or measured["status"] != "complete":
        raise ValueError("Unexpected or incomplete diagnostic")
    case = plan["calibration_cases"][0]
    captures = [r for r in measured["cases"] if r["policy"].startswith("capture")]
    if len(captures) != 4 or len(measured["cases"]) != 6:
        raise ValueError("Missing repeat captures")
    results = []
    for rank in (0, 1):
        runs = [load_run(root, case, row, rank) for row in captures]
        for a, b in itertools.combinations(range(4), 2):
            results.append({"rank": rank, "policies": [captures[a]["policy"], captures[b]["policy"]],
                            **compare_runs(runs[a], runs[b], len(case["prompt_ids"]))})
    report = {
        "status": "reviewed", "diagnostic_only": True, "training_eligible": False,
        "plan_sha256": file_digest(root / "plan.json"), "source_sha256": file_digest(Path(__file__)),
        "production_restored": True, "sealed_test_opened": False,
        "repeat_gate_passed": measured["summary"]["repeat_gate_passed"], "pairs": results,
        "limits": ["Only five target-layer boundaries and the initial draft outputs are localized.",
                   "This diagnostic does not establish a specific kernel cause or a serving speed gain."],
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    """Independently inspect the completed repeat trace on CPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = review(args.evidence, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "pairs"}, indent=2))


if __name__ == "__main__":
    main()
