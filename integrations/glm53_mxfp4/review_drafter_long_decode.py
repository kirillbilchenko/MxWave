"""Review the first equal-input decode graph and separate sparse selection order from membership."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from .analyze_sparse_prefill import masked_score_difference, selection_difference
from .audit_drafter_feature_export_numpy import arrays
from .drafter_capture_pairing import file_digest
from .review_drafter_components import active_value
from .review_drafter_long_repeat import difference
from .review_drafter_long_repeat import review as review_target


def compare(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    """Compare full explicit operands and results, retaining uncertainty about unused score columns."""
    a, b = first["trace"], second["trace"]
    if (a != b or not a["python_wrappers_removed"] or a["rows"] != 8
            or a["layers"] != [0, 1, 2, 3, 4] or a["capture_bytes"] > 64 * 1024**2):
        raise ValueError("Unexpected, changed or unbounded decode graph observations")
    for key in ("target_input_ids", "target_positions"):
        if not np.array_equal(first["x"][key], second["x"][key]):
            raise ValueError("Decode forwards do not have equal complete inputs and positions")
    comparisons = []
    exact_by_name = {}
    mapped: dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]] = {}
    for event in a["events"]:
        if event["category"] == "workspace":
            continue
        left = {leaf["path"]: active_value(first, event, leaf) for leaf in event["tensors"]}
        right = {leaf["path"]: active_value(second, event, leaf) for leaf in event["tensors"]}
        mapped[event["name"]] = left, right
        fields = {key: difference(left[key], right[key]) for key in left}
        exact = all(f["exact"] for f in fields.values())
        exact_by_name[event["name"]] = exact
        input_name = event["name"].removesuffix("/output") + "/input"
        comparisons.append({"name": event["name"], "category": event["category"],
                            "exact": exact, "fields": fields,
                            "matching_explicit_inputs": exact_by_name.get(input_name)
                            if event["name"].endswith("/output") else None})
    topk = "layer3.native_decode_topk#0"
    left, right = mapped[topk + "/input"]
    pools_a, pools_b = mapped[topk + "/output"]
    positions = first["x"]["target_positions"].reshape(-1)
    pools = selection_difference(pools_a[""], pools_b[""], (positions + 1) // 4)
    indexer = next(name for name in mapped if name.endswith(".indexer.indexer_op#0/output"))
    tokens_a, tokens_b = mapped[indexer]
    tokens = selection_difference(tokens_a[""], tokens_b[""], positions + 1)
    scores = difference(left["/scores"], right["/scores"])
    bounds = difference(left["/seq_lens"], right["/seq_lens"])
    params = next(e["parameters"] for e in a["events"] if e["name"] == topk + "/input")
    ends = left["/seq_lens"].reshape(-1)
    if (params["next_n"] != 1 or ends.shape != (8,) or not bounds["exact"]
            or not np.array_equal(ends, (positions + 1) // 4)):
        raise ValueError("Require flattened next_n=1 and explicit per-row pooled sequence bounds")
    consumed = masked_score_difference(left["/scores"], right["/scores"],
                                        np.zeros(8, dtype=np.int32), ends)
    first_result = next((e for e in comparisons if e["category"] == "result" and not e["exact"]), None)
    if first_result is not None and first_result["name"] == topk + "/output":
        first_result["matching_consumed_inputs"] = bounds["exact"] and consumed["exact"]
    return {"complete_input_ids_and_positions_exact": True,
            "first_different_result": first_result, "observations": comparisons,
            "decode_topk": {"parameters": params, "raw_score_buffer": scores,
                            "sequence_bounds": bounds,
                            "sequence_bound_values": [left["/seq_lens"].tolist(), right["/seq_lens"].tolist()],
                            "pool_selection": pools, "expanded_token_selection": tokens,
                            "unused_score_columns_may_be_present": True,
                            "consumed_scores": consumed,
                            "consumption_policy": "flattened next_n=1; [0, seq_lens[row])",
                            "all_consumed_scores_exact_implied_by_full_buffer": scores["exact"]}}


def review(root: Path, output: Path) -> dict[str, Any]:
    """Verify the full archive and locate graph-observed changes without claiming native qualification."""
    if output.exists():
        raise FileExistsError(output)
    target = review_target(root, output.with_name(output.stem + "-target.json"),
                           expected_revision="drafter-long-decode-v1")
    plan = json.loads((root / "plan.json").read_bytes())
    case = plan["calibration_cases"][0]
    pairs = []
    for rank in (0, 1):
        runs = []
        for capture in range(1, 5):
            path = root / f"ct-r{rank}" / f"context-{case['id']}-capture{capture}-0001-r{rank}.safetensors"
            meta = json.loads(path.with_suffix(".json").read_bytes())
            if file_digest(path) != meta["payload_sha256"] or meta["index"] != 1 or meta["rank"] != rank:
                raise ValueError("First-decode frame identity or digest changed")
            values = arrays(path)
            if values["target_positions"].tolist() != list(range(plan["decode_start_position"], plan["decode_start_position"] + 8)):
                raise ValueError("Different first-decode positions")
            runs.append({"trace": meta["decode_components"], "x": values})
        for a, b in itertools.combinations(range(4), 2):
            pairs.append({"rank": rank, "captures": [a + 1, b + 1], **compare(runs[a], runs[b])})
    result = {"status": "reviewed", "production_restored": True,
              "plan_sha256": file_digest(root / "plan.json"), "source_sha256": file_digest(Path(__file__)),
              "pairs": pairs, "repeat_gate_passed": target["repeat_gate_passed"],
              "training_eligible": False, "sealed_test_opened": False,
              "limits": ["Graph copies alter scheduling and are present in every request, including controls.",
                         "Matching explicit inputs do not establish matching hidden cache or kernel state.",
                         "Score comparison is limited to flattened next_n=1 and recorded per-row pooled bounds; unused columns are excluded.",
                         "No new arithmetic policy, serving-speed or full-model qualification claim."]}
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    """Independently review the restored first-decode trace."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = review(args.evidence, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "pairs"}, indent=2))


if __name__ == "__main__":
    main()
