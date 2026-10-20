"""Independently review native sparse-prefill scoring, selection and index expansion."""

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
from .review_drafter_long_repeat import difference


def event_fields(run: dict[str, Any], name: str) -> dict[str, np.ndarray]:
    """Resolve an exact observed operation into independently mapped raw payload fields."""
    events = [e for e in run["trace"]["events"] if e["name"] == name]
    if len(events) != 1:
        raise ValueError(f"Missing or repeated observation: {name}")
    fields = {}
    for descriptor in events[0]["tensors"]:
        value = run["values"][descriptor["key"]]
        if list(value.shape) != descriptor["shape"] or value.nbytes != descriptor["bytes"]:
            raise ValueError("Raw observation layout differs from its descriptor")
        fields[descriptor["path"]] = value
    return fields


def field_differences(first: dict[str, np.ndarray], second: dict[str, np.ndarray]) -> dict[str, Any]:
    """Compare all explicit fields without interpreting FP8 or BF16 bytes as integers numerically."""
    if set(first) != set(second):
        raise ValueError("Observed operand structure changed")
    return {key: difference(first[key], second[key]) for key in first}


def compare(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    """Separate consumed input, score, selection membership and selection order changes."""
    if ([e["name"] for e in first["trace"]["events"]]
            != [e["name"] for e in second["trace"]["events"]]):
        raise ValueError("Sparse observation order or multiplicity changed")
    indexer = "language_model.model.layers.3.self_attn.indexer.indexer_op#0"
    initial = field_differences(event_fields(first, indexer + "/input"),
                                event_fields(second, indexer + "/input"))
    expanded = selection_difference(event_fields(first, indexer + "/output")[""],
                                     event_fields(second, indexer + "/output")[""],
                                     np.arange(1, 3974, dtype=np.int64))
    chunks = []
    topk_names = [e["name"].removesuffix("/input") for e in first["trace"]["events"]
                 if e["name"].startswith("sparse_prefill.topk#") and e["name"].endswith("/input")]
    for name in topk_names:
        left = event_fields(first, name + "/input")
        right = event_fields(second, name + "/input")
        if (not np.array_equal(left["/1"], right["/1"])
                or not np.array_equal(left["/2"], right["/2"])):
            raise ValueError("Native score consumption bounds differ across repeats")
        starts, ends = left["/1"].reshape(-1), left["/2"].reshape(-1)
        if np.any(starts != 0):
            raise ValueError("This single-prompt review requires zero-based pool indices")
        scores = masked_score_difference(left["/0"], right["/0"], starts, ends)
        pools = selection_difference(event_fields(first, name + "/output")[""],
                                     event_fields(second, name + "/output")[""], ends)
        mqa_name = name.replace("sparse_prefill.topk", "sparse_prefill.mqa_logits")
        a, b = event_fields(first, mqa_name + "/input"), event_fields(second, mqa_name + "/input")
        full_inputs = field_differences(a, b)
        # Gather workspaces can contain unused keys after the largest causal end.
        # Neither those keys nor unused score columns can establish a causal drift.
        end = int(ends.max(initial=0))
        consumed_a, consumed_b = dict(a), dict(b)
        for key in ("/0/1/0", "/0/1/1"):
            if key not in a or a[key].shape[0] < end or b[key].shape[0] < end:
                raise ValueError("Cannot identify the gathered key or scale consumption extent")
            consumed_a[key], consumed_b[key] = a[key][:end], b[key][:end]
        consumed = field_differences(consumed_a, consumed_b)
        same_inputs = all(item["exact"] for item in consumed.values())
        between = [masked_score_difference(event_fields(run, mqa_name + "/output")[""],
                                           inputs["/0"], starts, ends)
                   for run, inputs in ((first, left), (second, right))]
        if not same_inputs:
            observation = "gathered_scoring_operands_differ"
        elif not scores["exact"]:
            observation = "scores_differ_with_equal_recorded_consumed_operands"
        elif pools["different_multiset_rows"]:
            observation = "selected_pool_membership_differs_with_identical_consumed_scores"
        elif pools["different_order_rows"]:
            observation = "selected_pool_order_differs_with_identical_consumed_scores"
        else:
            observation = "recorded_scoring_and_pool_selection_repeat"
        chunks.append({"name": name, "observation": observation,
                       "full_score_input_fields": full_inputs,
                       "consumed_score_input_fields": consumed,
                       "consumed_score_inputs_exact": same_inputs,
                       "consumed_scores": scores, "pool_selection": pools,
                       "compute_to_selection_score_checks": between})
    if not chunks:
        raise ValueError("No native score/selection chunks were observed")
    features = field_differences(
        {name: first["values"][name] for name in first["values"] if name.startswith("aux_layer_")},
        {name: second["values"][name] for name in second["values"] if name.startswith("aux_layer_")},
    )
    return {"indexer_input_fields": initial,
            "all_explicit_indexer_inputs_exact": all(r["exact"] for r in initial.values()),
            "expanded_token_selection": expanded, "chunks": chunks,
            "prefill_feature_fields": features,
            "initial_target_logits": difference(first["values"]["target_logits"],
                                                 second["values"]["target_logits"])}


def review(root: Path, output: Path) -> dict[str, Any]:
    """Validate all archived evidence and compare the four native first-prefill captures per rank."""
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((root / "evidence-manifest.json").read_bytes())
    for row in manifest["files"]:
        path = root / row["path"]
        if (not path.resolve().is_relative_to(root.resolve()) or path.stat().st_size != row["bytes"]
                or file_digest(path) != row["sha256"]):
            raise ValueError("Retained sparse-prefill evidence changed")
    state = json.loads((root / "controller.json").read_bytes())
    if state["phase"] != "complete_restored" or state["service_paused"] or not state["restoration"]["health_200"]:
        raise ValueError("Require completed diagnostic and restored production")
    plan = json.loads((root / "plan.json").read_bytes())
    measured = json.loads((root / "ct-measurement.json").read_bytes())
    if plan["revision"] != "drafter-sparse-prefill-v1" or measured["status"] != "complete":
        raise ValueError("Unexpected or incomplete sparse diagnostic")
    case = plan["calibration_cases"][0]
    captures = [r for r in measured["cases"] if r["policy"].startswith("capture")]
    if len(captures) != 4 or len(measured["cases"]) != 6:
        raise ValueError("Missing native controls or repeated captures")
    pairs = []
    for rank in (0, 1):
        runs = []
        for row in captures:
            label = case["id"] + "-" + row["policy"]
            path = root / f"ct-r{rank}" / f"context-{label}-0000-r{rank}.safetensors"
            meta = json.loads(path.with_suffix(".json").read_bytes())
            if (file_digest(path) != meta["payload_sha256"] or meta["rank"] != rank
                    or meta["case"] != label or meta["index"] != 0 or meta["draft_graph"] != [1, 8]):
                raise ValueError("Initial frame identity or digest differs")
            values = arrays(path)
            if (values["target_input_ids"].tolist() != case["prompt_ids"]
                    or values["target_positions"].tolist() != list(range(3973))
                    or int(values["target_num_sampled"].reshape(-1)[0]) != 1
                    or int(values["target_sampled_tokens"].reshape(-1)[0]) != row["result"]["token_ids"][0]):
                raise ValueError("Captured initial prefill differs from the plan or native response")
            trace = meta["sparse_prefill"]
            if trace["rows"] != 3973 or trace["layer"] != 3 or trace["arithmetic_replaced"]:
                raise ValueError("Unexpected sparse observation policy")
            runs.append({"values": values, "trace": trace})
        for a, b in itertools.combinations(range(4), 2):
            pairs.append({"rank": rank, "policies": [captures[a]["policy"], captures[b]["policy"]],
                          **compare(runs[a], runs[b])})
    result = {"status": "reviewed", "diagnostic_only": True, "training_eligible": False,
              "sealed_test_opened": False, "production_restored": True,
              "evidence_payloads_verified": len(manifest["files"]),
              "plan_sha256": file_digest(root / "plan.json"), "source_sha256": file_digest(Path(__file__)),
              "repeat_gate_passed": measured["summary"]["repeat_gate_passed"], "pairs": pairs,
              "limits": ["The observations can change GPU scheduling; they do not qualify native serving.",
                         "The diagnostic makes no speed or cross-model quality claim."]}
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    """Review sparse-score and selected-index evidence on CPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = review(args.evidence, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "pairs"}, indent=2))


if __name__ == "__main__":
    main()
