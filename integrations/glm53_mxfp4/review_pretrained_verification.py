"""Locate changed native decisions using equal accepted histories and verified traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .drafter_fp8_roundtrip import digest
from .review_drafter_verification import (
    compare_recurrent_inputs,
    difference,
    frames,
    logit_comparison,
    numeric,
)


def decision(
    records: list[dict[str, Any]], index: int, prompt: list[int], output: list[int],
) -> dict[str, Any]:
    """Select the actual sampled row and validate its causal input prefix and position."""
    frame = next(r for r in records
                 if r["output_start"] <= index < r["output_start"] + len(r["sampled"]))
    offset = index - frame["output_start"]
    x = frame["x"]
    row = int(x["target_logits_indices"].reshape(-1)[offset])
    position = len(prompt) + index - 1
    if int(x["target_positions"][row]) != position:
        raise ValueError("Decision row does not address the common-prefix boundary")
    history = prompt + output[:index]
    positions = x["target_positions"].reshape(-1)[:row + 1].tolist()
    inputs = x["target_input_ids"].reshape(-1)[:row + 1].tolist()
    if (any(p < 0 or p >= len(history) for p in positions)
            or inputs != [history[p] for p in positions]):
        raise ValueError("Observed decision inputs differ from the accepted causal history")
    if frame["sampled"][offset] != output[index]:
        raise ValueError("Retained native sample differs from the API token")
    return {"frame": frame, "offset": offset, "input_row": row,
            "position": position, "token": output[index],
            "logits": numeric(x["target_logits"][offset],
                              frame["meta"]["tensor_dtypes"]["target_logits"])}


def compare_decision(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Separate score differences from selected-token differences at one shared prefix."""
    a, b = left["frame"], right["frame"]
    start_equal = int(a["x"]["target_positions"][0]) == int(b["x"]["target_positions"][0])
    tokens = [left["token"], right["token"]]
    logits = logit_comparison(left["logits"], right["logits"], tokens)
    return {
        "position": left["position"], "frames": [a["path"], b["path"]],
        "logit_rows": [left["offset"], right["offset"]],
        "input_rows": [left["input_row"], right["input_row"]],
        "sampled_tokens": tokens, "pre_sampler_logits": logits,
        "sample_is_pre_sampler_argmax": [
            tokens[0] in logits["a_argmax_ids"], tokens[1] in logits["b_argmax_ids"]],
        "final_hidden": difference(a["x"]["target_final_hidden"][left["input_row"]],
                                   b["x"]["target_final_hidden"][right["input_row"]]),
        "graph_descriptors": [r["meta"].get("target_graph_descriptor") for r in (a, b)],
        "target_inputs": [r["x"]["target_input_ids"].tolist() for r in (a, b)],
        "target_positions": [r["x"]["target_positions"].tolist() for r in (a, b)],
        "before_state_boundaries_equal": start_equal,
        "selected_recurrent_inputs": compare_recurrent_inputs(a, b) if start_equal else None,
        "post_sampler_original_buffer": [
            {str(t): float(numeric(r["frame"]["x"]["target_logits_after_sampler"][r["offset"]],
                                  r["frame"]["meta"]["tensor_dtypes"][
                                      "target_logits_after_sampler"])[t]) for t in tokens}
            for r in (left, right)
        ],
    }


def review(root: Path, output: Path) -> dict[str, Any]:
    """Verify completed controls and artifacts before reporting the first score divergence."""
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((root / "evidence-manifest.json").read_bytes())
    for row in manifest["files"]:
        path = root / row["path"]
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Evidence path escapes root")
        if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError(f"Evidence changed: {row['path']}")
    state = json.loads((root / "controller.json").read_bytes())
    assert state["phase"] == "complete_restored" and not state["service_paused"]
    restore = state["restoration"]
    assert restore["health_200"] and restore["originals_running"]
    assert any(m["id"] == "glm-5.3-flash" and m["max_model_len"] == 500000
               for m in restore["models"]["data"])
    plan = json.loads((root / "plan.json").read_bytes())
    assert digest(root / "plan.json") == json.loads((root / "prepared.json").read_bytes())[
        "plan_sha256"]
    case = plan["calibration_cases"][0]
    assert case["split"] == "validation" and not plan["sealed_test_opened"]
    expected = plan["expected_outputs"]
    first = next(i for i, (a, b) in enumerate(zip(expected["a1"], expected["b1"], strict=True))
                 if a != b)
    captures, repeats = {}, []
    for arm in ("a1", "b1"):
        measured = json.loads((root / f"{arm}-measurement.json").read_bytes())
        assert measured["status"] == "complete"
        rows = measured["cases"]
        assert [r["policy"] for r in rows] == ["before", "capture1", "capture2", "after"]
        assert all(r["result"]["token_ids"] == expected[arm]
                   and r["counter_delta"] == rows[0]["counter_delta"] for r in rows)
        for rank in (0, 1):
            startup = json.loads((root / f"{arm}-r{rank}/probe-installed-r{rank}.json").read_bytes())
            layers = plan["variant_feature_layers"][plan["variants"][arm]]
            assert startup["target_feature_layers"] == layers
            assert startup["checkpoint_files_verified"] and startup["fc_native_bitwise_equal"]
            for policy in ("capture1", "capture2"):
                records = frames(root, arm, case["id"], policy, rank)
                sampled = [t for r in records for t in r["sampled"]]
                assert sampled[:plan["output_tokens"]] == expected[arm]
                captures[(arm, policy, rank)] = records
            aa, bb = (captures[(arm, policy, rank)] for policy in ("capture1", "capture2"))
            assert len(aa) == len(bb)
            names = ["target_input_ids", "target_positions", "target_logits",
                     "target_logits_after_sampler", "target_final_hidden", "returned_proposal",
                     *[f"aux_layer_{i}" for i in layers]]
            exact = all(np.array_equal(a["x"][n], b["x"][n])
                        for a, b in zip(aa, bb, strict=True) for n in names)
            repeats.append({"arm": arm, "rank": rank, "frames": len(aa),
                            "inputs_logits_features_proposals_exact": exact})
    rank_results = []
    for rank in (0, 1):
        decisions = []
        for index in range(first + 1):
            pair = [decision(captures[(arm, "capture1", rank)], index,
                             case["prompt_ids"], expected[arm]) for arm in ("a1", "b1")]
            value = compare_decision(*pair)
            value["output_index"] = index
            decisions.append(value)
        changed = [r for r in decisions if not r["pre_sampler_logits"]["bitwise_equal"]]
        rank_results.append({"rank": rank,
            "first_changed_logits": changed[0] if changed else None,
            "first_different_output": decisions[-1],
            "shared_history_decisions": [{
                "index": r["output_index"], "logits_exact": r["pre_sampler_logits"]["bitwise_equal"],
                "forward_kl": r["pre_sampler_logits"]["forward_kl_a_to_b"],
                "same_selected_token": r["sampled_tokens"][0] == r["sampled_tokens"][1],
            } for r in decisions]})
    report = {
        "status": "reviewed", "source_sha256": digest(Path(__file__)),
        "payloads_verified": len(manifest["files"]), "plan_sha256": digest(root / "plan.json"),
        "original_prefix_and_counter_controls_pass": True, "capture_repeats": repeats,
        "all_capture_repeats_exact": all(r["inputs_logits_features_proposals_exact"] for r in repeats),
        "first_different_output_index": first, "rank_results": rank_results,
        "production_restored": True, "sealed_test_opened": False,
        "limits": [
            "One development prompt; decision KL is not corpus quality.",
            "Different drafter feature taps are never treated as matching layer boundaries.",
            "Only layers0/4 recurrent state retained; cache values compared only at equal forward start positions.",
            "Recurrent slot selection uses the previously audited accepted-column-minus-one rule.",
            "Convolution-history slices are retained but not independently compared by this reviewer.",
            "Post-sampler copy observes the original logits buffer,not separately processed logits.",
            "Boundary differences locate a stage; they do not alone identify a responsible kernel.",
        ],
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    """Review retained evidence locally without modifying serving."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = review(args.root, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "rank_results"}, indent=2))


if __name__ == "__main__":
    main()
