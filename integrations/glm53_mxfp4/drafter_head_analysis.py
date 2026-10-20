"""Independently inspect retained draft-head tensors on CPU, without CUDA or vLLM."""

from __future__ import annotations

import hashlib
import json
import re
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file


def bf16_values(bits: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Decode BF16 bit patterns with no arithmetic rounding."""
    return (bits.view(np.uint16).astype(np.uint32) << 16).view(np.float32)


def greedy_path(
    ids: np.ndarray[Any, Any], scores: np.ndarray[Any, Any]
) -> list[int]:
    """Evaluate a zero-temperature chain by choosing each conditional maximum."""
    candidates = ids.reshape(7, 16)
    edges = scores.reshape(7, 16, 16)
    previous = 0
    output = []
    for step in range(7):
        previous = int(np.argmax(edges[step, previous]))
        output.append(int(candidates[step, previous]))
    return output


def counters(metrics: str) -> dict[str, float]:
    """Preserve every speculative counter label for exact repeat comparison."""
    result = {}
    for line in metrics.splitlines():
        if line.startswith("vllm:spec_decode_"):
            key, value = line.rsplit(" ", 1)
            if key.split("{", 1)[0].endswith("_total"):
                result[key] = float(value)
    return result


def verify_round_coverage(
    starts: list[int], prompt_tokens: int, response_tokens: int,
    reported_rounds: int, reported_accepted: int, accepted_by_position: list[int],
) -> dict[str, Any]:
    """Match graph positions to verified rounds, retaining speculative work past the cap."""
    response_end = prompt_tokens + response_tokens
    in_response = sum(position < response_end for position in starts)
    advances = [b - a - 1 for a, b in pairwise(starts)]
    accepted = advances[:reported_rounds]
    reconstructed = [sum(count > position for count in accepted) for position in range(7)]
    passed = (
        bool(starts) and starts[0] == prompt_tokens + 1
        and len(starts) > reported_rounds and in_response == reported_rounds
        and all(0 <= count <= 7 for count in advances)
        and sum(accepted) == reported_accepted and reconstructed == accepted_by_position
    )
    return {
        "passed": passed, "captured_replays": len(starts), "reported_rounds": reported_rounds,
        "response_end_exclusive": response_end, "replays_within_response": in_response,
        "post_cap_replay_starts": [p for p in starts if p >= response_end],
        "accepted_prefixes_from_positions": accepted,
        "reconstructed_accepted_total": sum(accepted),
        "reconstructed_accepted_by_position": reconstructed,
        "reported_accepted_total": reported_accepted,
        "reported_accepted_by_position": accepted_by_position,
        "literal_replay_count_equals_verified_rounds": len(starts) == reported_rounds,
    }


def analyze(root: Path) -> dict[str, Any]:
    """Recompute proposal differences, controls and sampled raw-logit statistics."""
    measurement = json.loads((root / "head-measurement.json").read_bytes())
    inputs = json.loads((root / "inputs.json").read_bytes())
    assert measurement["status"] == "complete" and len(measurement["cases"]) == 6
    state = json.loads((root / "controller.json").read_bytes())
    assert state["phase"] == "complete_restored" and not state["service_paused"]
    case_names = ("code_256", "explanation_256")
    report: dict[str, Any] = {
        "schema": "mxwave-draft-head-review-v1",
        "scope": "CT MXFP8, C1, identical seven-row draft states; shadow dense head",
        "timing_claim": False,
        "cases": {},
        "restored_original_production": state["restoration"]["originals_running"],
        "raw_logit_scope": "Independent full-logit checks only on retained sampled frames",
    }
    for case in case_names:
        requests = [x for x in measurement["cases"] if x["case"] == case]
        assert [x["policy"] for x in requests] == ["before", "shadow", "after"]
        counter_deltas = []
        for row in requests:
            start, end = counters(row["metrics_before"]), counters(row["metrics_after"])
            counter_deltas.append({k: v - start.get(k, 0) for k, v in end.items()})
        output_equal = all(
            r["result"]["token_ids"] == requests[0]["result"]["token_ids"] for r in requests
        )
        case_report: dict[str, Any] = {
            "request_outputs_unchanged": output_equal,
            "matches_original_speed_output": requests[0]["result"]["token_ids"] == next(
                x["histories"]["ct_mxfp8"]["token_ids"] for x in inputs["cases"]
                if x["case"] == case
            ),
            "request_counters_unchanged": counter_deltas[0] == counter_deltas[1] == counter_deltas[2],
            "counter_deltas": counter_deltas,
            "ranks": {},
        }
        common_payloads: dict[int, list[dict[str, str]]] = {}
        for rank in (0, 1):
            folder = root / f"head-r{rank}"
            installed = json.loads((folder / f"probe-installed-r{rank}.json").read_bytes())
            assert installed["dense_head_matches_prepared_source"] is True
            frames = sorted(folder.glob(f"frame-{case}-*-r{rank}.json"))
            assert frames
            result: dict[str, Any] = {
                "frames": len(frames), "rows": 7 * len(frames),
                "topk_order_changed_rows": 0, "topk_set_changed_rows": 0,
                "proposal_changed_frames": 0, "proposal_changed_tokens": 0,
                "projection_repeat_failures": [], "cpu_walk_failures": [],
                "graph_shadow_failures": [], "actual_mutation_failures": [],
                "raw_sampled_frames": 0, "raw_sampled_elements": 0,
                "raw_sampled_bit_differences": 0, "raw_sampled_max_abs": 0.0,
                "changed_proposals": [], "payloads": [],
            }
            common_payloads[rank] = []
            starts = []
            for index, path in enumerate(frames):
                frame = json.loads(path.read_bytes())
                assert frame["index"] == index and frame["case"] == case and frame["rank"] == rank
                tensor_path = path.with_suffix(".safetensors")
                digest = hashlib.sha256(tensor_path.read_bytes()).hexdigest()
                assert digest == frame["payload_sha256"]
                tensors = load_file(tensor_path)
                starts.append(int(tensors["sample_positions"][0]))
                graph_equal = all(
                    np.array_equal(tensors["graph_" + kind], tensors["packed_" + shadow])
                    for kind, shadow in (("ids", "ids"), ("values", "values"),
                                         ("scores", "shadow_scores"))
                )
                if not graph_equal or not all(frame["graph_shadow_exact"].values()):
                    result["graph_shadow_failures"].append(index)
                untouched = (
                    np.array_equal(tensors["after_proposal"], tensors["packed_proposal"])
                    and np.array_equal(tensors["after_realized"], tensors["packed_realized"])
                )
                if not untouched or not all(frame["actual_outputs_untouched"].values()):
                    result["actual_mutation_failures"].append(index)
                result["payloads"].append({"file": tensor_path.name, "sha256": digest})
                common = {}
                for name in ("packed", "dense"):
                    ids = tensors[name + "_ids"]
                    values = tensors[name + "_values"]
                    exact = (
                        np.array_equal(ids, tensors[name + "_ids_repeat"])
                        and np.array_equal(values, tensors[name + "_values_repeat"])
                        and frame["raw_sha256"][name] == frame["raw_sha256"][name + "_repeat"]
                    )
                    if not exact or not all(
                        frame["repeats"][name + "_" + k] for k in ("ids", "values", "raw")
                    ):
                        result["projection_repeat_failures"].append({"frame": index, "head": name})
                    proposal = tensors[name + "_proposal"].reshape(7).tolist()
                    scores = tensors[name + "_scores"]
                    if greedy_path(ids, scores) != proposal:
                        result["cpu_walk_failures"].append({"frame": index, "head": name})
                    common[name] = hashlib.sha256(
                        ids.tobytes() + values.tobytes() + scores.tobytes()
                        + tensors[name + "_proposal"].tobytes()
                    ).hexdigest()
                common_payloads[rank].append(common)
                packed, dense = tensors["packed_ids"], tensors["dense_ids"]
                result["topk_order_changed_rows"] += int(np.any(packed != dense, axis=1).sum())
                result["topk_set_changed_rows"] += int(
                    np.any(np.sort(packed, axis=1) != np.sort(dense, axis=1), axis=1).sum()
                )
                changed = int(
                    (tensors["packed_proposal"] != tensors["dense_proposal"]).sum()
                )
                result["proposal_changed_tokens"] += changed
                result["proposal_changed_frames"] += bool(changed)
                if changed:
                    result["changed_proposals"].append({
                        "frame": index,
                        "positions": tensors["sample_positions"].tolist(),
                        "packed": tensors["packed_proposal"].reshape(7).tolist(),
                        "dense": tensors["dense_proposal"].reshape(7).tolist(),
                    })
                if frame["full_raw_retained"]:
                    a, b = tensors["packed_raw"], tensors["dense_raw"]
                    assert list(a.shape) == frame["raw_shape"]
                    assert hashlib.sha256(a.tobytes()).hexdigest() == frame["raw_sha256"]["packed"]
                    assert hashlib.sha256(b.tobytes()).hexdigest() == frame["raw_sha256"]["dense"]
                    assert frame["tensor_dtypes"]["packed_raw"] == "torch.bfloat16"
                    af, bf = bf16_values(a), bf16_values(b)
                    assert np.isfinite(af).all() and np.isfinite(bf).all() and frame["raw_finite"]
                    differences = int((a != b).sum())
                    maximum = float(np.abs(af - bf).max())
                    assert differences == frame["raw_bit_differences"]
                    assert maximum == frame["raw_max_abs"]
                    result["raw_sampled_frames"] += 1
                    result["raw_sampled_elements"] += a.size
                    result["raw_sampled_bit_differences"] += differences
                    result["raw_sampled_max_abs"] = max(result["raw_sampled_max_abs"], maximum)
            draft_count = sum(
                v for k, v in counter_deltas[1].items()
                if k.split("{", 1)[0] == "vllm:spec_decode_num_drafts_total"
            )
            accepted_total = sum(
                v for k, v in counter_deltas[1].items()
                if k.split("{", 1)[0] == "vllm:spec_decode_num_accepted_tokens_total"
            )
            per_position = [0] * 7
            for key, value in counter_deltas[1].items():
                if key.startswith("vllm:spec_decode_num_accepted_tokens_per_pos_total{"):
                    match = re.search(r'position="(\d+)"', key)
                    assert match is not None
                    per_position[int(match[1])] += int(value)
            result["coverage"] = verify_round_coverage(
                starts, requests[1]["result"]["usage"]["prompt_tokens"], 256,
                int(draft_count), int(accepted_total), per_position,
            )
            result["all_draft_rounds_captured"] = result["coverage"]["passed"]
            case_report["ranks"][str(rank)] = result
        case_report["replicated_topk_scores_proposals_equal"] = common_payloads[0] == common_payloads[1]
        report["cases"][case] = case_report
    report["controls_pass"] = all(
        row["request_outputs_unchanged"] and row["request_counters_unchanged"]
        and row["matches_original_speed_output"]
        and row["replicated_topk_scores_proposals_equal"]
        and all(
            not rank["projection_repeat_failures"] and not rank["cpu_walk_failures"]
            and not rank["graph_shadow_failures"] and not rank["actual_mutation_failures"]
            and rank["all_draft_rounds_captured"]
            for rank in row["ranks"].values()
        )
        for row in report["cases"].values()
    )
    return report


if __name__ == "__main__":
    import sys

    directory = Path(sys.argv[1])
    result = analyze(directory)
    (directory / "head-review.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "controls_pass": result["controls_pass"],
        "cases": {
            case: {key: value for key, value in row["ranks"]["0"].items()
                   if key not in ("payloads", "changed_proposals")}
            for case, row in result["cases"].items()
        },
    }))
