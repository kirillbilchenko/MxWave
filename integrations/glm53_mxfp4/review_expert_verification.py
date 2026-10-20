"""Review equal-policy target verification without opening the sealed drafter test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .drafter_fp8_roundtrip import digest
from .review_drafter_verification import (
    compare_frames,
    frames,
    logit_comparison,
    numeric,
)


def review(root: Path, output: Path) -> dict[str, Any]:
    """Require original-service restoration and exact self-controls before cross-arm comparison."""
    assert not output.exists()
    manifest = json.loads((root / "evidence-manifest.json").read_bytes())
    for item in manifest["files"]:
        path = root / item["path"]
        assert path.resolve().is_relative_to(root.resolve())
        assert path.stat().st_size == item["bytes"] and digest(path) == item["sha256"]
    state = json.loads((root / "controller.json").read_bytes())
    assert state["phase"] == "complete_restored" and not state["service_paused"]
    restored = state["restoration"]
    assert restored["health_200"] and restored["originals_running"]
    assert any(m["max_model_len"] == 500000 for m in restored["models"]["data"])
    prepared = json.loads((root / "prepared.json").read_bytes())
    assert prepared["live"] == restored["live"] and prepared["originals"] == restored["originals"]
    plan = json.loads((root / "plan.json").read_bytes())
    assert digest(root / "plan.json") == prepared["plan_sha256"]
    assert plan["expert_reduction_policy"]["same_in_both_arms"]
    assert digest(root / "component-review.json") == plan["component_review_sha256"]
    case = plan["calibration_cases"][0]["id"]
    outputs, captures, repeat_controls, coverage = {}, {}, [], []
    graph_controls, counters = [], {}
    for arm in plan["arms"]:
        measured = json.loads((root / f"{arm}-measurement.json").read_bytes())
        assert measured["status"] == "complete" and not measured["sealed_test_opened"]
        rows = measured["cases"]
        assert [r["policy"] for r in rows] == ["before", "capture1", "capture2", "after"]
        ids = rows[0]["result"]["token_ids"]
        assert len(ids) == 32
        assert all(r["result"]["token_ids"] == ids
                   and r["counter_delta"] == rows[0]["counter_delta"] for r in rows)
        outputs[arm] = ids
        counters[arm] = rows[0]["counter_delta"]
        for rank in (0, 1):
            startup = json.loads((root / f"{arm}-r{rank}/trial-startup-r{rank}.json").read_bytes())
            policy = startup["expert_reduction"]
            assert policy["installed"] and policy["policy"] == "bounded_padding_default_occupancy"
            shapes = {tuple(x["geometry"][:3]) for x in policy["dispatches"]}
            assert {(8, 2048, 4096), (64, 4096, 1024)} <= shapes
            coverage.append({"arm": arm, "rank": rank, **policy})
            expected = json.loads((root / f"expected-runtime-r{rank}.json").read_bytes())
            expected["model.fc.weight"]["sha256"] = plan["context_payloads"][plan["variants"][arm]]
            assert expected == json.loads(
                (root / f"{arm}-r{rank}/draft-runtime-weights-r{rank}.json").read_bytes())
            for capture in ("capture1", "capture2"):
                records = frames(root, arm, case, capture, rank)
                assert [t for f in records for t in f["sampled"]][:32] == ids
                decode = [f for f in records if f["meta"]["target_num_draft_tokens"]]
                assert decode and all(
                    f["meta"]["target_graph_mode"] == "FULL"
                    and f["meta"]["target_tokens"] == f["meta"]["target_padded_tokens"] == 8
                    and f["meta"]["target_num_draft_tokens"] == 7
                    for f in decode
                )
                positions = [
                    int(f["x"]["target_positions"][0]) for f in decode
                    if f["meta"]["components_retained"]
                ]
                assert positions == [46, 54]
                graph_controls.append({
                    "arm": arm, "rank": rank, "capture": capture,
                    "full_target_graph_frames": len(decode),
                    "component_capture_positions": positions,
                })
                captures[arm, rank, capture] = records
            a, b = (captures[arm, rank, c] for c in ("capture1", "capture2"))
            names = ["target_input_ids", "target_positions", "target_logits",
                     "target_logits_after_sampler", "target_final_hidden",
                     *[f"aux_layer_{i}" for i in (5, 14, 24, 33, 42)]]
            assert len(a) == len(b)
            assert all(x["x"][name].tobytes() == y["x"][name].tobytes()
                       for x, y in zip(a, b, strict=True) for name in names)
            repeat_controls.append({"arm": arm, "rank": rank, "frames": len(a), "exact": True})
    first = next((i for i, (a, b) in enumerate(zip(outputs["a1"], outputs["b1"], strict=True))
                  if a != b), None)
    history_limit = 31 if first is None else first
    rank_results = []
    for rank in (0, 1):
        aa, bb = (captures[arm, rank, "capture1"] for arm in ("a1", "b1"))
        decisions, shared_frames = [], []
        for offset in range(history_limit + 1):
            chosen = [next(f for f in records if f["output_start"] <= offset <
                           f["output_start"] + len(f["sampled"])) for records in (aa, bb)]
            logits = [numeric(f["x"]["target_logits"][offset - f["output_start"]],
                              f["meta"]["tensor_dtypes"]["target_logits"]) for f in chosen]
            tokens = [outputs[arm][offset] for arm in ("a1", "b1")]
            decisions.append({"output_index": offset, "frames": [f["path"] for f in chosen],
                              **logit_comparison(*logits, tokens)})
        right = {f["output_start"]: f for f in bb}
        for a in aa:
            start = a["output_start"]
            if start <= history_limit and start in right:
                shared_frames.append(compare_frames(a, right[start]))
        kl = [d["forward_kl_a_to_b"] for d in decisions]
        rank_results.append({
            "rank": rank, "decisions_with_shared_history": decisions, "shared_frames": shared_frames,
            "max_kl": max(kl), "mean_kl": float(np.mean(kl)),
            "all_shared_decision_logits_exact": all(d["bitwise_equal"] for d in decisions),
            "all_aligned_shared_history_frame_inputs_equal": all(
                f["all_input_ids_equal"] and f["positions"][0] == f["positions"][1]
                for f in shared_frames
            ),
        })
    result = {
        "status": "verified", "payloads_verified": len(manifest["files"]),
        "source_sha256": digest(Path(__file__)), "plan_sha256": digest(root / "plan.json"),
        "cross_drafter_output_ids_exact": first is None, "first_different_output_index": first,
        "capture_repeat_controls": repeat_controls, "policy_capture_coverage": coverage,
        "observed_graph_controls": graph_controls,
        "cross_drafter_speculative_counters_equal": counters["a1"] == counters["b1"],
        "historical_native_output_changes": {
            arm: [i for i, (a, b) in enumerate(zip(ids, plan["historical_native_outputs"][arm],
                                                   strict=True)) if a != b]
            for arm, ids in outputs.items()
        },
        "rank_results": rank_results, "production_restored": True, "sealed_test_opened": False,
        "limits": [
            "One development prompt and one load per arm; not frozen drafter qualification.",
            "No serving speed claim; component timing failures remain retained.",
            "Decision comparisons stop after the first changed greedy token.",
            "Padding capture coverage and observed graph replay do not prove every batch size invariant.",
            "If both drafters propose identical inputs, this run does not stress differing future proposals.",
        ],
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    """Review archived full-model numerical verification independently on CPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = review(args.root, args.output)
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("rank_results", "policy_capture_coverage")}, indent=2))


if __name__ == "__main__":
    main()
