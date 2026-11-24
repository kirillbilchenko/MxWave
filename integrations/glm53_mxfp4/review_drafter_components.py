"""Independently compare component payloads on shared native verification rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .drafter_fp8_roundtrip import digest
from .review_drafter_verification import difference, frames, numeric
from .review_drafter_verification import review as review_native


def active_value(frame: dict[str, Any], event: dict[str, Any], leaf: dict[str, Any]) -> Any:
    """Trim dispatch capacity using the actual padded-token count, retaining all raw evidence."""
    value = frame["x"][leaf["payload_key"]]
    if (".moe_gemm" not in event["name"] or not event["name"].endswith("/input")
            or leaf["path"] not in ("/11", "/12")):
        return value
    count_leaf = next(x for x in event["tensors"] if x["path"] == "/13")
    count = int(frame["x"][count_leaf["payload_key"]].reshape(-1)[0])
    block = int(event["parameters"]["moe_block_size"])
    assert count >= 0 and block > 0 and count % block == 0
    length = count if leaf["path"] == "/11" else count // block
    assert value.ndim == 1 and length <= len(value)
    return value[:length]


def row_view(value: np.ndarray, event: dict[str, Any], leaf: dict[str, Any]) -> Any:
    """Select only known eight-row activations; do not reinterpret sorted expert indices."""
    if event["category"] == "workspace" or value.ndim == 0:
        return None
    # Sorted token/expert IDs describe the complete dispatch, not a per-token row.
    if (".moe_gemm" in event["name"] and event["name"].endswith("/input")
            and leaf["path"] != "/0"):
        return None
    # Installed marlin_moe.py returns flat token × top-k results and reduces
    # output.view(-1, topk, K) over routes (source hash retained in the report).
    if (".moe_gemm" in event["name"] and value.ndim == 2 and value.shape[0] == 64
            and event["parameters"].get("size_m") in (8, 64)):
        return value.reshape(8, 8, value.shape[-1])
    if value.shape[0] != 8:
        return None
    return value


def compare(a: dict[str, Any], b: dict[str, Any], shared: int) -> dict[str, Any]:
    """Compare chronological operations and identify the first changed shared-row result."""
    am, bm = a["meta"]["components"], b["meta"]["components"]
    assert am == bm, "Graph layouts differ"
    assert am["python_wrappers_removed"] and am["capture_bytes"] <= 64 * 1024**2
    events = []
    first = None
    for event in am["events"]:
        leaves = []
        for leaf in event["tensors"]:
            key = leaf["payload_key"]
            x, y = active_value(a, event, leaf), active_value(b, event, leaf)
            row = {"path": leaf["path"], "stored_shape": list(a["x"][key].shape),
                   **difference(x, y)}
            xx, yy = row_view(x, event, leaf), row_view(y, event, leaf)
            if xx is not None and yy is not None:
                metrics = difference(xx[:shared], yy[:shared])
                metrics["different_elements_per_row"] = [
                    int(np.count_nonzero(np.any(
                        np.ascontiguousarray(xx[i]).view(np.uint8).reshape(-1, xx.itemsize)
                        != np.ascontiguousarray(yy[i]).view(np.uint8).reshape(-1, yy.itemsize),
                        axis=1,
                    ))) for i in range(shared)
                ]
                if leaf["dtype"] in ("torch.bfloat16", "torch.float32"):
                    nx = numeric(xx[:shared], leaf["dtype"]).astype(np.float64)
                    ny = numeric(yy[:shared], leaf["dtype"]).astype(np.float64)
                    metrics["finite"] = bool(np.isfinite(nx).all() and np.isfinite(ny).all())
                    if metrics["finite"]:
                        metrics["max_abs"] = float(np.max(np.abs(nx - ny), initial=0))
                row["shared_rows"] = metrics
                if not metrics["bitwise_equal"] and event["category"] == "result" and first is None:
                    first = {"event": event["name"], "path": leaf["path"],
                             "shared_rows": metrics}
            leaves.append(row)
        events.append({k: v for k, v in event.items() if k != "tensors"} | {"tensors": leaves})
    return {
        "start_position": int(a["x"]["target_positions"][0]), "shared_rows": shared,
        "capture_bytes": am["capture_bytes"], "first_changed_shared_row_result": first,
        "events": events,
    }


def review(root: Path, output: Path) -> dict[str, Any]:
    """Require native controls and restoration, then check component repeats and both ranks."""
    if output.exists():
        raise FileExistsError(output)
    result = review_native(root, None)
    assert all(row["inputs_logits_features_exact"] for row in result["capture_repeats"])
    plan = json.loads((root / "plan.json").read_bytes())
    case = plan["calibration_cases"][0]["id"]
    assert plan["revision"] == "drafter-components-v2"
    retained = {}
    repeats = []
    for arm in plan["arms"]:
        measured = json.loads((root / f"{arm}-measurement.json").read_bytes())
        assert all(row["counter_delta"] == plan["expected_counter_deltas"][arm]
                   for row in measured["cases"])
        for rank in (0, 1):
            for policy in ("capture1", "capture2"):
                records = frames(root, arm, case, policy, rank)
                selected = {int(r["x"]["target_positions"][0]): r for r in records
                            if r["meta"]["components_retained"]}
                assert sorted(selected) == plan["component_start_positions"]
                retained[arm, rank, policy] = selected
            left, right = (retained[arm, rank, p] for p in ("capture1", "capture2"))
            for position in plan["component_start_positions"]:
                a, b = left[position], right[position]
                assert a["meta"]["components"] == b["meta"]["components"]
                changed = []
                for event in a["meta"]["components"]["events"]:
                    if event["category"] == "workspace":
                        continue
                    for leaf in event["tensors"]:
                        x, y = active_value(a, event, leaf), active_value(b, event, leaf)
                        if x.shape != y.shape or x.tobytes() != y.tobytes():
                            changed.append({"event": event["name"], "path": leaf["path"]})
                repeats.append({"arm": arm, "rank": rank, "start_position": position,
                                "changed_nonworkspace_tensors": changed})
    comparisons = []
    for rank in (0, 1):
        for position in plan["component_start_positions"]:
            a, b = (retained[arm, rank, "capture1"][position] for arm in ("a1", "b1"))
            assert np.array_equal(a["x"]["target_positions"], b["x"]["target_positions"])
            shared = 0
            for x, y in zip(a["x"]["target_input_ids"], b["x"]["target_input_ids"], strict=True):
                if x != y:
                    break
                shared += 1
            assert shared == (8 if position == 46 else 7)
            comparisons.append({"rank": rank, **compare(a, b, shared)})
    result.update(
        source_sha256=digest(Path(__file__)),
        component_repeats=repeats, component_comparisons=comparisons,
        recurrent_read_rule=None,
        expanded_expert_row_mapping={
            "source": "vllm/model_executor/layers/fused_moe/experts/marlin_moe.py",
            "source_sha256": "f00460310b033bcb84644677e16af83773464f026c0dad15bcfe25bd28b39745",
            "rule": "flat token × eight routes, then output.view(-1, topk, K).sum(dim=1)",
        },
        feature_labels={
            "source": "vllm/v1/worker/gpu/spec_decode/eagle/eagle3_utils.py",
            "source_sha256": "43e5ba8d3997712878ead63e9d5d5bc7c3e656650b8aa6a41b3e19f84d2f6487",
            "rule": "DFlash zero-based target_layer_ids become id+1 boundaries; aux_layer_5 follows decoder 5.",
        },
        limits=[
            "One development prompt, one load per arm; no speed or quality qualification.",
            "Shared rows require identical accepted history; this is not proof of causal leakage.",
            "Only early layers 0–4 are instrumented; later propagation is not localized.",
            "Sorted dispatch indices are not treated as per-token activations.",
            "Dispatch buffers use their recorded padded-token count; unused allocation tails are excluded.",
            "Uninitialized output workspace is excluded from repeat and first-difference conclusions.",
            "No recurrent cache is retained in this run; selected starting states were checked in V2.",
            "First changed observed result still needs a direct component control before root-cause attribution.",
        ],
    )
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    """Review completed retained component evidence without contacting a server."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = review(args.root, args.output)
    print(json.dumps({
        "status": result["status"], "payloads_verified": result["payloads_verified"],
        "component_repeats": result["component_repeats"],
        "first_differences": [{k: v for k, v in row.items() if k != "events"}
                              for row in result["component_comparisons"]],
    }, indent=2))


if __name__ == "__main__":
    main()
