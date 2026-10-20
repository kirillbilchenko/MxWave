"""Compare retained native traces only where their accepted input histories agree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .audit_drafter_feature_export_numpy import arrays
from .drafter_fp8_roundtrip import digest


def numeric(value: np.ndarray, dtype: str) -> np.ndarray:
    """Decode stored BF16 bits without interpreting them as integer measurements."""
    if dtype == "torch.bfloat16":
        return (value.astype(np.uint32) << 16).view(np.float32)
    return np.asarray(value)


def difference(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    """Report exact payload equality and mismatching elements for equal shapes."""
    if a.shape != b.shape:
        return {"same_shape": False, "shapes": [list(a.shape), list(b.shape)]}
    count = int(np.count_nonzero(a != b))
    return {"same_shape": True, "elements": a.size, "different_elements": count,
            "bitwise_equal": a.tobytes() == b.tobytes()}


def logit_comparison(a: np.ndarray, b: np.ndarray, tokens: list[int]) -> dict[str, Any]:
    """Compare full finite logits, preserving ties rather than claiming a unique winner."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    if a.shape != b.shape or a.ndim != 1 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Expected equal finite vocabulary logit rows")
    la = a - a.max()
    lb = b - b.max()
    la -= np.log(np.exp(la).sum())
    lb -= np.log(np.exp(lb).sum())
    return {
        **difference(a, b), "max_abs": float(np.abs(a - b).max()),
        "forward_kl_a_to_b": float(np.sum(np.exp(la) * (la - lb))),
        "a_argmax_ids": np.flatnonzero(a == a.max()).tolist(),
        "b_argmax_ids": np.flatnonzero(b == b.max()).tolist(),
        "selected_token_logits": {str(t): {"a": float(a[t]), "b": float(b[t])} for t in tokens},
    }


def frames(root: Path, arm: str, case: str, policy: str, rank: int) -> list[dict[str, Any]]:
    """Load verified frame maps and reconstruct the actually sampled output offsets."""
    result = []
    emitted: list[int] = []
    for path in sorted((root / f"{arm}-r{rank}").glob(
        f"context-{case}-{policy}-*-r{rank}.safetensors",
    )):
        meta = json.loads(path.with_suffix(".json").read_bytes())
        if digest(path) != meta["payload_sha256"]:
            raise ValueError(f"Frame checksum differs: {path}")
        x = arrays(path)
        count = int(x["target_num_sampled"].reshape(-1)[0])
        sampled = x["target_sampled_tokens"].reshape(-1)[:count].tolist()
        result.append({"path": path.name, "x": x, "meta": meta,
                       "output_start": len(emitted), "sampled": sampled})
        emitted.extend(sampled)
    if not result:
        raise ValueError("Missing capture frames")
    return result


def recurrent_inputs(frame: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Select the recurrent slot read by the audited native speculative kernel."""
    result = {}
    for layer, meta in frame["meta"]["state_layers"].items():
        if not meta["num_spec_decodes"]:
            continue
        x = frame["x"]
        prefix = f"before_layer_{layer}_"
        accepted = int(x[prefix + "meta_num_accepted_tokens"].reshape(-1)[0])
        slots = x[prefix + "meta_spec_state_indices_tensor"][0]
        if not 1 <= accepted <= len(slots):
            raise ValueError("Invalid accepted-state column")
        physical = int(slots[accepted - 1])
        if physical <= 0:
            raise ValueError("Native recurrent kernel would skip an invalid state")
        retained_row = meta["physical_state_slots"].index(physical)
        name = prefix + "cache_1"
        result[layer] = {
            "accepted_column": accepted - 1, "physical_slot": physical,
            "value": x[name][retained_row], "dtype": frame["meta"]["tensor_dtypes"][name],
        }
    return result


def compare_recurrent_inputs(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Compare actual read states, ignoring unused speculative cache columns."""
    aa, bb = recurrent_inputs(a), recurrent_inputs(b)
    result = {}
    for layer in sorted(aa.keys() & bb.keys()):
        left, right = aa[layer], bb[layer]
        result[layer] = {
            **difference(left["value"], right["value"]),
            "accepted_columns": [left["accepted_column"], right["accepted_column"]],
            "physical_slots": [left["physical_slot"], right["physical_slot"]],
        }
        if left["value"].shape == right["value"].shape:
            x = numeric(left["value"], left["dtype"]).astype(np.float64)
            y = numeric(right["value"], right["dtype"]).astype(np.float64)
            finite = bool(np.isfinite(x).all() and np.isfinite(y).all())
            result[layer]["finite"] = finite
            if finite:
                result[layer]["max_abs"] = float(np.abs(x - y).max())
    return result


def compare_frames(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Compare causally shared rows; future rejected tokens are explicitly excluded."""
    x, y = a["x"], b["x"]
    n = min(len(x["target_input_ids"]), len(y["target_input_ids"]))
    shared = 0
    for i in range(n):
        if (x["target_input_ids"][i] != y["target_input_ids"][i]
                or x["target_positions"][i] != y["target_positions"][i]):
            break
        shared += 1
    names = ["target_final_hidden", *[f"aux_layer_{i}" for i in (5, 14, 24, 33, 42)]]
    result = {
        "a_frame": a["path"], "b_frame": b["path"], "shared_input_rows": shared,
        "start_position": int(x["target_positions"][0]),
        "all_input_ids_equal": np.array_equal(x["target_input_ids"], y["target_input_ids"]),
        "input_ids": [x["target_input_ids"].tolist(), y["target_input_ids"].tolist()],
        "positions": [x["target_positions"].tolist(), y["target_positions"].tolist()],
        "shared_row_tensors": {
            name: {
                **difference(x[name][:shared], y[name][:shared]),
                "different_elements_per_row": [
                    int(np.count_nonzero(x[name][i] != y[name][i])) for i in range(shared)
                ],
            } for name in names
        },
        "before_num_computed": [x["before_num_computed"].tolist(),
                                y["before_num_computed"].tolist()],
        "before_num_accepted": [x["before_num_accepted"].tolist(),
                               y["before_num_accepted"].tolist()],
        "selected_recurrent_inputs": compare_recurrent_inputs(a, b),
    }
    return result


def review(root: Path, output: Path | None) -> dict[str, Any]:
    """Validate controls, restoration and hashes before locating the first changed decision."""
    if output is not None and output.exists():
        raise FileExistsError(output)
    manifest = json.loads((root / "evidence-manifest.json").read_bytes())
    for row in manifest["files"]:
        path = root / row["path"]
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Evidence path escapes root")
        if digest(path) != row["sha256"] or path.stat().st_size != row["bytes"]:
            raise ValueError(f"Evidence integrity failure: {row['path']}")
    state = json.loads((root / "controller.json").read_bytes())
    if state["phase"] != "complete_restored" or state["service_paused"]:
        raise ValueError("Successful run and restoration required")
    restore = state["restoration"]
    assert restore["health_200"] and restore["originals_running"]
    assert any(m["id"] == "glm-5.3-flash" and m["max_model_len"] == 500000
               for m in restore["models"]["data"])
    plan = json.loads((root / "plan.json").read_bytes())
    prepared = json.loads((root / "prepared.json").read_bytes())
    assert digest(root / "plan.json") == prepared["plan_sha256"]
    case = plan["calibration_cases"][0]["id"]
    captures = {}
    controls = {}
    for arm in plan["arms"]:
        measured = json.loads((root / f"{arm}-measurement.json").read_bytes())
        assert measured["status"] == "complete"
        rows = measured["cases"]
        assert [r["policy"] for r in rows] == ["before", "capture1", "capture2", "after"]
        assert all(r["result"]["token_ids"] == plan["expected_outputs"][arm]
                   and r["counter_delta"] == rows[0]["counter_delta"] for r in rows)
        controls[arm] = {"original_output_and_counter_controls_pass": True}
        for rank in (0, 1):
            expected = json.loads((root / f"expected-runtime-r{rank}.json").read_bytes())
            expected["model.fc.weight"]["sha256"] = plan["context_payloads"][plan["variants"][arm]]
            assert expected == json.loads(
                (root / f"{arm}-r{rank}/draft-runtime-weights-r{rank}.json").read_bytes())
            for policy in ("capture1", "capture2"):
                records = frames(root, arm, case, policy, rank)
                sampled = [t for r in records for t in r["sampled"]]
                assert sampled[:32] == plan["expected_outputs"][arm]
                captures[(arm, policy, rank)] = records
    repeats = []
    for arm in plan["arms"]:
        for rank in (0, 1):
            left, right = captures[(arm, "capture1", rank)], captures[(arm, "capture2", rank)]
            assert len(left) == len(right)
            names = ["target_input_ids", "target_positions", "target_logits",
                     "target_logits_after_sampler", "target_final_hidden",
                     *[f"aux_layer_{i}" for i in (5, 14, 24, 33, 42)]]
            equal = all(np.array_equal(a["x"][name], b["x"][name])
                        for a, b in zip(left, right, strict=True) for name in names)
            repeats.append({"arm": arm, "rank": rank, "frames": len(left),
                            "inputs_logits_features_exact": equal})
    first = next(i for i, (a, b) in enumerate(zip(
        plan["expected_outputs"]["a1"], plan["expected_outputs"]["b1"], strict=True,
    )) if a != b)
    rank_results = []
    for rank in (0, 1):
        aa, bb = captures[("a1", "capture1", rank)], captures[("b1", "capture1", rank)]
        shared_frames = []
        for a, b in zip(aa, bb):
            if (a["output_start"] != b["output_start"] or a["output_start"] > first):
                break
            shared_frames.append(compare_frames(a, b))
        chosen = []
        for records in (aa, bb):
            chosen.append(next(r for r in records
                               if r["output_start"] <= first < r["output_start"] + len(r["sampled"])))
        a, b = chosen
        offsets = [first - r["output_start"] for r in chosen]
        logits = [numeric(r["x"]["target_logits"][offset],
                          r["meta"]["tensor_dtypes"]["target_logits"])
                  for r, offset in zip(chosen, offsets, strict=True)]
        tokens = [plan["expected_outputs"][arm][first] for arm in ("a1", "b1")]
        decision = {
            "frames": [r["path"] for r in chosen], "logit_row_offsets": offsets,
            "sampled_tokens": tokens, "pre_sampler_logits": logit_comparison(*logits, tokens),
            "after_sampler_selected_logits": [
                {str(t): float(numeric(r["x"]["target_logits_after_sampler"][offset],
                               r["meta"]["tensor_dtypes"]["target_logits_after_sampler"])[t])
                 for t in tokens} for r, offset in zip(chosen, offsets, strict=True)
            ],
        }
        rank_results.append({"rank": rank, "shared_frames": shared_frames,
                             "first_different_decision": decision})
    result = {
        "status": "verified", "source_sha256": digest(Path(__file__)),
        "plan_sha256": digest(root / "plan.json"), "payloads_verified": len(manifest["files"]),
        "controls": controls, "capture_repeats": repeats,
        "first_different_output_index": first, "rank_results": rank_results,
        "production_restored": True, "sealed_test_opened": False,
        "recurrent_read_rule": {
            "column": "num_accepted_tokens - 1",
            "installed_kernel": "vllm/third_party/flash_linear_attention/ops/fused_recurrent.py",
            "audited_kernel_sha256":
                "371e7d733e90a7147fbca8dbfe67e62503ba663bd1570424bc0efa22c54f3e2b",
            "scope": "Audited installed V2 image source; selected layers 0 and 4 only.",
        },
        "limits": [
            "One development prompt and one load per arm; no speed or quality qualification.",
            "Shared row comparison is conditional on shared prior accepted history.",
            "Only target layers 5/14/24/33/42 and final hidden are retained.",
            "Recurrent states use the audited kernel's accepted-column minus one rule; unused columns are excluded.",
            "Only layers 0 and 4 have retained recurrent state; convolution-history slices are not compared.",
            "Post-sampler copies refer to the original logits storage, not an independently retained processed-logits tensor.",
            "No causal attribution to a particular kernel from boundary tensors alone.",
        ],
    }
    if output is not None:
        output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    """Review a completed archived trace without contacting serving infrastructure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = review(args.root, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "rank_results"}, indent=2))


if __name__ == "__main__":
    main()
