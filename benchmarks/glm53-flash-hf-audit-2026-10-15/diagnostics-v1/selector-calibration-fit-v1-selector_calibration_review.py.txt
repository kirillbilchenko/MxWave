"""Recompute the fitted projection's held-out metrics with NumPy, without PyTorch."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file


def bf16_weights(path: Path) -> dict[str, Any]:
    """Read BF16 safetensors payloads as exact IEEE float32 values."""
    data = path.read_bytes()
    size = int.from_bytes(data[:8], "little")
    header = json.loads(data[8:8 + size])
    result = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        assert info["dtype"] == "BF16"
        start, stop = info["data_offsets"]
        bits = np.frombuffer(data, dtype="<u2", count=(stop - start) // 2,
                             offset=8 + size + start)
        result[name] = (bits.astype(np.uint32) << 16).view(np.float32).reshape(info["shape"])
    return result


def prefix(proposal: list[int], gold: list[int]) -> int:
    """Count consecutive matches, stopping before the first wrong token."""
    count = 0
    for proposed, expected in zip(proposal, gold, strict=True):
        if proposed != expected:
            break
        count += 1
    return count


def inspect(capture: Path, fit: Path) -> dict[str, Any]:
    """Recompute validation and sealed-test aggregates from the actual saved BF16 patch."""
    report = json.loads((fit / "review.json").read_bytes())
    original = bf16_weights(capture / "ct-r0/selector-original.safetensors")
    patch_path = fit / "selector-projection.safetensors"
    assert hashlib.sha256(patch_path.read_bytes()).hexdigest() == report["patch_sha256"]
    patch = bf16_weights(patch_path)["model.candidate_selector.hidden_projection.weight"]
    assert np.isfinite(patch).all()
    relative_norm = float(np.linalg.norm((patch - original["projection"]).astype(np.float64))
                          / np.linalg.norm(original["projection"].astype(np.float64)))
    assert relative_norm <= 0.0101  # BF16 export can round a clipped FP32 update slightly outward.
    assert abs(relative_norm - report["relative_selected_update_norm"]) < 1e-6
    plan = json.loads((capture / "plan.json").read_bytes())
    measurement = json.loads((capture / "ct-measurement.json").read_bytes())
    outputs = {x["case"]: x["result"]["token_ids"] for x in measurement["cases"]
               if x["policy"] == "capture"}
    reviews: dict[str, Any] = {}
    for split in ("validation", "test"):
        values: dict[str, Any] = {}
        for case in plan["calibration_cases"]:
            if case["split"] != split:
                continue
            info = {"domain": case["domain"], "blocks": 0, "before_sum": 0, "after_sum": 0,
                    "native_sum": 0, "baseline_proposal_token_mismatches": 0}
            for path in sorted((capture / "ct-r0").glob(f"calibration-{case['id']}-*.safetensors")):
                tensors = load_file(path)
                start = int(tensors["sample_positions"][0]) - len(case["prompt_ids"])
                gold = outputs[case["id"]][start:start + 7]
                if len(gold) != 7:
                    continue
                hidden = (tensors["hidden"].view(np.uint16).astype(np.uint32) << 16)
                hidden = hidden.view(np.float32).reshape(7, 4096)
                ids = tensors["ids"].reshape(7, 16)
                unary = tensors["values"].reshape(7, 16).astype(np.float32)
                native = tensors["proposal"].reshape(7).tolist()
                for label, projection in (("before", original["projection"]), ("after", patch)):
                    projected = hidden @ projection.T
                    previous = int(tensors["anchors"].item())
                    proposal = []
                    for step in range(7):
                        terms = (original["successor"][ids[step]]
                                 * (original["predecessor"][previous] * projected[step]))
                        scores = unary[step] + np.sum(terms, axis=1, dtype=np.float32)
                        previous = int(ids[step, np.argmax(scores)])
                        proposal.append(previous)
                    info[label + "_sum"] += prefix(proposal, gold)
                    if label == "before":
                        info["baseline_proposal_token_mismatches"] += sum(
                            a != b for a, b in zip(proposal, native, strict=True)
                        )
                info["native_sum"] += prefix(native, gold)
                info["blocks"] += 1
            assert info["blocks"]
            values[case["id"]] = info
        aggregates = {}
        for label in ("before", "after"):
            aggregates[label] = {}
            for domain in ("all", "code", "explanation"):
                rows = [x for x in values.values() if domain == "all" or x["domain"] == domain]
                aggregates[label][domain] = sum(x[label + "_sum"] / x["blocks"] for x in rows) \
                    / len(rows)
        reviews[split] = {"prompts": values, **aggregates}
    expected = {
        "validation": {
            "before": report["validation_baseline"],
            "after": next(x["validation"] for x in report["history"]
                          if x["step"] == report["selected_step"])
            if report["selected_step"] else report["validation_baseline"],
        },
        "test": {"before": report["test_before"], "after": report["test_after"]},
    }
    compared = 0
    for split, arms in expected.items():
        for label, reference in arms.items():
            for domain in ("all", "code", "explanation"):
                assert reviews[split][label][domain] == reference[domain]
                compared += 1
            for name, data in reference["prompts"].items():
                independently = reviews[split]["prompts"][name]
                assert independently[label + "_sum"] == data["prefix_sum"]
                assert independently["blocks"] == data["blocks"]
                assert independently["native_sum"] == data["native_prefix_sum"]
                compared += 3
    before, after = reviews["test"]["before"], reviews["test"]["after"]
    passed = (report["selected_step"] > 0 and after["all"] > before["all"]
              and after["code"] >= before["code"]
              and after["explanation"] >= before["explanation"])
    assert passed == report["offline_screen_passed"]
    return {
        "schema": "mxwave-selector-fit-numpy-review-v1", "numeric_fields_verified": compared,
        "relative_selected_update_norm_float64": relative_norm, "splits": reviews,
        "offline_screen_passed": passed, "serving_qualified": False,
        "review_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "training_review_sha256": hashlib.sha256((fit / "review.json").read_bytes()).hexdigest(),
        "patch_sha256": report["patch_sha256"],
    }


if __name__ == "__main__":
    import sys

    result = inspect(Path(sys.argv[1]), Path(sys.argv[2]))
    output = Path(sys.argv[3])
    assert not output.exists()
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "numeric_fields_verified": result["numeric_fields_verified"],
        "offline_screen_passed": result["offline_screen_passed"],
    }))
