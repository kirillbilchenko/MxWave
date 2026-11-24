"""Retain the original model gates and require evidence of the intended input-only route."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import analyze_warmed_model as warmed
from benchmark_full_model import write
from fp32_input_policy import policy

INTEGRATION_ARCHIVE_SHA256 = "a7b9f74fa1df33108caf61cc65db4c63997bbcf8e9dd378f4890efc65f1b5e80"


def validate_policy(data: dict[str, Any], arm: str) -> None:
    """Require both workers to attest to the frozen adapter at every observed model stage."""
    if arm not in ("control", "candidate"):
        raise ValueError("Undeclared comparison arm")
    expected = {**policy(), "enabled": arm == "candidate"}
    for stage in ("snapshot_before", "snapshot_before_profiles", "snapshot_after"):
        samples = data[stage]
        if len(samples) != 2 or {s["rank"] for s in samples} != {0, 1}:
            raise ValueError("Require both ranks for FP32 model policy attestation")
        if any(sample.get("fp32_input_policy") != expected for sample in samples):
            raise ValueError("Observed FP32 input policy differs from the frozen experiment")


def verify_decode_mechanism(profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Require FP32 input, original output and reduction calls inside both-rank graph replays."""
    rows = []
    for profile in profiles:
        labels = profile["labels"]
        if len(labels) != 1 or not labels[0].startswith(("control-", "candidate-")):
            raise ValueError("Unbound profile label")
        arm = "candidate" if labels[0].startswith("candidate-") else "control"
        expected = 8 if "-c8-" in labels[0] else 16
        selected = [s for s in profile["steps"] if not s["batch"]["has_prefill"]
                    and s["batch"]["num_tokens_after_padding"] == expected]
        if not selected:
            raise ValueError("Missing expected C8/C16 decode shape")
        for step in selected:
            variants = step["kernel_variants"]
            rebuild = sum(v["count"] for v in variants
                          if "bf16triplebitmap_decompress_kernel" in v["name"].lower())
            original = [v for v in variants if "BF16TripleBitmap_MM_Kernel" in v["name"]]
            fp32 = [v for v in variants if "BF16TripleBitmap_MM_FP32_Kernel" in v["name"]]
            reducers = sum(v["count"] for v in variants if "MxWaveReduceFP32" in v["name"])
            counts = sum(v["count"] for v in original), sum(v["count"] for v in fp32), reducers
            if rebuild or step["cuda_graph_launches"] <= 0:
                raise ValueError("Decode reconstructed weights or skipped graph replay")
            if counts != ((34, 34, 34) if arm == "candidate" else (0, 0, 0)):
                raise ValueError("Actual model input/output kernel route differs from the plan")
            if arm == "candidate" and (
                any(v["grid"][1] != 64 for v in original)
                # Native launch: padded output rows * Split_K / TILE_M = 12608 * 2 / 64.
                or any(v["grid"][1] != 394 for v in fp32)
            ):
                raise ValueError("Input or output projection geometry changed")
            rows.append({"arm": arm, "rank": profile["rank"], "label": step["label"],
                "sequence": step["sequence"], "actual_rows": step["batch"]["num_tokens"],
                "physical_rows": expected, "graph_launches": step["cuda_graph_launches"],
                "reconstruction_calls": rebuild, "original_output_calls": counts[0],
                "FP32_input_calls": counts[1], "FP32_reduction_calls": counts[2]})
    covered = {(r["arm"], r["rank"], r["physical_rows"]) for r in rows}
    if covered != {(a, r, m) for a in ("control", "candidate") for r in (0, 1) for m in (8, 16)}:
        raise ValueError("Missing an arm, rank or physical row count in model profiles")
    return rows


def review(root: Path) -> dict[str, Any]:
    """Recompute all 46 numerical comparisons and all original speed decisions unchanged."""
    plan = json.loads((root / "warmed-model-plan.json").read_bytes())
    if plan.get("fp32_input_integration_archive_sha256") != INTEGRATION_ARCHIVE_SHA256:
        raise ValueError("Missing independently verified input-adapter parent")
    if plan.get("fp32_input_policy") != policy():
        raise ValueError("Frozen input policy changed")
    for arm in ("control", "candidate"):
        validate_policy(json.loads((root / f"{arm}-warmed-model.json").read_bytes()), arm)
    result = warmed.review(root)
    result.update(fp32_input_policy_checked_on_both_ranks=True,
                  fp32_input_integration_archive_sha256=INTEGRATION_ARCHIVE_SHA256,
                  fp32_parent_exact_word_screen_passed=False,
                  fp32_parent_input_timing_screen_passed=False,
                  fp32_input_only=True)
    return result


def main() -> None:
    """Write the fresh model review while preserving failed historical and current gates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    result = review(args.root)
    write(args.root / "quality-model-review.json", result)
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("numerical_comparisons", "CPU_memory_observations")}))


if __name__ == "__main__":
    main()
