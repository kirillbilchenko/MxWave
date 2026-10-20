"""Recompute pilot acceptance from exported CT tokens and checkpoint changes with NumPy."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .audit_drafter_feature_export_numpy import arrays
from .drafter_fp8_roundtrip import digest


def weight_delta(patch: Path, reference: Path) -> dict[str, float | int]:
    """Compare serialized BF16 bits and accumulate weight distances in bounded FP64 chunks."""
    actual = arrays(patch)["fc.weight"].reshape(-1)
    original = arrays(reference)["fc.weight"].reshape(-1)
    if actual.shape != original.shape:
        raise ValueError("Context checkpoint shape differs")
    changed = 0
    delta_squared = reference_squared = 0.0
    for start in range(0, len(actual), 1024 * 1024):
        a, b = actual[start:start + 1024 * 1024], original[start:start + 1024 * 1024]
        changed += int(np.count_nonzero(a != b))
        x = (a.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        y = (b.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        if not np.isfinite(x).all():
            raise ValueError("Non-finite saved context weight")
        delta_squared += float(np.square(x - y).sum())
        reference_squared += float(np.square(y).sum())
    return {"changed_elements": changed, "relative_l2": math.sqrt(delta_squared / reference_squared)}


def review(run: Path, dataset: Path, reference: Path, output: Path) -> dict[str, Any]:
    """Audit saved proposals, domain aggregation, selection and each saved BF16 checkpoint."""
    if output.exists():
        raise FileExistsError(output)
    report = json.loads((run / "report.json").read_bytes())
    if report["status"] != "complete":
        raise ValueError("Pilot is not complete")
    manifest = json.loads((dataset / "manifest.json").read_bytes())
    if digest(dataset / "manifest.json") != report["inputs"]["feature_manifest_sha256"]:
        raise ValueError("Feature manifest differs")
    cases = {r["id"]: r for r in manifest["rows"] if r["split"] == "validation"}
    fields = tokens = 0
    for evaluation in [report["baseline"], *report["evaluations"]]:
        domains: dict[str, list[float]] = {}
        for row in evaluation["rows"]:
            case = cases[row["id"]]
            path = dataset / case["path"]
            if digest(path) != case["sha256"]:
                raise ValueError("CT label file changed")
            ids = arrays(path)["input_ids"]
            positions = np.asarray(row["anchors"])[:, None] + np.arange(1, 8)[None, :]
            correct = np.asarray(row["proposal"]) == ids[positions]
            prefixes = np.cumprod(correct.astype(np.int64), axis=1).sum(1)
            if (prefixes.tolist() != row["prefixes"]
                    or float(prefixes.mean()) != row["mean_prefix"]
                    or not np.allclose(correct.mean(0), row["per_position_agreement"],
                                       rtol=0, atol=1e-7)):
                raise ValueError("Reported acceptance differs from CT token comparison")
            if case["domain"] != row["domain"]:
                raise ValueError("Prompt domain changed")
            domains.setdefault(row["domain"], []).append(float(prefixes.mean()))
            fields += 3
            tokens += correct.size
        for domain, values in domains.items():
            if float(np.mean(values)) != evaluation["domains"][domain]["mean_prefix"]:
                raise ValueError("Domain acceptance mean differs")
            fields += 1
    checkpoints = []
    eligible = []
    baseline = report["baseline"]["domains"]
    for row in report["evaluations"]:
        path = run / row["path"]
        if digest(path) != row["sha256"]:
            raise ValueError("Saved checkpoint hash differs")
        delta = weight_delta(path, reference)
        if (delta["relative_l2"] <= 0.01) != (row["relative_l2"] <= 0.01):
            raise ValueError("Corrected saved checkpoint distance changes gate eligibility")
        checkpoints.append({
            "step": row["step"], **delta, "reported_fp32_relative_l2": row["relative_l2"],
            "distance_correction_required": not math.isclose(
                delta["relative_l2"], row["relative_l2"], rel_tol=1e-3,
            ),
        })
        domains = row["domains"]
        if (set(domains) == {"code", "chat", "explanation"}
                and domains["code"]["mean_prefix"] > baseline["code"]["mean_prefix"]
                and all(domains[d]["mean_prefix"] >= 0.98 * baseline[d]["mean_prefix"]
                        for d in ("chat", "explanation"))
                and delta["relative_l2"] <= 0.01):
            eligible.append(row)
    selected = sorted(eligible, key=lambda r: (
        -r["domains"]["code"]["mean_prefix"], r["domains"]["code"]["unary_ce"], r["step"],
    ))
    selected_step = selected[0]["step"] if selected else None
    reported_step = report["selected"]["step"] if report["selected"] else None
    if selected_step != reported_step or bool(selected) != report["offline_screen_pass"]:
        raise ValueError("Frozen checkpoint selection does not reproduce")
    result = {
        "status": "verified", "report_sha256": digest(run / "report.json"),
        "source_sha256": digest(Path(__file__)), "acceptance_fields_recomputed": fields,
        "proposal_token_comparisons": tokens, "checkpoint_deltas": checkpoints,
        "selected_step": selected_step, "offline_screen_pass": bool(selected),
        "limits": "Recomputes acceptance, FP64 weight distances and selection; preserves reported FP32 distance discrepancies. Does not replay CE forwards or measure native speed.",
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    """Review a completed pilot without opening sealed data or modifying any weights."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "dataset", "reference", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(review(args.run, args.dataset, args.reference, args.output), indent=2))


if __name__ == "__main__":
    main()
