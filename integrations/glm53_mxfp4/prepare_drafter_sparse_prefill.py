"""Freeze focused sparse-score and selection evidence after equal-input indexer drift."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from .drafter_fp8_roundtrip import digest


def prepare(template: Path, review: Path, output: Path) -> dict[str, Any]:
    """Keep the target policy fixed and capture operands for membership/order analysis."""
    checked = json.loads(review.read_bytes())
    if (checked["status"] != "reviewed" or not checked["production_restored"]
            or checked["plan_sha256"] != digest(template / "plan.json")
            or len(checked["pairs"]) != 12):
        raise ValueError("Require the completed independent early-prefill review")
    for row in checked["pairs"]:
        first = row["boundaries"]["first_different_observation"]
        if (first is None or not first["name"].endswith(".indexer.indexer_op#0/output")
                or first["matching_explicit_inputs"] is not True):
            raise ValueError("Require matching explicit inputs at the first differing indexer")
    manifest = json.loads((template / "source-manifest.json").read_bytes())
    if not all(digest(template / name) == sha for name, sha in manifest.items()):
        raise ValueError("Early-prefill diagnostic sources changed")
    output.mkdir(parents=True, exist_ok=False)
    for name in manifest:
        shutil.copyfile(template / name, output / name)
    source = Path(__file__).with_name("drafter_sparse_prefill_probe.py")
    text = source.read_text().replace("from .decode_boundary_capture", "from decode_boundary_capture")
    text = text.replace("from .drafter_context_probe", "from drafter_context_probe")
    (output / source.name).write_text(text)
    shutil.copyfile(review, output / "parent-boundary-review.json")
    plan = json.loads((template / "plan.json").read_bytes())
    plan.update(
        revision="drafter-sparse-prefill-v1", private_port=8032,
        purpose="Distinguish masked score drift, selection membership and ordering at layer 3.",
        parent_plan_sha256=digest(template / "plan.json"), parent_review_sha256=digest(review),
        full_logits_frames=1, source_sha256=digest(Path(__file__)),
        notes=[
            "Same original drafter, target weights, fixed-seven draft width and runtime policies.",
            "Same three short warmups, long native before/four captures/native after, 128 output tokens.",
            "Retain raw layer-3 indexer inputs/output, gathered MQA score operands and in-place top-k output.",
            "Score buffers contain unused columns: only per-row start/end bounds define consumed values.",
            "Separate selected membership, duplicate/invalid indices, causality and index order.",
            "Stream-ordered GPU copies, bounded to 512 MiB/rank; transferred after prefill, not per operation.",
            "Observations may still alter scheduling; no arithmetic replacement or serving timing claim.",
            "Production restores on every exit; no training or sealed-test access.",
        ],
    )
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    for name in ("prepare.py", "control.py", "measure_features.py"):
        path = output / name
        path.write_text(path.read_text().replace("8031", "8032").replace("29704", "29705")
                        .replace("mxwave-drafter-prefill-boundary-v1", "mxwave-drafter-sparse-prefill-v1"))
    for name in ("trial_plugin.py", "trial_boot.py"):
        path = output / name
        text = path.read_text()
        if "drafter_prefill_boundary_probe" not in text:
            raise ValueError("Expected the frozen early-prefill probe hook")
        path.write_text(text.replace("drafter_prefill_boundary_probe", "drafter_sparse_prefill_probe"))
    for path in output.glob("*.py"):
        compile(path.read_bytes(), str(path), "exec")
    manifest = {p.name: digest(p) for p in sorted(output.iterdir())
                if p.is_file() and p.name != "source-manifest.json"}
    (output / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"plan_sha256": digest(output / "plan.json"), "rows": 3973,
            "requests": plan["requests"], "training_eligible": False, "sealed_test_opened": False}


def main() -> None:
    """Prepare the focused diagnostic locally, without changing a service or opening test data."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("template", "review", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.template, args.review, args.output), indent=2))


if __name__ == "__main__":
    main()
