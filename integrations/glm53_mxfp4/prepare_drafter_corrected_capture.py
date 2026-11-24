"""Freeze the eight-prompt smoke under independently verified sparse ordering policies."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from .drafter_fp8_roundtrip import digest
from .prepare_drafter_fresh_capture import frozen_cases, smoke_cases


def prepare(
    template: Path, qualified: Path, review: Path, dataset: Path, output: Path,
) -> dict[str, Any]:
    """Preserve the original smoke corpus and require the completed numerical correction gate."""
    checked = json.loads(review.read_bytes())
    if (checked["status"] != "reviewed" or not checked["production_restored"]
            or not checked["component_checks_passed"] or not checked["target_repeat_passed"]
            or not checked["repeat_gate_passed"] or len(checked["target_pairs"]) != 12
            or checked["plan_sha256"] != digest(qualified / "plan.json")):
        raise ValueError("Require the independently verified restored ordering correction")
    for root in (template, qualified):
        manifest = json.loads((root / "source-manifest.json").read_bytes())
        if not all(digest(root / name) == sha for name, sha in manifest.items()):
            raise ValueError("Frozen capture or qualified runtime sources changed")
    old = json.loads((template / "plan.json").read_bytes())
    plan = json.loads((qualified / "plan.json").read_bytes())
    cases, hashes = frozen_cases(dataset)
    selected = smoke_cases(cases)
    if (old["revision"] != "drafter-fresh-capture-smoke-v3"
            or plan["revision"] != "drafter-canonical-decode-v1"
            or len(selected) != 8 or selected != old["calibration_cases"]
            or hashes != old["frozen_dataset_hashes"]):
        raise ValueError("Keep the original eight frozen training/validation prompts")
    output.mkdir(parents=True, exist_ok=False)
    for name in manifest:
        shutil.copyfile(qualified / name, output / name)
    shutil.copyfile(review, output / "ordering-qualification-review.json")
    for name in ("calibration_cases", "control_cases", "quotas", "full_logits_frames"):
        plan[name] = old[name]
    for name in ("warmup_cases", "decode_start_position"):
        plan.pop(name, None)
    policy_sources = {name: digest(qualified / name) for name in (
        "canonical_sparse_prefill.py", "canonical_sparse_decode.py",
    )}
    plan.update(
        revision="drafter-fresh-capture-smoke-v4", private_port=8036, master_port=29709,
        purpose="Repeat the frozen eight-prompt capture with both qualified ordering fixes.",
        requests=12, client_deadline_seconds=1200, outage_seconds=3300,
        ordering_source_sha256=policy_sources,
        ordering_qualification_plan_sha256=digest(qualified / "plan.json"),
        ordering_qualification_review_sha256=digest(review), source_sha256=digest(Path(__file__)),
        notes=[
            "Same eight prompts and two longest-prompt before/capture/after controls as failed v3.",
            "Exact qualified prefill/decode ordering and expert policies; original fixed-seven drafter.",
            "No operation observers or full-logit retention; retain accepted-history feature evidence.",
            "Inherit verified component qualification without repeating startup timing experiments.",
            "Both ranks must independently export identical accepted BF16 features before full capture.",
            "No training, sealed-test access or serving-speed claim; restore production on every exit.",
        ],
    )
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    for name in ("prepare.py", "control.py"):
        path = output / name
        path.write_text(path.read_text().replace("8035", "8036").replace("29708", "29709")
                        .replace("mxwave-drafter-canonical-decode-v1",
                                 "mxwave-drafter-fresh-capture-smoke-v4"))
    (output / "measure_features.py").write_text(
        (template / "measure_features.py").read_text().replace("8028", "8036"),
    )
    path = output / "trial_plugin.py"
    text = path.read_text()
    anchor = '''        worker.mxwave_decode_component = check_canonical_sparse_decode.qualify(
            pathlib.Path(f"/trial/decode-fixture-r{rank}.safetensors"),
            pathlib.Path(f"/traces/decode-component-r{rank}.json"))'''
    if text.count(anchor) != 1:
        raise ValueError("Unexpected qualified decode startup hook")
    replacement = '''        worker.mxwave_decode_component = {
            "status": "inherited_verified", "rank": rank,
            "parent_review_sha256": json.loads(pathlib.Path("/trial/plan.json").read_bytes())["ordering_qualification_review_sha256"]}'''
    path.write_text(text.replace(anchor, replacement))
    for path in output.glob("*.py"):
        compile(path.read_bytes(), str(path), "exec")
    frozen = {p.name: digest(p) for p in sorted(output.iterdir())
              if p.is_file() and p.name != "source-manifest.json"}
    (output / "source-manifest.json").write_text(json.dumps(frozen, indent=2) + "\n")
    return {"plan_sha256": digest(output / "plan.json"), "frozen_files": len(frozen),
            "cases": 8, "requests": 12, "private_port": 8036, "sealed_test_opened": False}


def main() -> None:
    """Prepare locally, requiring verified correction evidence before any GPU work."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("template", "qualified", "review", "dataset", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.template, args.qualified, args.review, args.dataset, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
