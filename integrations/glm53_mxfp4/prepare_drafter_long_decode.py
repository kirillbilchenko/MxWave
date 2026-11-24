"""Freeze an observed first-decode graph only after canonical prefill independently repeats."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .audit_drafter_feature_export_numpy import arrays
from .drafter_fp8_roundtrip import digest


def prepare(template: Path, evidence: Path, review: Path, output: Path) -> dict[str, Any]:
    """Keep qualified prefill ordering and capture the first equal-input eight-row decode."""
    checked = json.loads(review.read_bytes())
    if (checked["status"] != "reviewed" or not checked["production_restored"]
            or not checked["component_checks_passed"] or checked["target_repeat_passed"]
            or checked["plan_sha256"] != digest(evidence / "plan.json")
            or len(checked["target_repeat_summary"]) != 12
            or not all(p["all_prefill_features_exact"] for p in checked["target_repeat_summary"])):
        raise ValueError("Require verified canonical prefill, component checks and remaining decode drift")
    manifest = json.loads((template / "source-manifest.json").read_bytes())
    if not all(digest(template / name) == sha for name, sha in manifest.items()):
        raise ValueError("Canonical full-model template changed")
    plan = json.loads((template / "plan.json").read_bytes())
    case = plan["calibration_cases"][0]
    start = len(case["prompt_ids"])
    for rank in (0, 1):
        frames = []
        for capture in range(1, 5):
            path = evidence / f"ct-r{rank}" / f"context-{case['id']}-capture{capture}-0001-r{rank}.safetensors"
            meta = json.loads(path.with_suffix(".json").read_bytes())
            if digest(path) != meta["payload_sha256"]:
                raise ValueError("First-decode evidence changed")
            value = arrays(path)
            if value["target_positions"].reshape(-1).tolist() != list(range(start, start + 8)):
                raise ValueError("Unexpected first-decode position layout")
            frames.append(value)
        if not all(np.array_equal(f["target_input_ids"], frames[0]["target_input_ids"]) for f in frames):
            raise ValueError("First-decode full input IDs differ; do not compare unequal forwards")
        if all(np.array_equal(f["aux_layer_5"], frames[0]["aux_layer_5"]) for f in frames):
            raise ValueError("Expected first-decode target drift by layer five")
    output.mkdir(parents=True, exist_ok=False)
    for name in manifest:
        shutil.copyfile(template / name, output / name)
    for name in ("decode_boundary_capture.py", "decode_state_values.py", "drafter_component_probe.py",
                 "drafter_verification_probe.py", "drafter_long_decode_probe.py"):
        text = Path(__file__).with_name(name).read_text().replace("from .", "from ")
        (output / name).write_text(text)
    shutil.copyfile(review, output / "parent-canonical-review.json")
    plan.update(
        revision="drafter-long-decode-v1", private_port=8034,
        purpose="Locate first equal-input decode drift after bit-identical canonical prefill.",
        parent_canonical_plan_sha256=digest(evidence / "plan.json"),
        parent_canonical_review_sha256=digest(review), decode_start_position=start,
        source_sha256=digest(Path(__file__)),
        notes=[
            "Keep native selected-pool canonicalization in prefill; no new arithmetic policy.",
            "Prior component checks independently passed; do not repeat startup component timings.",
            "Add bounded stream-ordered graph copies through layers0-4 for one request/eight rows.",
            "Retain native decode top-k score buffer, sequence bounds and selected pools at layer3.",
            "Only first decode payload is retained; full target feature/logit histories remain captured.",
            "Graph copies are present in ALL controls and may change scheduling; this is diagnostic only.",
            "No training, sealed-test access, quality/speed pass or promotion; restore production on exit.",
        ],
    )
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    for name in ("prepare.py", "control.py", "measure_features.py"):
        path = output / name
        path.write_text(path.read_text().replace("8033", "8034").replace("29706", "29707")
                        .replace("mxwave-drafter-canonical-prefill-v1", "mxwave-drafter-long-decode-v1"))
    path = output / "trial_plugin.py"
    text = path.read_text()
    a = text.index("        worker.mxwave_sparse_component = check_canonical_sparse_prefill.qualify(")
    b = text.index("        worker.mxwave_sparse_policy = canonical_sparse_prefill.install()", a)
    text = text[:a] + '''        worker.mxwave_sparse_component = {
            "status": "inherited_verified", "rank": rank,
            "parent_review_sha256": json.loads(pathlib.Path("/trial/plan.json").read_bytes())["parent_canonical_review_sha256"]}
''' + text[b:]
    anchor = "        import drafter_head_probe\n        drafter_head_probe.arm_capture(worker)"
    if text.count(anchor) != 1:
        raise ValueError("Unexpected pre-graph load hook")
    text = text.replace(anchor, "        import drafter_long_decode_probe\n"
                        "        drafter_long_decode_probe.arm_capture(worker)\n" + anchor)
    anchor = '        report["context_probe"] = drafter_context_probe.install(worker)'
    if text.count(anchor) != 1:
        raise ValueError("Unexpected passive context probe hook")
    text = text.replace(anchor, '        import drafter_long_decode_probe\n'
                        '        report["context_probe"] = drafter_long_decode_probe.install(worker)')
    path.write_text(text)
    path = output / "trial_boot.py"
    text = path.read_text(); anchor = "        import verify_capture_tokenizer\n"
    if text.count(anchor) != 1:
        raise ValueError("Unexpected preflight hook")
    path.write_text(text.replace(anchor, "        import drafter_long_decode_probe\n"
                                "        assert callable(drafter_long_decode_probe.arm_capture)\n" + anchor))
    for path in output.glob("*.py"):
        compile(path.read_bytes(), str(path), "exec")
    frozen = {p.name: digest(p) for p in sorted(output.iterdir())
              if p.is_file() and p.name != "source-manifest.json"}
    (output / "source-manifest.json").write_text(json.dumps(frozen, indent=2) + "\n")
    return {"plan_sha256": digest(output / "plan.json"), "frozen_files": len(frozen),
            "private_port": 8034, "decode_position": start, "requests": 6}


def main() -> None:
    """Prepare a bounded trace locally after independent review of the completed fix attempt."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("template", "evidence", "review", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.template, args.evidence, args.review, args.output), indent=2))


if __name__ == "__main__":
    main()
