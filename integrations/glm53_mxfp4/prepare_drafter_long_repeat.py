"""Capture repeated long-prompt features/logits after the fresh capture's repeat failure."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from .drafter_fp8_roundtrip import digest
from .prepare_drafter_verification import MEASURE


def prepare(template: Path, output: Path) -> dict[str, Any]:
    """Retain the same target/drafter policy and collect divergence evidence without training."""
    original = json.loads((template / "source-manifest.json").read_bytes())
    if not all(digest(template / n) == sha for n, sha in original.items()):
        raise ValueError("Fresh-capture sources changed")
    plan = json.loads((template / "plan.json").read_bytes())
    if plan["revision"] != "drafter-fresh-capture-smoke-v3":
        raise ValueError("Use the exact failed longest-prompt capture policy")
    longest = max((r for r in plan["calibration_cases"] if r["split"] == "train"),
                  key=lambda row: len(row["prompt_ids"]))
    output.mkdir(parents=True, exist_ok=False)
    for name in original:
        shutil.copyfile(template / name, output / name)
    warmups = plan["calibration_cases"][:3]
    plan.update(
        revision="drafter-long-repeat-v1", private_port=8030,
        parent_capture_plan_sha256=digest(template / "plan.json"),
        purpose="Locate the first long-prompt self-repeat divergence; no training or pass claim.",
        calibration_cases=[longest], warmup_cases=warmups, requests=6,
        full_logits_frames=160, client_deadline_seconds=900, outage_seconds=2700,
        control_cases=[], quotas={"train": {"code": 1}},
        source_sha256=digest(Path(__file__)),
        notes=[
            "Same original drafter and qualified C1 target policy as failed fresh capture.",
            "Replay the three preceding short prompts before repeating the 3973-token training prompt.",
            "Before, four captures, after; each response has 128 tokens and seven fixed drafts.",
            "Retain full target logits, five target features, native draft proposals and query positions.",
            "Differences are recorded without early abort so both sides of a divergence are retained.",
            "A completed diagnostic is not a passing repeat gate or an eligible training dataset.",
            "No weights, kernels, thresholds or sealed data change; production restores on every exit.",
        ],
    )
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    for name in ("prepare.py", "control.py"):
        path = output / name
        path.write_text(path.read_text().replace("8028", "8030").replace("29701", "29703")
                        .replace("mxwave-drafter-fresh-capture-smoke-v3", "mxwave-drafter-long-repeat-v1"))
    measure = MEASURE.replace("8022", "8030")
    measure = measure.replace(
        '        for _ in range(2):\n            idle();report["warmups"].append(m.stream(plan["warmup_case"]["prompt_ids"],32))\n',
        '        for warmup in plan["warmup_cases"]:\n'
        '            idle();report["warmups"].append(m.stream(warmup["prompt_ids"],128))\n',
    )
    measure = measure.replace('("before","capture1","capture2","after")',
                              '("before","capture1","capture2","capture3","capture4","after")')
    for line in (
        '                assert result["token_ids"]==plan["expected_outputs"][arm]\n',
        '                assert result["token_ids"]==prior["result"]["token_ids"]\n',
        '                assert row["counter_delta"]==prior["counter_delta"]\n',
    ):
        if measure.count(line) != 1:
            raise ValueError("Diagnostic template changed")
        measure = measure.replace(line, "")
    measure = measure.replace('                prior=report["cases"][0]\n', "")
    measure = measure.replace(
        '"native_outputs_match_prior":True,"before_capture_repeat_after_exact":True,',
        '"repeat_gate_passed":all(r["result"]["token_ids"]==report["cases"][0]["result"]["token_ids"] '
        'and r["counter_delta"]==report["cases"][0]["counter_delta"] for r in report["cases"]),'
        '"diagnostic_only":True,"training_eligible":False,',
    )
    (output / "measure_features.py").write_text(measure)
    for path in output.glob("*.py"):
        compile(path.read_bytes(), str(path), "exec")
    manifest = {p.name: digest(p) for p in sorted(output.iterdir())
                if p.is_file() and p.name != "source-manifest.json"}
    (output / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"plan_sha256": digest(output / "plan.json"), "requests": 6, "warmups": 3,
            "prompt_tokens": len(longest["prompt_ids"]), "training_eligible": False,
            "sealed_test_opened": False}


def main() -> None:
    """Prepare a focused trace locally, preserving all numerical policies."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("template", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.template, args.output), indent=2))


if __name__ == "__main__":
    main()
