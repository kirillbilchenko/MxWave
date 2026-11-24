"""Freeze a native five-draft comparison of the selected small Modal pilot."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from .drafter_fp8_roundtrip import digest
from .prepare_pretrained_drafter import replace_one

REVISION = "drafter-modal-native-v1"
ARMS = ("original_a", "adapted", "original_b")
ORIGINAL_SHA = "45f34ef126aa258d8e16bd605a3a05c36eb3fef3ed0f57aef20191db49f1ef46"
CANDIDATE_SHA = "6dca2af520c6df5105eca783a07d1b170ea16ddbcefe31aba6ece54cec741925"


def prepare(template: Path, pilot: Path, checkpoint: Path, output: Path) -> dict[str, Any]:
    """Bind the selected checkpoint and all development cases before any serving outage."""
    sources = json.loads((template / "source-manifest.json").read_bytes())
    assert all(digest(template / name) == sha for name, sha in sources.items())
    plan = json.loads((template / "plan.json").read_bytes())
    capture = json.loads((pilot / "prepared/plan.json").read_bytes())
    export = json.loads((checkpoint / "export-review.json").read_bytes())
    assert plan["revision"] == "ct-draft-length-v1"
    assert capture["revision"] == "drafter-modal-pilot-v1"
    assert not plan["sealed_test_opened"] and not capture["sealed_test_opened"]
    assert export["status"] == "verified" and export["selected_step"] == 16
    assert export["original_sha256"] == ORIGINAL_SHA
    assert export["exported_sha256"] == digest(checkpoint / "model.safetensors") == CANDIDATE_SHA
    assert export["config_bytes_identical"] and export["only_changed_tensor"] == "fc.weight"
    files = {n: digest(checkpoint / n) for n in ("model.safetensors", "config.json")}
    assert files["config.json"] == plan["drafter_files"]["modal_fp8"]["config.json"]
    cases = [c for c in capture["calibration_cases"] if c["split"] == "validation"]
    assert len(cases) == len({c["id"] for c in cases}) == 16
    assert {d: sum(c["domain"] == d for c in cases) for d in ("code", "chat", "explanation")} == {
        "code": 8, "chat": 4, "explanation": 4,
    }
    output.mkdir(parents=True, exist_ok=False)
    for name in sources:
        shutil.copyfile(template / name, output / name)
    plan.update(
        revision=REVISION, arms=list(ARMS), depths=dict.fromkeys(ARMS, 5),
        modes=dict.fromkeys(ARMS, "ct_mxfp8"),
        variants={a: "modal_fp8_step16" if a == "adapted" else "modal_fp8" for a in ARMS},
        checkpoint_directories={
            "modal_fp8": "drafter-pretrained-modal-v1-modal_fp8-checkpoint",
            "modal_fp8_step16": "drafter-modal-pilot-v1-step16",
        },
        purpose="Native development check of selected Modal context adaptation at five drafts.",
        calibration_cases=cases, trials=6, requests=96, private_port=8044,
        outage_seconds=4500, client_deadline_seconds=1500,
        target_feature_layers=capture["target_feature_layers"], full_logits_frames=0,
        parent_plan_sha256=digest(template / "plan.json"),
        pilot_plan_sha256=digest(pilot / "prepared/plan.json"),
        pilot_report_sha256=digest(pilot / "run/report.json"),
        native_export_sha256=digest(checkpoint / "export-review.json"),
        source_sha256=digest(Path(__file__)), selected_step=16,
        development_thresholds={"code_geomean_tps_gain": .08,
                                "maximum_chat_explanation_tps_regression": .02},
        notes=[
            "Same CT target, compression, serving settings and five native drafts in all arms.",
            "Only the Modal fc.weight differs; all 137 other tensor payloads and config match.",
            "All 16 pilot validation prompts, six seed42 shuffled cycles, 256 output tokens.",
            "These prompts selected the checkpoint: this tests native transfer, not generalization.",
            "No sealed data, further training, automatic promotion or production benchmark arm.",
            "Original/adapted/original loads bracket drift; this is not request-interleaved A/B.",
            "Retain output differences: target output invariance remains unqualified.",
            "Independent bounded restoration watchdog restores original 500k production.",
        ],
    )
    plan["drafter_files"]["modal_fp8_step16"] = files
    plan["speculative_config"]["num_speculative_tokens"] = 5
    plan["speculative_config"]["num_speculative_tokens_per_batch_size"][0][2] = 5
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    for name in ("prepare.py", "control.py"):
        path = output / name
        path.write_text(path.read_text().replace("8042", "8044").replace("29715", "29717")
                        .replace("mxwave-ct-draft-length-v1", "mxwave-" + REVISION))
    replace_one(output / "prepare.py",
                '                    mount[0] = str(ROOT.parent / ("drafter-pretrained-modal-v1-"\n'
                '                                      + plan["variants"][arm] + "-checkpoint"))',
                '                    mount[0] = str(ROOT.parent /\n'
                '                        plan["checkpoint_directories"][plan["variants"][arm]])')
    path = output / "prepare.py"
    path.write_text(path.read_text().replace('["s7a"]', '["original_a"]')
                    .replace('["s7b"]', '["original_b"]'))
    replace_one(output / "pretrained_startup.py", '    if variant=="modal_fp8":',
                '    if variant in ("modal_fp8", "modal_fp8_step16"):')
    replace_one(output / "pretrained_startup.py",
                '            "checkpoint_files_verified":True,"fc_native_bitwise_equal":True,',
                '            "checkpoint_files_verified":True,"fc_native_bitwise_equal":True,\n'
                '            "checkpoint_files":expected,')
    # This uses the reviewed client unchanged apart from the frozen experiment identity/count.
    path = output / "measure_features.py"
    path.write_text(path.read_text().replace('"ct-draft-length-v1"', json.dumps(REVISION))
                    .replace('len(plan["calibration_cases"]) == 8',
                             'len(plan["calibration_cases"]) == 16'))
    for path in output.glob("*.py"):
        compile(path.read_bytes(), str(path), "exec")
    frozen = {p.name: digest(p) for p in sorted(output.iterdir())
              if p.is_file() and p.name != "source-manifest.json"}
    (output / "source-manifest.json").write_text(json.dumps(frozen, indent=2) + "\n")
    return {"status": "frozen_not_launched", "revision": REVISION,
            "plan_sha256": digest(output / "plan.json"), "requests": 288,
            "sources": len(frozen), "sealed_test_opened": False}


def main() -> None:
    """Prepare the native development screen without starting or stopping production."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("template", "pilot", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.template, args.pilot, args.checkpoint, args.output), indent=2))


if __name__ == "__main__":
    main()
