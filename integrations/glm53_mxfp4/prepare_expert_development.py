"""Freeze the existing development comparison under the verified expert reduction policy."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from .drafter_fp8_roundtrip import digest

_RUN = '''"""Run the prepared comparison and retain evidence after successful restoration."""
from __future__ import annotations
import json, pathlib, subprocess, time
root = pathlib.Path(__file__).parent
with (root / "control-queue.log").open("xb") as log:
    result = subprocess.run(["python3", "-S", "-B", "-u", str(root / "control.py")],
                            stdout=log, stderr=subprocess.STDOUT)
(root / "run-prepared-status.json").write_text(json.dumps(
    {"step": "control", "returncode": result.returncode, "epoch": time.time()}) + "\\n")
state = json.loads((root / "controller.json").read_bytes())
if state["phase"] in ("complete_restored", "failed_restored") and not state["service_paused"]:
    with (root / "finalize-queue.log").open("xb") as log:
        retained = subprocess.run(["python3", "-S", "-B", "-u", str(root / "finalize.py")],
                                  stdout=log, stderr=subprocess.STDOUT)
    (root / "run-prepared-status.json").write_text(json.dumps(
        {"step": "finalize", "returncode": retained.returncode,
         "control_returncode": result.returncode, "epoch": time.time()}) + "\\n")
'''


def prepare(
    template: Path, verification_root: Path, verification_review: Path, output: Path,
) -> dict[str, Any]:
    """Preserve prompts, weights and repetition policy while applying the proven target hook."""
    review = json.loads(verification_review.read_bytes())
    assert review["status"] == "verified" and review["production_restored"]
    assert review["cross_drafter_output_ids_exact"] and not review["sealed_test_opened"]
    assert {r["rank"] for r in review["rank_results"]} == {0, 1}
    assert all(r["all_shared_decision_logits_exact"] for r in review["rank_results"])
    assert all(r["exact"] for r in review["capture_repeat_controls"])
    assert sum(r["full_target_graph_frames"] for r in review["observed_graph_controls"]) == 80
    assert digest(verification_root / "plan.json") == review["plan_sha256"]
    evidence = json.loads((verification_root / "evidence-manifest.json").read_bytes())
    for row in evidence["files"]:
        path = verification_root / row["path"]
        assert path.resolve().is_relative_to(verification_root.resolve())
        assert path.stat().st_size == row["bytes"] and digest(path) == row["sha256"]
    previous = json.loads((verification_root / "plan.json").read_bytes())
    original = json.loads((template / "source-manifest.json").read_bytes())
    assert all(digest(template / name) == sha for name, sha in original.items())
    plan = json.loads((template / "plan.json").read_bytes())
    assert plan["arms"] == ["a1", "b1", "a2"] and plan["requests"] == 48
    assert plan["output_tokens"] == 256 and plan["outage_seconds"] == 2400
    assert len(plan["calibration_cases"]) == 8
    assert all(r["split"] == "validation" for r in plan["calibration_cases"])
    assert plan["context_payloads"] == previous["context_payloads"]
    output.mkdir(parents=True, exist_ok=False)
    for name in original:
        shutil.copyfile(template / name, output / name)
    shutil.copyfile(template / "plan.json", output / "development-parent-plan.json")
    policy_sources = {}
    for name in ("expert_reduction_runtime", "expert_reduction_policy",
                 "expert_reduction_padding_gpu"):
        source = Path(__file__).parent / (name + ".py")
        assert digest(source) == digest(verification_root / source.name)
        shutil.copyfile(source, output / source.name)
        policy_sources[source.name] = digest(source)
    shutil.copyfile(verification_review, output / "verification-review.json")
    plan.update(
        revision="expert-fixed-development-v1", private_port=8027,
        purpose="Frozen eight-prompt A/B/A development comparison under equal expert reduction.",
        parent_development_plan_sha256=digest(template / "plan.json"),
        verification_plan_sha256=review["plan_sha256"],
        verification_review_sha256=digest(verification_review),
        source_sha256=digest(Path(__file__)),
        expert_reduction_policy=previous["expert_reduction_policy"],
        expert_reduction_source_sha256=policy_sources,
    )
    plan["notes"] = [n for n in plan["notes"] if "alternative GPU computation" not in n] + [
        "The exact default-occupancy expert padding hook from the passing C1 check is applied in all arms.",
        "Only the target reduction policy changes relative to the previous frozen development run.",
        "Eight prompts, six seed-42 cycles, 256 output tokens, seven drafts and original/candidate/original loads stay fixed.",
        "No feature or logit copies; lightweight native draft graph descriptor counting only.",
        "This comparison measures the adapted drafter with padding in both arms, not padding cost against native.",
        "Only one candidate load; final serving speed, larger batches and the sealed test remain unqualified.",
        "The target policy remains bounded to 16 rows; C8/C16 extension and full 4,000-prompt training are separate.",
    ]
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    for name in ("prepare.py", "control.py", "measure_features.py"):
        path = output / name
        text = path.read_text().replace("8020", "8027").replace("29693", "29700")
        path.write_text(text.replace("mxwave-drafter-context-native-v1",
                                     "mxwave-expert-fixed-development-v1"))
    for name in ("prepare.py", "control.py"):
        path = output / name
        text = path.read_text()
        anchor = '    plan = json.loads((ROOT / "plan.json").read_bytes())\n'
        assert text.count(anchor) == 1
        path.write_text(text.replace(anchor, anchor + (
            '    verification = ROOT.parent / "expert-fixed-verification-v1"\n'
            '    verified = json.loads((verification / "controller.json").read_bytes())\n'
            '    assert verified["phase"] == "complete_restored" and not verified["service_paused"]\n'
            '    assert hashlib.sha256((verification / "plan.json").read_bytes()).hexdigest() == plan["verification_plan_sha256"]\n'
        )))
    path = output / "trial_plugin.py"
    text = path.read_text()
    anchor = "        import drafter_head_probe\n"
    assert text.count(anchor) == 1
    text = text.replace(anchor, (
        "        import expert_reduction_runtime\n"
        "        expert_reduction_runtime.install()\n"
    ) + anchor)
    anchor = '        report["trial"] = worker.mxwave_trial_snapshot()\n'
    assert text.count(anchor) == 1
    path.write_text(text.replace(anchor, anchor + (
        "        import expert_reduction_runtime\n"
        '        report["expert_reduction"] = expert_reduction_runtime.snapshot()\n'
        '        assert report["expert_reduction"]["installed"]\n'
        '        shapes = {tuple(x["geometry"][:3]) for x in report["expert_reduction"]["dispatches"]}\n'
        "        assert {(8, 2048, 4096), (64, 4096, 1024)} <= shapes\n"
    )))
    path = output / "trial_boot.py"
    text = path.read_text()
    anchor = "        import drafter_head_probe\n"
    assert text.count(anchor) == 1
    path.write_text(text.replace(anchor, (
        "        import expert_reduction_runtime\n"
        "        from expert_reduction_policy import block_quantum, capacity_for\n"
        "        assert callable(expert_reduction_runtime.install)\n"
        "        assert block_quantum(2048, 128, 48, 3) == 9\n"
        "        assert capacity_for(448, 8, 9) == 504\n"
    ) + anchor))
    (output / "run-prepared.py").write_text(_RUN)
    for path in output.glob("*.py"):
        compile(path.read_bytes(), str(path), "exec")
    manifest = {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()}
    (output / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "plan_sha256": digest(output / "plan.json"), "source_files": len(manifest),
        "arms": plan["arms"], "requests": 144, "sealed_test_opened": False,
        "outage_seconds": plan["outage_seconds"],
    }


def main() -> None:
    """Prepare the reviewable experiment on CPU before contacting production."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("template", "verification-root", "verification-review", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.template, args.verification_root,
                             args.verification_review, args.output), indent=2))


if __name__ == "__main__":
    main()
