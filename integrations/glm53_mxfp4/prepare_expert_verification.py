"""Prepare a bounded full-model verification of the component-proven expert policy."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from .drafter_fp8_roundtrip import digest
from .prepare_drafter_verification import MEASURE


def prepare(template: Path, output: Path, component_review: Path) -> dict[str, Any]:
    """Freeze equal target policies after exact replay and a retained overhead measurement."""
    review = json.loads(component_review.read_bytes())
    assert review["status"] == "verified"
    assert review["timing_controls_total"] == 16
    assert len(review["timings"]) == review["timing_controls_total"]
    assert review["telemetry_available"] and review["native_output_matches"] == 48
    original = json.loads((template / "source-manifest.json").read_bytes())
    assert all(digest(template / name) == sha for name, sha in original.items())
    output.mkdir(parents=True, exist_ok=False)
    for name in original:
        shutil.copyfile(template / name, output / name)
    source = Path(__file__).parent
    for name in ("expert_reduction_runtime", "expert_reduction_policy",
                 "expert_reduction_padding_gpu"):
        shutil.copyfile(source / (name + ".py"), output / (name + ".py"))
    shutil.copyfile(component_review, output / "component-review.json")
    plan = json.loads((template / "plan.json").read_bytes())
    plan["historical_native_outputs"] = plan.pop("expected_outputs")
    plan["historical_native_counter_deltas"] = plan.pop("expected_counter_deltas")
    plan.update(
        revision="expert-fixed-verification-v1", private_port=8026,
        purpose="Verify drafter-independent target outputs with the same full-K expert policy.",
        parent_plan_sha256=digest(template / "plan.json"),
        component_review_sha256=digest(component_review),
        component_timing_controls={
            "passed": review["timing_controls_passed"], "total": review["timing_controls_total"],
            "failed_controls_retained": True, "overall_overhead_qualified": False,
            "next_run_scope": "Numerical verification only; no serving speed claim.",
        },
        source_sha256=digest(Path(__file__)),
        expert_reduction_policy={
            "mode": "bounded_padding_default_occupancy", "target_rows_max": 16,
            "prefill_above_limit": "native", "same_in_both_arms": True,
        },
        notes=[
            "Both original/adapted drafters use the same CT target and expert reduction policy.",
            "One prior development prompt, fixed seven draft tokens, 32 generated tokens.",
            "Before/capture/repeat/after outputs and counters must be exact within each arm.",
            "Old native output IDs/counters remain historical: the changed reduction policy can change rounding.",
            "Cross-arm output identity and shared accepted-history features/logits are reviewed separately.",
            "The component probe verifies replay of the selected native eight-row target graph.",
            "Policy applies to target expert projections at <=16 target rows; larger prefills stay native.",
            "No target weights, quantization, drafter weights, KL threshold or sealed data changed.",
            "This short verification is not the frozen multi-prompt drafter qualification.",
            "25-minute cutoff and independent original-service restoration watchdog.",
        ],
    )
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    for name in ("prepare.py", "control.py"):
        path = output / name
        text = path.read_text().replace("8024", "8026").replace("29697", "29699")
        path.write_text(text.replace("mxwave-drafter-components-v2",
                                     "mxwave-expert-fixed-verification-v1"))
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
    text = text.replace(anchor, anchor + (
        "        import expert_reduction_runtime\n"
        '        report["expert_reduction"] = expert_reduction_runtime.snapshot()\n'
        '        assert report["expert_reduction"]["installed"]\n'
        '        shapes = {tuple(x["geometry"][:3]) for x in report["expert_reduction"]["dispatches"]}\n'
        "        assert {(8, 2048, 4096), (64, 4096, 1024)} <= shapes\n"
    ))
    path.write_text(text)
    path = output / "trial_boot.py"
    text = path.read_text()
    anchor = "        import drafter_component_probe\n"
    assert text.count(anchor) == 1
    path.write_text(text.replace(anchor, (
        "        import expert_reduction_runtime\n"
        "        assert callable(expert_reduction_runtime.install)\n"
    ) + anchor))
    text = MEASURE.replace("8022", "8026")
    anchor = '                assert result["token_ids"]==plan["expected_outputs"][arm]\n'
    assert text.count(anchor) == 1
    text = text.replace(anchor, "")
    text = text.replace('"native_outputs_match_prior":True',
                        '"same_policy_self_repeat_passes":True')
    (output / "measure_features.py").write_text(text)
    for path in output.glob("*.py"):
        compile(path.read_bytes(), str(path), "exec")
    manifest = {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()}
    (output / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"plan_sha256": digest(output / "plan.json"), "source_files": len(manifest),
            "requests": 8, "sealed_test_opened": False, "outage_seconds": plan["outage_seconds"]}


def main() -> None:
    """Prepare locally; a passing independent component review is required."""
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("template", "output", "component-review"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.template, args.output, args.component_review), indent=2))


if __name__ == "__main__":
    main()
