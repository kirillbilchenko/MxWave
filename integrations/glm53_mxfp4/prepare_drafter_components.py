"""Freeze a bounded component trace of the already reproduced verification divergence."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from .drafter_fp8_roundtrip import digest
from .prepare_drafter_verification import MEASURE


def prepare(template: Path, audit: Path, output: Path) -> dict[str, Any]:
    """Reuse verified controls/restoration and add passive eight-row component copies."""
    if output.exists():
        raise FileExistsError(output)
    original_manifest = json.loads((template / "source-manifest.json").read_bytes())
    assert all(digest(template / name) == value for name, value in original_manifest.items())
    output.mkdir(parents=True)
    for name in original_manifest:
        shutil.copyfile(template / name, output / name)
    source = Path(__file__).parent
    for name in (
        "decode_boundary_capture", "decode_state_values", "drafter_component_probe",
        "drafter_verification_probe", "drafter_context_probe",
    ):
        text = (source / f"{name}.py").read_text()
        for module in ("decode_boundary_capture", "drafter_context_probe",
                       "drafter_verification_probe"):
            text = text.replace(f"from .{module} import", f"from {module} import")
        (output / f"{name}.py").write_text(text)
    plan = json.loads((template / "plan.json").read_bytes())
    counters = {}
    for arm in plan["arms"]:
        measurement = json.loads((template / f"{arm}-measurement.json").read_bytes())
        assert measurement["status"] == "complete"
        rows = measurement["cases"]
        assert len(rows) == 4
        assert all(row["counter_delta"] == rows[0]["counter_delta"] for row in rows)
        counters[arm] = rows[0]["counter_delta"]
    plan.update(
        revision="drafter-components-v2", private_port=8024,
        purpose="Find the first changed early target operation under native eight-row replay.",
        state_capture_layers=[], component_layers=[0, 1, 2, 3, 4],
        component_start_positions=[46, 54], max_component_graph_bytes=64 * 1024**2,
        expected_counter_deltas=counters, source_sha256=digest(Path(__file__)),
        parent_plan_sha256=digest(template / "plan.json"),
        notes=[
            "V1 failed before requests: token count alone selected two different eight-token graphs.",
            "V2 selects FULL, one request, eight uniform tokens, no LoRA; actual replay identity checked.",
            "Same development prompt, native target/draft arithmetic and original/adapted arms as V2.",
            "Only graph-written independent copies added; Python component wrappers removed after capture.",
            "Capture attention, mHC, dense MLPs and expert routing/GEMMs in layers 0 through 4.",
            "At most 64 MiB of retained graph copies; store components only at start positions 46 and 54.",
            "Before/capture/repeat/after outputs AND counters must reproduce the frozen V2 result.",
            "No new raw recurrent-cache copies; V2 separately retained selected state equality.",
            "One actual eight-row target graph, each selected layer once, four expert GEMMs required.",
            "No timing, training, promotion, sealed-test or relaxed-quality claim.",
            "25-minute experiment cutoff and independent restoration watchdog retained.",
        ],
    )
    audited = {
        "v1/worker/gpu/cudagraph_utils.py": "cudagraph_utils.py",
        "model_executor/layers/fused_moe/layer.py": "moe.py",
        "model_executor/layers/fused_moe/runner/moe_runner.py": "moe_runner.py",
        "model_executor/layers/fused_moe/routed_experts.py": "routed_experts.py",
        "model_executor/layers/fused_moe/router/fused_moe_router.py": "fused_moe_router.py",
    }
    plan["probe_vllm_sources"].update({name: digest(audit / file) for name, file in audited.items()})
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    for name in ("prepare.py", "control.py"):
        path = output / name
        text = path.read_text().replace("8022", "8024").replace("29695", "29697")
        path.write_text(text.replace("mxwave-drafter-verification-v2", "mxwave-drafter-components-v2"))
    path = output / "trial_plugin.py"
    text = path.read_text()
    anchor = "        drafter_head_probe.arm_capture(worker)\n"
    assert text.count(anchor) == 1
    text = text.replace(anchor, anchor + (
        "        import drafter_component_probe\n"
        "        drafter_component_probe.arm_capture(worker)\n"
    ))
    path.write_text(text.replace("drafter_verification_probe", "drafter_component_probe"))
    path = output / "trial_boot.py"
    text = path.read_text().replace("drafter_verification_probe", "drafter_component_probe")
    anchor = "        assert callable(drafter_component_probe.install)\n"
    assert text.count(anchor) == 1
    path.write_text(text.replace(anchor, anchor + (
        "        print(json.dumps(drafter_component_probe.validate_graph_candidates()))\n"
        "        from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner\n"
        "        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts\n"
        "        assert callable(MoERunner._apply_quant_method)\n"
        "        assert callable(MoERunner._maybe_reduce_final_output)\n"
        "        assert callable(RoutedExperts.forward_modular)\n"
    )))
    text = MEASURE.replace("8022", "8024")
    anchor = '                assert row["counter_delta"]==prior["counter_delta"]\n'
    assert text.count(anchor) == 1
    text = text.replace(anchor, anchor + (
        '                assert row["counter_delta"]==plan["expected_counter_deltas"][arm]\n'
    ))
    (output / "measure_features.py").write_text(text)
    for path in output.glob("*.py"):
        compile(path.read_bytes(), str(path), "exec")
    manifest = {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()}
    (output / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "plan_sha256": digest(output / "plan.json"), "source_files": len(manifest),
        "requests": 8, "output_tokens": 32, "component_frames_expected": 16,
        "sealed_test_opened": False, "outage_seconds": plan["outage_seconds"],
    }


def main() -> None:
    """Prepare locally without contacting or stopping production."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("template", "audit", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.template, args.audit, args.output), indent=2))


if __name__ == "__main__":
    main()
