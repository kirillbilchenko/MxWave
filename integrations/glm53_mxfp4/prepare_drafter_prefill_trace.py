"""Freeze an early-operation trace after repeated target-prefill feature hashes differ."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from .drafter_fp8_roundtrip import digest


def prepare(template: Path, screen: Path, review: Path, output: Path) -> dict[str, Any]:
    """Keep numerical policies fixed and add whole-operand digest observations to prefill."""
    observed = json.loads(screen.read_bytes())
    fields = observed["distinct_payloads_by_field"]
    if (observed["rank"] != 0 or fields["target_input_ids"] != 1
            or fields["target_positions"] != 1 or fields["aux_layer_5"] <= 1):
        raise ValueError("Require same-input target-prefill drift before instrumenting early layers")
    checked = json.loads(review.read_bytes())
    if (checked["status"] != "reviewed" or not checked["production_restored"]
            or checked["plan_sha256"] != digest(template / "plan.json")
            or len(checked["pairs"]) != 12
            or any(row["all_prefill_features_exact"] for row in checked["pairs"])):
        raise ValueError("Require completed independent review on both ranks")
    manifest = json.loads((template / "source-manifest.json").read_bytes())
    if not all(digest(template / name) == sha for name, sha in manifest.items()):
        raise ValueError("Repeat diagnostic sources changed")
    output.mkdir(parents=True, exist_ok=False)
    for name in manifest:
        shutil.copyfile(template / name, output / name)
    for name in ("decode_boundary_capture.py", "decode_state_values.py",
                 "drafter_prefill_boundary_probe.py"):
        text = Path(__file__).with_name(name).read_text()
        text = text.replace("from .decode_boundary_capture", "from decode_boundary_capture")
        text = text.replace("from .drafter_context_probe", "from drafter_context_probe")
        (output / name).write_text(text)
    shutil.copyfile(screen, output / "parent-prefill-screen.json")
    shutil.copyfile(review, output / "parent-independent-review.json")
    plan = json.loads((template / "plan.json").read_bytes())
    plan.update(
        revision="drafter-prefill-boundary-v1", private_port=8031,
        purpose="Find the first differing target-prefill operation with equal full operands.",
        parent_plan_sha256=digest(template / "plan.json"), parent_screen_sha256=digest(screen),
        parent_review_sha256=digest(review),
        prefill_boundary_rows=3973, prefill_boundary_layers=[0, 1, 2, 3, 4, 5],
        source_sha256=digest(Path(__file__)), client_deadline_seconds=1200,
        outage_seconds=3000,
    )
    plan["notes"] += [
        "Hash full operands/results at early target layers, excluding scratch and unused routing tails.",
        "CPU observations synchronize the eager prefill; they may alter overlap, so controls remain essential.",
        "No prefix-only hashing: differences elsewhere in the batch may change dispatch/reduction choices.",
        "A trace that becomes repeatable under observation does not qualify the unobserved runtime.",
    ]
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    for name in ("prepare.py", "control.py", "measure_features.py"):
        path = output / name
        path.write_text(path.read_text().replace("8030", "8031").replace("29703", "29704")
                        .replace("mxwave-drafter-long-repeat-v1", "mxwave-drafter-prefill-boundary-v1"))
    path = output / "trial_plugin.py"
    text = path.read_text()
    anchor = '        report["context_probe"] = drafter_context_probe.install(worker)'
    if text.count(anchor) != 1:
        raise ValueError("Unexpected context probe hook")
    path.write_text(text.replace(anchor, '        import drafter_prefill_boundary_probe\n'
                                '        report["context_probe"] = drafter_prefill_boundary_probe.install(worker)'))
    path = output / "trial_boot.py"
    text = path.read_text()
    anchor = "        import verify_capture_tokenizer\n"
    assert text.count(anchor) == 1
    path.write_text(text.replace(anchor, "        import drafter_prefill_boundary_probe\n"
                                "        assert callable(drafter_prefill_boundary_probe.install)\n" + anchor))
    for path in output.glob("*.py"):
        compile(path.read_bytes(), str(path), "exec")
    manifest = {p.name: digest(p) for p in sorted(output.iterdir())
                if p.is_file() and p.name != "source-manifest.json"}
    (output / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"plan_sha256": digest(output / "plan.json"), "rows": 3973,
            "requests": plan["requests"], "training_eligible": False, "sealed_test_opened": False}


def main() -> None:
    """Prepare a CPU-reviewable diagnostic with no serving or weight mutation."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("template", "screen", "review", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.template, args.screen, args.review, args.output), indent=2))


if __name__ == "__main__":
    main()
