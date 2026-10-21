"""Freeze a short Modal DFlash2 capture with its own six target feature streams."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from .drafter_fp8_roundtrip import digest
from .prepare_drafter_feature_capture import select_cases
from .prepare_pretrained_drafter import replace_one
from .prepare_selector_calibration import _MEASURE

REVISION = "drafter-modal-pilot-v1"
QUOTAS = {
    "train": {"code": 40, "chat": 12, "explanation": 12},
    "validation": {"code": 8, "chat": 4, "explanation": 4},
}


def prepare(template: Path, dataset: Path, output: Path) -> dict[str, Any]:
    """Reuse the verified service controller; capture no sealed data or trained weights."""
    sources = json.loads((template / "source-manifest.json").read_bytes())
    if not all(digest(template / name) == sha for name, sha in sources.items()):
        raise ValueError("Pretrained template changed")
    cases, hashes = select_cases(dataset, QUOTAS)
    output.mkdir(parents=True, exist_ok=False)
    for name in sources:
        shutil.copyfile(template / name, output / name)
    plan = json.loads((output / "plan.json").read_bytes())
    plan.update(
        revision=REVISION, arms=["ct"], modes={"ct": "ct_mxfp8"},
        variants={"ct": "modal_fp8"}, private_port=8043,
        purpose="Bounded fresh Modal context features for a 64-update CPU adaptation pilot.",
        calibration_cases=cases, control_cases=[cases[0]["id"], cases[-1]["id"]],
        requests=84, output_tokens=128, max_frames_per_case=160, full_logits_frames=0,
        target_feature_layers=[23, 27, 31, 35, 39, 43], quotas=QUOTAS,
        frozen_dataset_hashes=hashes, prompt_limit=1024, outage_seconds=2100,
        client_deadline_seconds=1200, sealed_test_opened=False,
        source_sha256=digest(Path(__file__)),
        notes=[
            "64 training and 16 validation prompts, selected before observing any new output.",
            "Current pretrained Modal FP8 checkpoint; no training or target changes during capture.",
            "Seven parallel drafts for training capture; later serving comparison uses matched widths.",
            "Six native feature taps; old Inco five-tap captures cannot substitute for this dataset.",
            "C1 recorder; parallel request capture requires request-keyed routing and mask validation.",
            "Before/capture/after controls, accepted-prefix reconstruction and both-rank verification.",
            "Original 500k production restored by the independent watchdog before CPU training.",
            "No sealed evaluation, full training, automatic promotion or instrumented speed claim.",
        ],
    )
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    for name in ("prepare.py", "control.py"):
        path = output / name
        path.write_text(path.read_text().replace("8040", "8043").replace("29713", "29716")
                        .replace("mxwave-drafter-pretrained-modal-v1", "mxwave-" + REVISION))
    replace_one(output / "trial_plugin.py", "        return result\n\n    loaded.",
                "        import drafter_head_probe\n"
                "        drafter_head_probe.arm_capture(worker)\n"
                "        return result\n\n    loaded.")
    replace_one(output / "trial_plugin.py",
                '        report["context_probe"] = pretrained_startup.install(worker)',
                '        report["pretrained"] = pretrained_startup.install(worker)\n'
                '        import drafter_context_probe\n'
                '        report["context_probe"] = drafter_context_probe.install(worker)')
    for name in ("drafter_context_probe.py", "drafter_head_probe.py"):
        shutil.copyfile(Path(__file__).with_name(name), output / name)
    (output / "measure_features.py").write_text(
        _MEASURE.replace("8018", "8043").replace('"capture_cases": 48', '"capture_cases": 80'),
    )
    # One peer archive avoids thousands of serial SCP handshakes after restoration.
    path = output / "finalize.py"
    text = path.read_text()
    begin = text.index('        script = \'import json,pathlib,sys;')
    end = text.index('    for block in plan["arms"]:', begin)
    text = text[:begin] + '''        peer_archive = ROOT / (block + "-r1-transfer.tar")
        c.run(1, ["tar", "-cf", str(peer_archive), "-C", str(directory), "."], timeout=300)
        subprocess.run(["scp", "-i", "/home/kirya/.ssh/id_glm_cluster",
                        "kirya@192.168.192.2:" + str(peer_archive), str(peer_archive)],
                       check=True, timeout=600)
        with tarfile.open(peer_archive) as peer:
            peer.extractall(directory, filter="data")
''' + text[end:]
    text = text.replace('"queue.log", "finalize-queue.log"):',
                        '"queue.log", "finalize-queue.log", "run-prepared-status.json",\n'
                        '                      "ct-r1-transfer.tar"):')
    path.write_text(text)
    for path in output.glob("*.py"):
        compile(path.read_bytes(), str(path), "exec")
    frozen = {p.name: digest(p) for p in sorted(output.iterdir())
              if p.is_file() and p.name != "source-manifest.json"}
    (output / "source-manifest.json").write_text(json.dumps(frozen, indent=2) + "\n")
    return {
        "revision": REVISION, "status": "prepared_not_launched", "requests": 84,
        "plan_sha256": digest(output / "plan.json"), "source_files": len(frozen),
        "quotas": QUOTAS, "sealed_test_opened": False,
        "maximum_prompt_tokens": max(len(r["prompt_ids"]) for r in cases),
    }


def main() -> None:
    """Prepare the bounded capture locally without changing production."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("template", "dataset", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.template, args.dataset, args.output), indent=2))


if __name__ == "__main__":
    main()
