"""Freeze a short native first-divergence trace after the existing speed comparison."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from .drafter_fp8_roundtrip import digest
from .prepare_drafter_verification import MEASURE
from .prepare_pretrained_drafter import replace_one

REVISION = "drafter-pretrained-verification-v1"

QUEUE = '''"""Wait for the owned timing experiment and its verified production restoration."""
from __future__ import annotations
import hashlib,json,os,pathlib,subprocess,time
ROOT=pathlib.Path(__file__).parent
PARENT=ROOT.parent/"drafter-pretrained-modal-v1"

def status(phase,**extra):
    value={"phase":phase,"pid":os.getpid(),"updated_epoch":time.time(),**extra}
    path=ROOT/"queue.json";tmp=path.with_suffix(".incomplete")
    tmp.write_text(json.dumps(value,indent=2)+"\\n");tmp.replace(path)

def main():
    plan=json.loads((ROOT/"plan.json").read_bytes())
    sources=json.loads((ROOT/"source-manifest.json").read_bytes())
    assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==sha for n,sha in sources.items())
    started=time.time()
    while True:
        assert time.time()-started<5400,"Comparison wait expired; no new outage"
        assert hashlib.sha256((PARENT/"plan.json").read_bytes()).hexdigest()==plan["parent_plan_sha256"]
        parent=json.loads((PARENT/"controller.json").read_bytes())
        assert parent["phase"] not in ("failed_restored","preparation_failed"),"Parent failed; do not chain an outage"
        runner=PARENT/"run-prepared-status.json"
        service=PARENT/"final-service-review.json"
        if parent["phase"]=="complete_restored" and not parent["service_paused"] and runner.exists() and service.exists():
            done=json.loads(runner.read_bytes());review=json.loads(service.read_bytes())
            if done["step"]=="finalize":
                assert done["returncode"]==0 and done["control_returncode"]==0
                assert review["status"]=="verified" and review["health_200"] and review["originals_running"]
                assert parent["restoration"]["health_200"] and parent["restoration"]["originals_running"]
                for arm in ("a1","b1"):
                    measurement=json.loads((PARENT/(arm+"-measurement.json")).read_bytes())
                    assert measurement["status"]=="complete" and measurement["repeat_tokens_and_counters_exact"]
                    rows=[r for r in measurement["cases"] if r["case"]==plan["calibration_cases"][0]["id"]]
                    assert len(rows)==6
                    assert all(r["result"]["token_ids"][:64]==plan["expected_outputs"][arm] for r in rows)
                break
        status("waiting_for_timing_and_restoration",parent_phase=parent["phase"])
        time.sleep(20)
    for step in ("prepare","run-prepared"):
        status(step)
        with (ROOT/(step+"-launch.log")).open("xb") as log:
            result=subprocess.run(["python3","-S","-B","-u",str(ROOT/(step+".py"))],stdout=log,stderr=subprocess.STDOUT)
        assert result.returncode==0,(step,result.returncode)
    state=json.loads((ROOT/"controller.json").read_bytes())
    done=json.loads((ROOT/"run-prepared-status.json").read_bytes())
    assert state["phase"]=="complete_restored" and not state["service_paused"]
    assert done["step"]=="finalize" and done["returncode"]==0 and done["control_returncode"]==0
    status("complete_retained_requires_review",training_started=False)

if __name__=="__main__":
    try:main()
    except BaseException as error:
        status("failed",error=repr(error));raise
'''


def prepare(template: Path, evidence: Path, specification: Path, output: Path) -> dict[str, Any]:
    """Pin one observed divergent prefix without changing the running comparison."""
    manifest = json.loads((template / "source-manifest.json").read_bytes())
    if not all(digest(template / name) == sha for name, sha in manifest.items()):
        raise ValueError("Frozen timing template changed")
    spec = json.loads(specification.read_bytes())
    if (spec["parent_plan_sha256"] != digest(template / "plan.json")
            or spec["case"]["split"] != "validation" or spec["sealed_test_opened"]):
        raise ValueError("Diagnostic does not match the frozen development case")
    expected = {}
    measurement_hashes = {}
    for arm, filename in (("a1", "a1-measurement.json"),
                          ("b1", "b1-progress-first-round.json")):
        path = evidence / filename
        rows = json.loads(path.read_bytes())["cases"]
        rows = [r for r in rows if r["case"] == spec["case"]["id"]]
        expected[arm] = rows[0]["result"]["token_ids"][:64]
        if not all(r["result"]["token_ids"][:64] == expected[arm] for r in rows):
            raise ValueError("Observed prefix does not repeat")
        measurement_hashes[filename] = digest(path)
    first = spec["first_divergent_output_index_zero_based"]
    assert first == 47 and expected["a1"][:first] == expected["b1"][:first]
    assert expected["a1"][first] != expected["b1"][first]
    assert all(len(x) == 64 for x in expected.values())
    if expected != {"a1": spec["expected_uninstrumented_prefixes"]["original"],
                    "b1": spec["expected_uninstrumented_prefixes"]["modal"]}:
        raise ValueError("Diagnostic output-prefix specification changed")
    output.mkdir(parents=True, exist_ok=False)
    for name in manifest:
        shutil.copyfile(template / name, output / name)
    for name in ("drafter_context_probe.py", "drafter_verification_probe.py",
                 "pretrained_verification_probe.py"):
        text = Path(__file__).with_name(name).read_text().replace("from .", "from ")
        (output / name).write_text(text)
    plan = json.loads((output / "plan.json").read_bytes())
    plan.update(
        revision=REVISION, arms=["a1", "b1"],
        modes={"a1": "ct_mxfp8", "b1": "ct_mxfp8"},
        variants={"a1": "original", "b1": "modal"},
        purpose="Native first differing target decision with original/Modal drafters.",
        parent_plan_sha256=digest(template / "plan.json"),
        diagnostic_specification_sha256=digest(specification),
        input_measurement_sha256=measurement_hashes,
        calibration_cases=[spec["case"]], requests=4, output_tokens=64,
        private_port=8041, outage_seconds=1800, client_deadline_seconds=480,
        expected_outputs=expected, max_frames_per_case=80, full_logits_frames=80,
        state_capture_layers=[0, 4], max_state_bytes_per_frame=128 * 1024**2,
        max_capture_bytes_per_rank=8 * 1024**3,
        variant_feature_layers={"original": [5, 14, 24, 33, 42],
                                "modal": [23, 27, 31, 35, 39, 43]},
        source_sha256=digest(Path(__file__)),
        notes=[
            "One observed development prompt,64 output tokens; no training or sealed evaluation.",
            "Reproduce both earlier uninstrumented prefixes,including divergence at index47.",
            "Before/passive capture/repeated capture/after: exact within-arm tokens/counters.",
            "Native graph references and eager copies only; no logits/proposals/state substitution.",
            "Retain full pre/post-sampler logits,actual graph shapes,proposals and KDA layers0/4 state.",
            "Five/six feature taps follow each native checkpoint; unlike taps are not compared as equals.",
            "Compare only actual shared accepted histories; unused speculative slots are not evidence.",
            "Parent timing comparison and original500k restoration must complete before launch.",
            "Thirty-minute watchdog; original production restores on completion/failure.",
        ],
    )
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    for name in ("prepare.py", "control.py"):
        path = output / name
        path.write_text(path.read_text().replace("8040", "8041").replace("29713", "29714")
                        .replace("mxwave-drafter-pretrained-modal-v1", "mxwave-" + REVISION))
    replace_one(output / "control.py", "free > 20 * 1024**3", "free > 40 * 1024**3")
    replace_one(output / "trial_plugin.py", "        return result\n\n    loaded.",
                "        import drafter_head_probe\n"
                "        drafter_head_probe.arm_capture(worker)\n"
                "        return result\n\n    loaded.")
    replace_one(output / "trial_plugin.py",
                "        import pretrained_startup\n"
                '        report["context_probe"] = pretrained_startup.install(worker)',
                "        import pretrained_verification_probe\n"
                '        report["context_probe"] = pretrained_verification_probe.install(worker)')
    replace_one(output / "trial_boot.py", "        import verify_capture_tokenizer\n",
                "        import pretrained_verification_probe\n"
                "        assert callable(pretrained_verification_probe.install)\n"
                "        import verify_capture_tokenizer\n")
    (output / "measure_features.py").write_text(MEASURE.replace("8022", "8041"))
    (output / "wait_for_comparison.py").write_text(QUEUE)
    for path in output.glob("*.py"):
        compile(path.read_bytes(), str(path), "exec")
    frozen = {p.name: digest(p) for p in sorted(output.iterdir())
              if p.is_file() and p.name != "source-manifest.json"}
    (output / "source-manifest.json").write_text(json.dumps(frozen, indent=2) + "\n")
    return {"revision": REVISION, "plan_sha256": digest(output / "plan.json"),
            "source_files": len(frozen), "requests": 8, "output_tokens": 64,
            "sealed_test_opened": False, "gpu_launched": False}


def main() -> None:
    """Prepare locally; the explicit remote queue owns the subsequent launch."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("template", "evidence", "specification", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.template, args.evidence, args.specification, args.output), indent=2))


if __name__ == "__main__":
    main()
