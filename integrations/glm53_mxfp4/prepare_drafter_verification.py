"""Freeze a short original/candidate trace at the first native output divergence."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from .drafter_fp8_roundtrip import digest

MEASURE = '''"""Capture one development prompt with before/repeat/after native controls."""
import argparse,json,pathlib,re,sys,time
import measure_serving as m
ROOT=pathlib.Path(__file__).parent
sys.path.insert(0,str(ROOT.parent/"full-c16admissionv1-src"))
import full_model_control as c
m.BASE="http://127.0.0.1:8021"

def idle():
    text=m.request("/metrics").decode()
    values=[float(line.rsplit(" ",1)[1]) for line in text.splitlines()
            if re.match(r"vllm:num_requests_(running|waiting)(\\\\{| )",line)]
    assert values and not any(values)

def marker(arm,active,case):
    idle()
    program=("import pathlib,sys;p=pathlib.Path(sys.argv[1]);"
             "q=p.with_suffix('.incomplete');q.write_text(sys.argv[2]);q.replace(p)")
    for rank in (0,1):
        c.run(rank,["python3","-S","-c",program,
            str(ROOT/f"{arm}-r{rank}"/"probe-marker.json"),
            json.dumps({"active":active,"case":case})])

def counters():
    return {key:float(value) for line in m.request("/metrics").decode().splitlines()
            if line.startswith("vllm:spec_decode_") and "_total" in line
            for key,value in [line.rsplit(" ",1)]}

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--arm",required=True)
    arm=parser.parse_args().arm
    plan=json.loads((ROOT/"plan.json").read_bytes());assert arm in plan["arms"]
    path=ROOT/f"{arm}-measurement.json";assert not path.exists()
    report={"status":"running","arm":arm,"cases":[],"warmups":[],
            "timing_claim":False,"sealed_test_opened":False}
    m.write(path,report)
    try:
        for _ in range(2):
            idle();report["warmups"].append(m.stream(plan["warmup_case"]["prompt_ids"],32))
        for case in plan["calibration_cases"]:
            for policy in ("before","capture1","capture2","after"):
                marker(arm,policy.startswith("capture"),case["id"]+"-"+policy)
                before=counters()
                result=m.stream(case["prompt_ids"],plan["output_tokens"])
                idle();after=counters()
                row={"case":case["id"],"policy":policy,"result":result,
                     "counter_delta":{k:v-before.get(k,0) for k,v in after.items()}}
                report["cases"].append(row);m.write(path,report)
                prior=report["cases"][0]
                assert result["token_ids"]==plan["expected_outputs"][arm]
                assert result["token_ids"]==prior["result"]["token_ids"]
                assert row["counter_delta"]==prior["counter_delta"]
                print(json.dumps({"arm":arm,"policy":policy,"complete":True}),flush=True)
        marker(arm,False,"complete")
        report.update(status="complete",summary={"requests":len(report["cases"]),
            "native_outputs_match_prior":True,"before_capture_repeat_after_exact":True,
            "timing_claim":False})
        m.write(path,report)
    except BaseException as exc:
        report.update(status="failed",error=repr(exc));m.write(path,report);raise
if __name__=="__main__":main()
'''


def prepare(template: Path, audit: Path, output: Path) -> dict[str, Any]:
    """Reuse verified serving/restoration sources with fresh trace paths and bounds."""
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    names = (
        "trial_boot.py", "trial_worker.py", "trial_plugin.py", "revision.json",
        "rounding-reference.json", "kda_mxfp8_runtime.py", "kda_mxfp8_roundtrip.py",
        "measure_serving.py", "prepare.py", "control.py", "finalize.py",
        "verify_capture_tokenizer.py", "drafter_head_probe.py", "verify_native_checkpoint.py",
        "expected-runtime-r0.json", "expected-runtime-r1.json", "run-prepared.py",
    )
    for name in names:
        shutil.copyfile(template / name, output / name)
    source = Path(__file__).parent
    shutil.copyfile(source / "drafter_context_probe.py", output / "drafter_context_probe.py")
    probe = (source / "drafter_verification_probe.py").read_text().replace(
        "from .drafter_context_probe import ContextCapture",
        "from drafter_context_probe import ContextCapture",
    )
    (output / "drafter_verification_probe.py").write_text(probe)
    plan = json.loads((template / "plan.json").read_bytes())
    case = next(r for r in plan["calibration_cases"] if r["id"] == "opencoder-evol-15654")
    expected = {}
    for arm in ("a1", "b1"):
        rows = json.loads((template / f"{arm}-measurement.json").read_bytes())["cases"]
        expected[arm] = next(r for r in rows if r["case"] == case["id"])["result"]["token_ids"][:32]
    assert expected["a1"][:21] == expected["b1"][:21]
    assert expected["a1"][21] != expected["b1"][21]
    plan.update(
        revision="drafter-verification-v1", arms=["a1", "b1"],
        modes={"a1": "ct_mxfp8", "b1": "ct_mxfp8"},
        variants={"a1": "original", "b1": "candidate"},
        purpose="First-divergence target/logit/state trace; no timing or training claim.",
        calibration_cases=[case], requests=4, output_tokens=32,
        private_port=8021, client_deadline_seconds=300, outage_seconds=1500,
        max_frames_per_case=40, full_logits_frames=40, state_capture_layers=[0, 4],
        max_state_bytes_per_frame=128 * 1024**2, expected_outputs=expected,
        source_sha256=digest(Path(__file__)),
        notes=[
            "One previously observed development prompt; 32 output tokens; sealed test closed.",
            "Before / passive capture / repeated capture / after; outputs and counters must match.",
            "Each arm must reproduce its prior uninstrumented 32-token output prefix.",
            "Native target/draft graphs, inputs and sampled outputs are never replaced.",
            "Retain full pre/post-sampler logits, five target features and selected pre-forward KDA slots.",
            "All eight physical speculative cache slots retained; unused slots are not valid equality evidence.",
            "Identical accepted history is checked offline; future rejected draft inputs may differ.",
            "Only selected layers 0 and 4 have raw pre-forward recurrent caches; not a complete cache trace.",
            "No optimizer, target quantization, serving promotion or relaxed quality gate.",
            "25-minute experiment cutoff; original production restored on completion/failure.",
        ],
    )
    audited = {
        "v1/worker/gpu/model_runner.py": "model_runner.py",
        "v1/worker/gpu/model_states/mamba_hybrid.py": "mamba_hybrid.py",
        "models/glm5next/nvidia/model.py": "glm5next_model.py",
        "models/glm5next/nvidia/kda.py": "glm5next_kda.py",
    }
    plan["probe_vllm_sources"].update({name: digest(audit / file) for name, file in audited.items()})
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    for name in ("prepare.py", "control.py"):
        p = output / name
        text = p.read_text().replace("8020", "8021").replace("29693", "29694")
        text = text.replace("mxwave-drafter-context-native-v1", "mxwave-drafter-verification-v1")
        p.write_text(text)
    p = output / "trial_plugin.py"
    p.write_text(p.read_text().replace(
        "import drafter_context_probe\n        report[\"context_probe\"] = drafter_context_probe.install",
        "import drafter_verification_probe\n        report[\"context_probe\"] = drafter_verification_probe.install",
    ))
    p = output / "trial_boot.py"
    p.write_text(p.read_text().replace(
        "        import verify_capture_tokenizer\n",
        "        import drafter_verification_probe\n"
        "        assert callable(drafter_verification_probe.install)\n"
        "        import verify_capture_tokenizer\n",
    ))
    (output / "measure_features.py").write_text(MEASURE)
    manifest = {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()}
    (output / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for path in output.glob("*.py"):
        compile(path.read_bytes(), str(path), "exec")
    return {
        "plan_sha256": digest(output / "plan.json"), "source_files": len(manifest),
        "requests": 8, "output_tokens": 32, "sealed_test_opened": False,
    }


def main() -> None:
    """Prepare the trace without contacting or stopping production."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("template", "audit", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.template, args.audit, args.output), indent=2))


if __name__ == "__main__":
    main()
