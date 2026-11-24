"""Prepare a bounded A/B/A native development comparison for the selected context patch."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from .drafter_fp8_roundtrip import digest

_MEASURE = '''"""Measure fixed validation prompts; never open the sealed serving test."""
from __future__ import annotations
import argparse, json, pathlib, random, re, statistics, time
import measure_serving as m
ROOT = pathlib.Path(__file__).parent
m.BASE = "http://127.0.0.1:8020"

def counters():
    text = m.request("/metrics").decode()
    return {key: float(value) for line in text.splitlines()
            if line.startswith("vllm:spec_decode_") and "_total" in line
            for key, value in [line.rsplit(" ", 1)]}

def idle():
    text = m.request("/metrics").decode()
    active = [float(line.rsplit(" ", 1)[1]) for line in text.splitlines()
              if re.match(r"vllm:num_requests_(running|waiting)(\\{| )", line)]
    assert active and not any(active), "Unexpected concurrent request"

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--arm", required=True)
    arm=parser.parse_args().arm
    plan=json.loads((ROOT/"plan.json").read_bytes()); assert arm in plan["arms"]
    path=ROOT/f"{arm}-measurement.json"; assert not path.exists()
    report={"status":"running","arm":arm,"cases":[],"warmups":[],
            "sealed_test_opened":False,"final_speed_qualification":False}
    m.write(path,report)
    try:
        for _ in range(2):
            idle(); report["warmups"].append(m.stream(plan["warmup_case"]["prompt_ids"],256))
        order=list(plan["calibration_cases"])
        rng=random.Random(42)
        for trial in range(6):
            rng.shuffle(order)
            for case in order:
                idle(); before=counters()
                result=m.stream(case["prompt_ids"],256)
                idle(); after=counters()
                row={"case":case["id"],"domain":case["domain"],"trial":trial,
                     "result":result,"counter_delta":{k:v-before.get(k,0) for k,v in after.items()}}
                previous=next((r for r in report["cases"] if r["case"]==case["id"]),None)
                if previous is not None:
                    assert result["token_ids"]==previous["result"]["token_ids"], "Within-arm output repeat failed"
                    assert row["counter_delta"]==previous["counter_delta"], "Within-arm speculative repeat failed"
                report["cases"].append(row); m.write(path,report)
            print(json.dumps({"arm":arm,"trial":trial,"requests":len(report["cases"])}),flush=True)
        summary={}
        for domain in ("code","chat","explanation"):
            rows=[r for r in report["cases"] if r["domain"]==domain]
            summary[domain]={"requests":len(rows),"median_tps":statistics.median(256/r["result"]["latency_seconds"] for r in rows),
                             "median_ttft":statistics.median(r["result"]["ttft_seconds"] for r in rows)}
        report.update(status="complete",summary=summary,repeat_tokens_and_counters_exact=True)
        m.write(path,report)
    except BaseException as exc:
        report.update(status="failed",error=repr(exc));m.write(path,report);raise

if __name__=="__main__": main()
'''
_VERIFY = '''"""Check the exact native checkpoint in each actual serving container."""
import hashlib,json,os
from pathlib import Path

def verify():
    plan=json.loads(Path("/trial/plan.json").read_bytes())
    variant=os.environ["MXWAVE_DRAFTER_VARIANT"]
    expected=plan["drafter_files"][variant]
    for name,sha in expected.items():
        with (Path("/draft")/name).open("rb") as stream:
            assert hashlib.file_digest(stream,"sha256").hexdigest()==sha,(variant,name)
    print(json.dumps({"native_drafter_file_verification":variant,"passed":True}),flush=True)
'''


def prepare(template: Path, candidate: Path, output: Path) -> dict[str, Any]:
    """Freeze both checkpoint identities, development prompts and restoring controller sources."""
    export = json.loads((candidate / "export-review.json").read_bytes())
    if export["status"] != "verified" or not export["config_bytes_identical"]:
        raise ValueError("Native export is not verified")
    if digest(candidate / "model.safetensors") != export["exported_sha256"]:
        raise ValueError("Native candidate file changed")
    output.mkdir(parents=True, exist_ok=False)
    names = ("trial_boot.py", "trial_worker.py", "trial_plugin.py", "revision.json",
             "rounding-reference.json", "kda_mxfp8_runtime.py", "kda_mxfp8_roundtrip.py",
             "measure_serving.py", "prepare.py", "control.py", "finalize.py", "launch_followup.py",
             "verify_capture_tokenizer.py", "drafter_head_probe.py", "drafter_context_probe.py")
    for name in names:
        shutil.copyfile(template / name, output / name)
    plan = json.loads((template / "plan.json").read_bytes())
    all_cases = plan["calibration_cases"]
    cases = [r for r in all_cases if r["split"] == "validation"]
    if len(cases) != 8:
        raise ValueError("Expected exactly the frozen eight development prompts")
    fc = next(r for r in export["tensors"] if r["name"] == "fc.weight")
    plan.update(
        revision="drafter-context-native-v1", arms=["a1", "b1", "a2"],
        modes={a: "ct_mxfp8" for a in ("a1", "b1", "a2")},
        variants={"a1": "original", "b1": "candidate", "a2": "original"},
        purpose="Native reload and repeated development acceptance/speed comparison; sealed test closed.",
        private_port=8020, requests=48, output_tokens=256, outage_seconds=2400,
        client_deadline_seconds=720, calibration_cases=cases, control_cases=[],
        warmup_case=next(r for r in all_cases if r["split"] == "train"),
        drafter_files={
            "original": {"model.safetensors": export["original_sha256"],
                         "config.json": digest(candidate / "config.json")},
            "candidate": {"model.safetensors": export["exported_sha256"],
                          "config.json": digest(candidate / "config.json")},
        },
        context_payloads={"original": fc["original_sha256"], "candidate": fc["exported_sha256"]},
        selected_step=24, export_review_sha256=digest(candidate / "export-review.json"),
        notes=[
            "Original / candidate / original loads on the identical fixed CT MXFP8 target.",
            "Six seed-42 shuffled prompt cycles per load; two excluded warmups per load.",
            "C1, 256 fixed output tokens, temperature zero, seed 42, seven draft tokens, prefix cache off.",
            "All 88 effective runtime tensors must match the prior native inventory except selected FC.",
            "Only lightweight graph descriptor counting; no feature copies or alternative GPU computation.",
            "Within-arm token/speculative-counter repeats required; cross-arm tokens checked after retention.",
            "Native replay counts logged at first use and every 64 calls; final tail may be unlogged.",
            "Existing resource sampler retains clock/thermal telemetry across all arms.",
            "Eight validation prompts are reused development data, not the sealed serving qualification.",
            "Only one candidate load; repeated baseline diagnoses load drift but cannot qualify final speed.",
            "Original production restored automatically on completion, failure or 40-minute experiment deadline.",
        ],
    )
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    for rank in (0, 1):
        shutil.copyfile(template / f"ct-r{rank}/draft-runtime-weights-r{rank}.json",
                        output / f"expected-runtime-r{rank}.json")
    for name in ("prepare.py", "control.py"):
        p = output / name
        text = p.read_text().replace("8019", "8020").replace("29692", "29693").replace(
            "mxwave-drafter-feature-capture-v1", "mxwave-drafter-context-native-v1",
        )
        p.write_text(text)
    p = output / "prepare.py"
    text = p.read_text().replace(
        'env.update(MXWAVE_TRIAL_ARM=mode, VLLM_SERVER_DEV_MODE="1")',
        'env.update(MXWAVE_TRIAL_ARM=mode, VLLM_SERVER_DEV_MODE="1",\n'
        '                       MXWAVE_DRAFTER_VARIANT=plan["variants"][arm])',
    )
    anchor = '                if mount[1] == "/trial":'
    assert text.count(anchor) == 1
    text = text.replace(anchor,
        '                if mount[1] == "/draft" and plan["variants"][arm] == "candidate":\n'
        '                    mount[0] = str(ROOT.parent / "drafter-context-native-v1-checkpoint")\n'
        + anchor)
    text = text.replace('result["cpu_checks"][f"{mode}-r{rank}"]',
                        'result["cpu_checks"][f"{arm}-r{rank}"]')
    p.write_text(text)
    p = output / "trial_boot.py"
    text = p.read_text().replace('    import hashlib\n',
                                 '    import hashlib\n    import verify_native_checkpoint\n'
                                 '    verify_native_checkpoint.verify()\n', 1)
    p.write_text(text)
    p = output / "drafter_context_probe.py"
    text = p.read_text()
    start = text.index('    capture = ContextCapture(runner, root, plan, rank)')
    end = text.index('    report = {', start)
    text = text[:start] + '''    import os
    expected = json.loads((Path("/trial") / f"expected-runtime-r{rank}.json").read_bytes())
    expected["model.fc.weight"]["sha256"] = plan["context_payloads"][os.environ["MXWAVE_DRAFTER_VARIANT"]]
    assert inventory == expected, "Unexpected effective native drafter weights"
    original_replay = spec.query_cudagraph_manager.run_fullgraph
    counts = {}
    def replay(descriptor):
        key = (descriptor.num_reqs, descriptor.num_tokens)
        result = original_replay(descriptor)
        counts[key] = counts.get(key, 0) + 1
        if counts[key] == 1 or counts[key] % 64 == 0:
            print(json.dumps({"native_draft_replay": list(key), "count": counts[key], "rank": rank}), flush=True)
        return result
    spec.query_cudagraph_manager.run_fullgraph = replay
''' + text[end:]
    text = text.replace('"observation": "Passive copies; original target, sampling and draft outputs returned"',
                        '"observation": "Exact runtime inventory; descriptor counting only, no tensor copies"')
    p.write_text(text)
    p = output / "launch_followup.py"
    p.write_text(p.read_text().replace('"selector-calibration-capture-v1"',
                                      '"drafter-feature-capture-v1"'))
    (output / "measure_features.py").write_text(_MEASURE)
    (output / "verify_native_checkpoint.py").write_text(_VERIFY)
    for path in output.glob("*.py"):
        compile(path.read_bytes(), str(path), "exec")
    manifest = {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()}
    (output / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"plan_sha256": digest(output / "plan.json"), "sources": len(manifest),
            "arms": plan["arms"], "requests_per_arm": 48, "sealed_test_opened": False}


def main() -> None:
    """Prepare files without contacting or stopping production."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("template", "candidate", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.template, args.candidate, args.output), indent=2))


if __name__ == "__main__":
    main()
