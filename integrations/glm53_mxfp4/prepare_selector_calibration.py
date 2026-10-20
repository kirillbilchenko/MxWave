"""Freeze a selector-only data capture using the existing production restore controller."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

_MEASURE = '''"""Collect separate calibration data with native-output repeat controls."""
from __future__ import annotations
import argparse
import json
import pathlib
import re
import sys
import time
import measure_serving as m
ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT.parent / "full-c16admissionv1-src"))
import full_model_control as c
m.BASE = "http://127.0.0.1:8018"

def idle():
    for _ in range(2):
        text = m.request("/metrics").decode()
        values = [float(line.rsplit(" ", 1)[1]) for line in text.splitlines()
                  if re.match(r"vllm:num_requests_(running|waiting)(\\{| )", line)]
        assert values and not any(values)
        time.sleep(0.5)

def marker(active, case):
    idle()
    program = ("import pathlib,sys;p=pathlib.Path(sys.argv[1]);"
               "q=p.with_suffix('.incomplete');q.write_text(sys.argv[2]);q.replace(p)")
    for rank in (0, 1):
        c.run(rank, ["python3", "-S", "-c", program,
                    str(ROOT / f"ct-r{rank}" / "probe-marker.json"),
                    json.dumps({"active": active, "case": case})])

def counters(text):
    return {key: float(value) for line in text.splitlines()
            if line.startswith("vllm:spec_decode_") and "_total" in line
            for key, value in [line.rsplit(" ", 1)]}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["ct"], required=True)
    parser.parse_args()
    path = ROOT / "ct-measurement.json"
    assert not path.exists()
    plan = json.loads((ROOT / "plan.json").read_bytes())
    report = {"status": "running", "cases": [], "timing_claim": False, "controls": []}
    m.write(path, report)
    try:
        for case in plan["calibration_cases"]:
            controlled = case["id"] in plan["control_cases"]
            policies = ("before", "capture", "after") if controlled else ("capture",)
            rows = []
            for policy in policies:
                marker(policy == "capture", case["id"])
                before = counters(m.request("/metrics").decode())
                result = m.stream(case["prompt_ids"], plan["output_tokens"])
                idle()
                after = counters(m.request("/metrics").decode())
                row = {"case": case["id"], "split": case["split"], "policy": policy,
                       "result": result, "counter_delta":
                       {k: v - before.get(k, 0) for k, v in after.items()}}
                rows.append(row)
                report["cases"].append(row)
                m.write(path, report)
            if controlled:
                output_equal = all(r["result"]["token_ids"] == rows[0]["result"]["token_ids"]
                                   for r in rows)
                counters_equal = all(r["counter_delta"] == rows[0]["counter_delta"] for r in rows)
                report["controls"].append({"case": case["id"], "output_equal": output_equal,
                                           "counters_equal": counters_equal})
                assert output_equal and counters_equal, report["controls"][-1]
            print(json.dumps({"case": case["id"], "complete": True,
                              "requests": len(report["cases"])}), flush=True)
        marker(False, "complete")
        report.update(status="complete", summary={"requests": len(report["cases"]),
                      "capture_cases": 48, "controls_passed": True, "timing_claim": False})
        m.write(path, report)
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        m.write(path, report)
        raise

if __name__ == "__main__":
    main()
'''


def prepare(template: Path, prompts: Path, output: Path) -> dict[str, str]:
    """Create a new immutable source bundle without touching services or checkpoints."""
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    source = Path(__file__).parent
    for name in (
        "trial_boot.py", "trial_worker.py", "trial_plugin.py", "revision.json",
        "rounding-reference.json", "kda_mxfp8_runtime.py", "kda_mxfp8_roundtrip.py",
        "measure_serving.py", "prepare.py", "control.py", "finalize.py", "launch_followup.py",
    ):
        shutil.copy2(template / name, output / name)
    for name in ("drafter_head_probe.py", "drafter_context_probe.py",
                 "selector_calibration_probe.py"):
        shutil.copy2(source / name, output / name)
    rows = json.loads(prompts.read_bytes())
    assert len(rows) == 48 and len({r["id"] for r in rows}) == 48
    plan = json.loads((template / "plan.json").read_bytes())
    for key in ("cases", "profile_case_indices", "full_logits_frames", "target_feature_layers"):
        plan.pop(key, None)
    plan.update(
        revision="selector-calibration-capture-v1", arms=["ct"], modes={"ct": "ct_mxfp8"},
        purpose="Collect new CT native draft states for a bounded selector-projection pilot.",
        private_port=8018, requests=52, output_tokens=128, max_frames_per_case=160,
        client_deadline_seconds=1200, outage_seconds=2100,
        calibration_cases=rows, control_cases=[rows[0]["id"], rows[-1]["id"]],
        notes=[
            "32 train / 8 validation / 8 sealed test prompts; no earlier benchmark prompts.",
            "Only native selector states retained on rank zero; no target features.",
            "Only 1,048,576 projection weights may train later; target/backbone frozen.",
            "Data capture changes no weights; before/capture/after token and counter controls.",
            "Original production restored before any CPU training; private port 8018.",
            "No timing or quality claims from instrumented data capture.",
        ],
    )
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    p = output / "trial_plugin.py"
    p.write_text(p.read_text().replace("drafter_context_probe", "selector_calibration_probe"))
    for name in ("prepare.py", "control.py"):
        p = output / name
        p.write_text(p.read_text().replace("8017", "8018").replace("29690", "29691")
                     .replace("mxwave-drafter-context-v1", "mxwave-selector-calibration-capture-v1")
                     .replace("measure_context.py", "measure_calibration.py"))
    p = output / "launch_followup.py"
    p.write_text(p.read_text().replace('"drafter-head-v3"', '"drafter-context-v1"'))
    (output / "measure_calibration.py").write_text(_MEASURE)
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(output.iterdir()) if p.is_file()}
    (output / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for p in output.glob("*.py"):
        compile(p.read_bytes(), str(p), "exec")
    return manifest


def main() -> None:
    """Build the capture bundle from the previously verified controller templates."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare(args.template, args.prompts, args.output)
    print(json.dumps({"source_files": len(manifest), "output": str(args.output)}))


if __name__ == "__main__":
    main()
