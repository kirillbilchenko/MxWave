"""Freeze the full CT feature capture after the fresh-policy smoke export passes."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from .drafter_fp8_roundtrip import digest
from .drafter_training_schedule import TrainingSchedule
from .prepare_drafter_fresh_capture import frozen_cases

EXPORT = '''
def copy_peer(path):
    import subprocess
    subprocess.run(["scp", "-i", "/home/kirya/.ssh/id_glm_cluster", str(path),
                    "kirya@192.168.192.2:" + str(path)], check=True, timeout=180)

def export_chunk(start, stop):
    import concurrent.futures
    sys.path.insert(0, str(ROOT / "export_lib"))
    from integrations.glm53_mxfp4.drafter_capture_pairing import pair_chunk
    marker(False, "export")
    copy_peer(ROOT / "ct-measurement.json")
    prepared = json.loads((ROOT / "prepared.json").read_bytes())
    def export_rank(rank):
        name = prepared["launches"]["ct"][rank]["name"]
        return c.run(rank, ["docker", "exec", "-e", "CUDA_VISIBLE_DEVICES=",
            "-e", "PYTHONPATH=/trial/export_lib", name, "python3", "-B", "-m",
            "integrations.glm53_mxfp4.drafter_incremental_export", "--root", "/capture",
            "--rank", str(rank), "--start", str(start), "--stop", str(stop)], timeout=900)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(export_rank, (0, 1)):
            print(result, flush=True)
    name = f"export-{start:04d}-{stop:04d}-r1.json"
    c.write(ROOT / name, c.remote_json(1, ROOT / name))
    pair_chunk(ROOT, start, stop)
    copy_peer(ROOT / f"export-{start:04d}-{stop:04d}-r0.json")
    copy_peer(ROOT / f"paired-{start:04d}-{stop:04d}.json")
    program = ("import sys,pathlib;sys.path.insert(0,sys.argv[1]);"
               "from integrations.glm53_mxfp4.drafter_capture_pairing import release_staging;"
               "print(release_staging(pathlib.Path(sys.argv[2]),int(sys.argv[3]),"
               "int(sys.argv[4]),int(sys.argv[5])))")
    for rank in (0, 1):
        c.run(rank, ["python3", "-S", "-B", "-c", program, str(ROOT / "export_lib"),
                     str(ROOT), str(rank), str(start), str(stop)], timeout=600)

def disk_guard():
    program = "import shutil,sys;assert shutil.disk_usage(sys.argv[1]).free > 20*1024**3"
    for rank in (0, 1):
        c.run(rank, ["python3", "-S", "-c", program, str(ROOT)])
'''

FINALIZE = '''"""Finalize paired incremental exports only after original production restoration."""
import hashlib, json, pathlib, sys
ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT / "export_lib"))
from integrations.glm53_mxfp4.drafter_capture_pairing import file_digest, compare_ranks
sys.path.insert(0, str(ROOT.parent / "full-c16admissionv1-src"))
import full_model_control as c

def main():
    state = json.loads((ROOT / "controller.json").read_bytes())
    assert state["phase"] in ("complete_restored", "failed_restored") and not state["service_paused"]
    assert state["restoration"]["health_200"] and state["restoration"]["originals_running"]
    assert c.health(8000)
    if state["phase"] != "complete_restored":
        c.write(ROOT / "partial-capture.json", {"status":"failed_restored", "error":state.get("error"),
                "completed_chunks":[p.name for p in sorted(ROOT.glob("paired-*.json"))]})
        return
    plan = json.loads((ROOT / "plan.json").read_bytes())
    measurement = json.loads((ROOT / "ct-measurement.json").read_bytes())
    assert measurement["status"] == "complete" and len(measurement["cases"]) == plan["requests"]
    assert all(r["output_equal"] and r["counters_equal"] for r in measurement["controls"])
    assert len(measurement["controls"]) == len(plan["control_cases"])
    rows = []
    for start in range(0, len(plan["calibration_cases"]), 64):
        stop = min(start + 64, len(plan["calibration_cases"]))
        receipts = [json.loads((ROOT / f"export-{start:04d}-{stop:04d}-r{r}.json").read_bytes())
                    for r in (0,1)]
        paired = compare_ranks(*receipts)
        assert paired["plan_sha256"] == file_digest(ROOT / "plan.json")
        for row in paired["rows"]:
            row["rank_features_bitwise_equal"] = True
        rows.extend(paired["rows"])
    assert [r["id"] for r in rows] == [r["id"] for r in plan["calibration_cases"]]
    assert sum(r["split"]=="train" for r in rows)==4000 and sum(r["split"]=="validation" for r in rows)==200
    for rank in (0,1):
        program = ("import pathlib,json,sys;r=pathlib.Path(sys.argv[1]);sys.path.insert(0,str(r/'export_lib'));"
                   "from integrations.glm53_mxfp4.drafter_capture_pairing import file_digest;"
                   "rows=json.loads((r/'pending-manifest.json').read_bytes())['rows'];"
                   "assert all(file_digest(r/'accepted'/x['path'])==x['sha256'] for x in rows)")
        report = {"schema":"mxwave-ct-drafter-feature-dataset-v1", "status":"verified",
            "plan_sha256":file_digest(ROOT / "plan.json"), "rows":rows,
            "frozen_dataset_hashes":plan["frozen_dataset_hashes"],
            "expert_reduction_policy":plan["expert_reduction_policy"],
            "all_rank_features_bitwise_equal":True, "sealed_test_opened":False,
            "production_restored":True, "training_performed":False,
            "total_bytes":sum(r["bytes"] for r in rows),
            "total_feature_rows":sum(r["feature_rows"] for r in rows),
            "controls":measurement["controls"],
            "raw_staging":"Released only after independent NumPy audits and equal exports on both nodes; hashes and metadata retained."}
        if rank == 0:
            c.write(ROOT / "pending-manifest.json", report)
        else:
            import subprocess
            subprocess.run(["scp","-i","/home/kirya/.ssh/id_glm_cluster",str(ROOT/"pending-manifest.json"),
                "kirya@192.168.192.2:"+str(ROOT/"pending-manifest.json")],check=True,timeout=180)
        c.run(rank,["python3","-S","-c",program,str(ROOT)],timeout=900)
    # Both final disk copies must verify before either dataset is usable by training.
    for rank in (0, 1):
        publish = "import pathlib,sys;r=pathlib.Path(sys.argv[1]);p=r/'accepted/manifest.json';assert not p.exists();p.write_bytes((r/'pending-manifest.json').read_bytes())"
        c.run(rank,["python3","-S","-c",publish,str(ROOT)])
    c.write(ROOT/"postprocess-complete.json", {"status":"verified", "prompts":len(rows),
            "manifest_sha256":file_digest(ROOT/"accepted/manifest.json"), "production_restored":True})

if __name__ == "__main__":
    main()
'''


def prepare(
    smoke: Path, smoke_export: Path, numpy_review: Path, dataset: Path, output: Path,
) -> dict[str, Any]:
    """Require a restored, independently verified smoke capture before freezing full collection."""
    report = json.loads((smoke_export / "manifest.json").read_bytes())
    audit = json.loads(numpy_review.read_bytes())
    if (report["status"] != "verified" or not report["production_restored"]
            or not report["all_rank_features_bitwise_equal"]
            or report["plan_sha256"] != digest(smoke / "plan.json")
            or audit["status"] != "verified" or audit["different_elements"] != 0
            or audit["export_manifest_sha256"] != digest(smoke_export / "manifest.json")
            or len(report["rows"]) != 8):
        raise ValueError("Fresh-policy smoke capture/export must pass first")
    original = json.loads((smoke / "source-manifest.json").read_bytes())
    if not all(digest(smoke / name) == sha for name, sha in original.items()):
        raise ValueError("Smoke sources changed")
    cases, hashes = frozen_cases(dataset)
    plan = json.loads((smoke / "plan.json").read_bytes())
    if hashes != plan["frozen_dataset_hashes"] or not plan["prefill_dispatch_source_sha256"]:
        raise ValueError("Frozen corpus or qualified prefill policy differs")
    output.mkdir(parents=True, exist_ok=False)
    for name in original:
        shutil.copyfile(smoke / name, output / name)
    controls = list(dict.fromkeys(
        [cases[i]["id"] for i in range(0, len(cases), 512)]
        + [cases[-1]["id"], *plan["control_cases"]],
    ))
    plan.update(
        revision="drafter-full-capture-v1", private_port=8029,
        purpose="Collect the full frozen 4000/200 corpus with independent incremental rank export.",
        calibration_cases=cases, control_cases=controls, requests=len(cases) + 2 * len(controls),
        quotas={split: dict(Counter(r["domain"] for r in cases if r["split"] == split))
                for split in ("train", "validation")},
        client_deadline_seconds=43200, outage_seconds=45900, chunk_size=64,
        release_verified_raw_staging=True, source_sha256=digest(Path(__file__)),
        smoke_plan_sha256=digest(smoke / "plan.json"),
        smoke_manifest_sha256=digest(smoke_export / "manifest.json"),
        smoke_numpy_audit_sha256=digest(numpy_review),
        notes=[
            "All 4000 frozen training and 200 validation prompts; sealed test remains unopened.",
            "Same numerical target policy, prefill budget and original fixed-seven drafter as verified smoke.",
            "Each rank independently exports and audits accepted rows every 64 prompts.",
            "Raw staging is released only after the two export payload hashes and all history statistics agree.",
            "Both accepted copies, raw-frame metadata, hashes and audit receipts are retained.",
            "Failure retains completed chunks and restores original production; partial data cannot start training.",
            "Twelve-hour client limit plus startup/restoration allowance; independent restoration watchdog.",
        ],
    )
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    for name in ("prepare.py", "control.py", "measure_features.py"):
        path = output / name
        path.write_text(path.read_text().replace("8028", "8029").replace("29701", "29702")
                        .replace("mxwave-drafter-fresh-capture-smoke-v3", "mxwave-drafter-full-capture-v1"))
    path = output / "prepare.py"
    text = path.read_text()
    anchor = '            for mount in entry["mounts"]:\n'
    assert text.count(anchor) == 1
    path.write_text(text.replace(anchor, '            entry["mounts"].append([str(ROOT), "/capture", "rw"])\n' + anchor))
    path = output / "measure_features.py"
    text = path.read_text().replace("\ndef main():", EXPORT + "\ndef main():")
    text = text.replace('        for case in plan["calibration_cases"]:\n',
                        '        for index, case in enumerate(plan["calibration_cases"]):\n'
                        '            if index % 64 == 0:\n                disk_guard()\n')
    anchor = '            print(json.dumps({"case": case["id"], "complete": True,\n'
    assert text.count(anchor) == 1
    text = text.replace(anchor, (
        '            if (index + 1) % 64 == 0 or index + 1 == len(plan["calibration_cases"]):\n'
        '                export_chunk((index // 64) * 64, index + 1)\n'
    ) + anchor)
    path.write_text(text)
    (output / "finalize.py").write_text(FINALIZE)
    package = output / "export_lib/integrations/glm53_mxfp4"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package.parent / "__init__.py").write_text("")
    for module in ("drafter_capture_pairing", "drafter_incremental_export", "drafter_feature_sequence",
                   "audit_drafter_feature_export_numpy", "drafter_fp8_roundtrip"):
        shutil.copyfile(Path(__file__).with_name(module + ".py"), package / (module + ".py"))
    protocol_path = Path("benchmarks/glm53-flash-hf-audit-2026-10-15/diagnostics-v1/"
                         "drafter-context-pilot-v3-protocol.json")
    protocol = json.loads(protocol_path.read_bytes())
    protocol.pop("baseline_reference")
    protocol.pop("supersedes")
    protocol.update(schema="mxwave-drafter-context-full-v1", bounded_feature_loading=True,
                    capture_plan_sha256=digest(output / "plan.json"),
                    purpose="One seeded full-data context-projection adaptation; serving gates unchanged.")
    protocol["optimizer"].update(maximum_steps=4000, warmup_steps=200)
    protocol["data"].update(train_prompts=4000, validation_prompts=200,
                            training_order="One seed-42 permutation of all eligible frozen training prompts; no replacement or repeat.")
    protocol["validation"].update(steps=[0, 250, 500, 1000, 2000, 3000, 4000],
                                  short_response_adjustment="Evaluate the last eligible training step even when earlier than 4000.")
    protocol["limits"] = [
        "Offline improvement does not prove native FP8 acceptance or serving speed.",
        "Native reload and frozen development gates are required before the sealed test.",
        "No confidence head, selector adaptation or target-weight change.",
    ]
    TrainingSchedule.from_protocol(protocol)
    (output / "training-protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    for path in output.rglob("*.py"):
        compile(path.read_bytes(), str(path), "exec")
    manifest = {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*"))
                if p.is_file() and p.name != "source-manifest.json"}
    (output / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"plan_sha256": digest(output / "plan.json"), "requests": plan["requests"],
            "training_protocol_sha256": digest(output / "training-protocol.json"),
            "prompts": len(cases), "sealed_test_opened": False, "started": False}


def main() -> None:
    """Prepare the full run locally without altering any service."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("smoke", "smoke-export", "numpy-review", "dataset", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.smoke, args.smoke_export, args.numpy_review, args.dataset,
                             args.output), indent=2))


if __name__ == "__main__":
    main()
