"""Continue a frozen capture into verified local CPU training without changing serving."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

REMOTE_BASE = (
    "/home/kirya/local-spark/experiments/glm53-mxfp4-audit-2026-10-15/full-model-v1"
)
MODULES = (
    "continue_drafter_full_training", "train_drafter_context_pilot", "audit_drafter_specforge",
    "cpu_drafter_linear", "drafter_feature_store", "drafter_forward_preflight",
    "drafter_fp8_roundtrip", "drafter_training_schedule", "export_drafter_context",
)


def digest(path: Path) -> str:
    """Hash retained files without loading their full contents."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def remote(host: str, script: str) -> str:
    """Run a bounded status, verification or retention program on a named capture host."""
    return subprocess.check_output(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, "python3", "-"],
        input=script, text=True, timeout=120,
    )


def verify_dataset(
    root: Path, plan: dict[str, Any], expected_sha: str, plan_sha256: str,
) -> dict[str, Any]:
    """Require the exact frozen corpus and every verified accepted payload before training."""
    path = root / "manifest.json"
    manifest = json.loads(path.read_bytes())
    if (digest(path) != expected_sha or manifest["status"] != "verified"
            or not manifest["production_restored"] or not manifest["all_rank_features_bitwise_equal"]
            or manifest["sealed_test_opened"] or manifest["training_performed"]
            or manifest["plan_sha256"] != plan_sha256
            or manifest["frozen_dataset_hashes"] != plan["frozen_dataset_hashes"]
            or manifest.get("ordering_source_sha256") != plan["ordering_source_sha256"]
            or manifest.get("ordering_qualification_review_sha256")
            != plan["ordering_qualification_review_sha256"]):
        raise ValueError("Full dataset is not the verified corrected capture")
    rows = manifest["rows"]
    cases = plan["calibration_cases"]
    if (len(rows) != 4200 or len(cases) != 4200
            or Counter(row["split"] for row in rows) != {"train": 4000, "validation": 200}
            or len({row["id"] for row in rows}) != len(rows)):
        raise ValueError("Missing, duplicated or sealed capture rows")
    for row, case in zip(rows, cases, strict=True):
        for key in ("id", "split", "domain", "prompt_sha256", "token_ids_sha256"):
            if row[key] != case[key]:
                raise ValueError("Captured sample differs from the frozen corpus")
        payload = root / row["path"]
        if (row["path"] != f"{row['split']}-{row['id']}.safetensors" or payload.is_symlink()
                or payload.stat().st_size != row["bytes"] or digest(payload) != row["sha256"]
                or row["response_tokens"] != 128 or row["different_elements"] != 0
                or not row["rank_features_bitwise_equal"]
                or row["numpy_bf16_elements_compared"] != row["feature_rows"] * 20480
                or row["feature_rows"] != len(case["prompt_ids"]) + 127):
            raise ValueError("Accepted payload identity, shape or audit differs")
    return {"status": "verified", "manifest_sha256": expected_sha, "prompts": len(rows),
            "bytes": sum(row["bytes"] for row in rows), "sealed_test_opened": False}


def backup_results(root: Path, plan_sha256: str) -> dict[str, Any]:
    """Retain immutable training outputs on Spark2 and independently hash every archived file."""
    files = [root / name for name in (
        "continuation.json", "continuation-source-manifest.json", "plan.json",
        "training-protocol.json", "serving-thresholds.json", "local-dataset-verification.json",
        "training.log", "export.log",
    ) if (root / name).exists()]
    for name in ("frozen", "run", "native-candidate"):
        files += [p for p in sorted((root / name).rglob("*"))
                  if p.is_file() and "__pycache__" not in p.parts]
    if any(p.is_symlink() for p in files):
        raise ValueError("Training evidence contains an unexpected symlink")
    entries = [{"path": str(p.relative_to(root)), "bytes": p.stat().st_size, "sha256": digest(p)}
               for p in files]
    manifest = root / "result-manifest.json"
    manifest.write_text(json.dumps({"files": entries}, indent=2) + "\n")
    archive = root / "results-v1.tar.gz"
    with tarfile.open(archive, "w:gz", compresslevel=1) as tar:
        for path in files + [manifest]:
            tar.add(path, arcname=str(path.relative_to(root)), recursive=False)
    archive_sha = digest(archive)
    destination = REMOTE_BASE + "/drafter-full-training-" + plan_sha256[:12] + "-retained"
    remote("spark2", f"import pathlib;pathlib.Path({destination!r}).mkdir()")
    subprocess.run(["scp", str(archive), "spark2:" + destination + "/results-v1.tar.gz"],
                   check=True, timeout=3600)
    script = f'''import pathlib,json,hashlib,tarfile
p=pathlib.Path({destination!r});archive=p/'results-v1.tar.gz'
assert hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()=={archive_sha!r}
found={{}};manifest=None
with tarfile.open(archive,'r|gz') as tar:
 for member in tar:
  assert member.isfile()
  f=tar.extractfile(member)
  if member.name=='result-manifest.json':manifest=json.load(f)
  else:found[member.name]={{'bytes':member.size,'sha256':hashlib.file_digest(f,'sha256').hexdigest()}}
assert found=={{r['path']:{{'bytes':r['bytes'],'sha256':r['sha256']}} for r in manifest['files']}}
r={{'status':'verified','payloads':len(found),'archive_sha256':{archive_sha!r},'path':str(p)}}
(p/'backup-verification.json').write_text(json.dumps(r));print(json.dumps(r))
'''
    report = json.loads(remote("spark2", script))
    (root / "backup-verification.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def prepare(capture: Path, output: Path, inputs: dict[str, Path]) -> dict[str, Any]:
    """Freeze the trainer and paths after the complete capture plan has been prepared."""
    manifest = json.loads((capture / "source-manifest.json").read_bytes())
    if not all(digest(capture / name) == sha for name, sha in manifest.items()):
        raise ValueError("Capture sources changed")
    plan = json.loads((capture / "plan.json").read_bytes())
    protocol = json.loads((capture / "training-protocol.json").read_bytes())
    if (plan["revision"] != "drafter-full-capture-v1"
            or len(plan["calibration_cases"]) != 4200
            or protocol["capture_plan_sha256"] != digest(capture / "plan.json")
            or not plan.get("ordering_source_sha256")):
        raise ValueError("Require the corrected frozen full capture")
    output.mkdir(parents=True, exist_ok=False)
    package = output / "frozen/integrations/glm53_mxfp4"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    for module in MODULES:
        shutil.copyfile(Path(__file__).with_name(module + ".py"), package / (module + ".py"))
    for name in ("plan.json", "training-protocol.json", "serving-thresholds.json"):
        shutil.copyfile(capture / name, output / name)
    config = {
        "remote_root": REMOTE_BASE + "/drafter-full-capture-v1",
        "plan_sha256": digest(output / "plan.json"),
        "protocol_sha256": digest(output / "training-protocol.json"),
        "wait_seconds": plan["outage_seconds"] + 1800,
        "inputs": {key: str(path.resolve(strict=True)) for key, path in inputs.items()},
        "input_metadata_sha256": {
            str(path.resolve()): digest(path) for path in (
                inputs["weights"] / "training-rows.json", inputs["checkpoint"] / "config.json",
                inputs["native"] / "config.json",
            )
        },
        "sealed_test_opened": False, "production_mutations": False,
    }
    (output / "continuation.json").write_text(json.dumps(config, indent=2) + "\n")
    frozen = {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*"))
              if p.is_file()}
    (output / "continuation-source-manifest.json").write_text(json.dumps(frozen, indent=2) + "\n")
    return {"status": "prepared", "plan_sha256": config["plan_sha256"],
            "frozen_files": len(frozen), "started": False}


def run(config_path: Path) -> None:
    """Wait for restored paired capture, transfer verified data, train and conditionally export."""
    root = config_path.resolve().parent
    sources = json.loads((root / "continuation-source-manifest.json").read_bytes())
    if not all(digest(root / name) == sha for name, sha in sources.items()):
        raise ValueError("Frozen continuation sources changed")
    config = json.loads(config_path.read_bytes())
    plan = json.loads((root / "plan.json").read_bytes())
    state_path = root / "continuation-status.json"
    if state_path.exists():
        raise FileExistsError(state_path)
    started = time.time()
    state: dict[str, Any] = {"pid": os.getpid(), "started_epoch": started,
                            "plan_sha256": config["plan_sha256"], "sealed_test_opened": False}

    def update(phase: str, **values: Any) -> None:
        state.update(phase=phase, updated_epoch=time.time(), **values)
        temp = state_path.with_suffix(".incomplete")
        temp.write_text(json.dumps(state, indent=2) + "\n")
        temp.replace(state_path)
        print(json.dumps({"phase": phase, "epoch": state["updated_epoch"], **values}), flush=True)

    try:
        update("waiting_for_verified_capture")
        script = f'''import pathlib,json,hashlib
p=pathlib.Path({config["remote_root"]!r});f=p/'controller.json'
s=json.loads(f.read_bytes()) if f.exists() else {{'phase':'not_started'}}
post=p/'postprocess-complete.json';m=p/'ct-measurement.json';done=p/'run-prepared-status.json'
print(json.dumps({{'phase':s['phase'],'service_paused':s.get('service_paused'),
 'restoration':s.get('restoration'), 'error':s.get('error'),
 'requests':len(json.loads(m.read_bytes())['cases']) if m.exists() else 0,
 'postprocess':json.loads(post.read_bytes()) if post.exists() else None,
 'runner':json.loads(done.read_bytes()) if done.exists() else None,
 'plan_sha256':hashlib.file_digest((p/'plan.json').open('rb'),'sha256').hexdigest()}}))
'''
        while True:
            status = json.loads(remote("spark", script))
            if status["plan_sha256"] != config["plan_sha256"]:
                raise ValueError("Remote capture plan changed")
            if status["phase"] in ("failed_restored", "preparation_failed"):
                raise ValueError(f"Capture did not qualify: {status['error']}")
            runner = status["runner"]
            if runner and runner["step"] == "finalize" and runner["returncode"]:
                raise ValueError("Capture finalization failed; preserve all data and skip training")
            if status["phase"] == "complete_restored" and status["postprocess"]:
                restored = status["restoration"]
                if (status["service_paused"] or not restored["health_200"]
                        or not restored["originals_running"]
                        or status["postprocess"]["status"] != "verified"):
                    raise ValueError("Unverified restoration or capture finalization")
                break
            if time.time() - started > config["wait_seconds"]:
                raise TimeoutError("Verified capture did not finish within its frozen deadline")
            update("waiting_for_verified_capture", capture_phase=status["phase"],
                   captured_requests=status["requests"])
            time.sleep(30)
        expected = status["postprocess"]["manifest_sha256"]
        script = f'''import pathlib,hashlib,json
p=pathlib.Path({config["remote_root"]!r})/'accepted/manifest.json'
print(json.dumps({{'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()}}))
'''
        if any(json.loads(remote(host, script))["sha256"] != expected
               for host in ("spark", "spark2")):
            raise ValueError("Paired published manifests differ")
        update("transferring_verified_features", manifest_sha256=expected)
        needed = plan["full_corpus"]["accepted_feature_bytes"] * 1.1 + 10 * 1024**3
        if shutil.disk_usage(root).free < needed:
            raise ValueError("Insufficient local space; preserve remote accepted copies")
        destination = root / "capture"
        destination.mkdir()
        script = f'''import pathlib,json,tarfile,sys
p=pathlib.Path({config["remote_root"]!r})/'accepted'
m=json.loads((p/'manifest.json').read_bytes())
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as t:
 t.add(p/'manifest.json',arcname='accepted/manifest.json',recursive=False)
 for row in m['rows']:
  assert pathlib.Path(row['path']).name==row['path']
  t.add(p/row['path'],arcname='accepted/'+row['path'],recursive=False)
'''
        with subprocess.Popen(
            ["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30",
             "-o", "ServerAliveCountMax=3", "spark", "python3", "-"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        ) as transfer:
            assert transfer.stdin is not None and transfer.stdout is not None
            transfer.stdin.write(script.encode())
            transfer.stdin.close()
            with tarfile.open(fileobj=transfer.stdout, mode="r|") as archive:
                archive.extractall(destination, filter="data")
            if transfer.wait(timeout=60):
                raise ValueError("Verified dataset transfer failed")
        dataset = destination / "accepted"
        receipt = verify_dataset(dataset, plan, expected, config["plan_sha256"])
        (root / "local-dataset-verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
        if (not all(digest(root / name) == sha for name, sha in sources.items())
                or not all(digest(Path(name)) == sha
                           for name, sha in config["input_metadata_sha256"].items())):
            raise ValueError("Frozen trainer, protocol or target-weight metadata changed")
        inputs = config["inputs"]
        env = dict(os.environ, PYTHONPATH=inputs["deps"])
        argv = [sys.executable, "-u", "-m", "integrations.glm53_mxfp4.train_drafter_context_pilot",
                "--dataset", str(dataset), "--protocol", str(root / "training-protocol.json"),
                "--output", str(root / "run")]
        for name in ("checkpoint", "checkout", "weights"):
            argv += ["--" + name, inputs[name]]
        update("training", local_dataset=receipt)
        with (root / "training.log").open("xb") as log:
            subprocess.run(argv, cwd=root / "frozen", env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        report = json.loads((root / "run/report.json").read_bytes())
        if report["status"] != "complete" or report["protocol_sha256"] != config["protocol_sha256"]:
            raise ValueError("Training did not complete the frozen protocol")
        if report["offline_screen_pass"]:
            update("exporting_selected_candidate", selected_step=report["selected"]["step"])
            argv = [sys.executable, "-u", "-m", "integrations.glm53_mxfp4.export_drafter_context",
                    "--source", inputs["native"], "--run", str(root / "run"),
                    "--output", str(root / "native-candidate")]
            with (root / "export.log").open("xb") as log:
                subprocess.run(argv, cwd=root / "frozen", env=env, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
        update("retaining_training_results", offline_screen_pass=report["offline_screen_pass"])
        backup = backup_results(root, config["plan_sha256"])
        update("complete", offline_screen_pass=report["offline_screen_pass"], backup=backup,
               native_reload_performed=False, speed_gain_proven=False)
    except BaseException as exc:
        update("failed", error=repr(exc))
        raise


def main() -> None:
    """Prepare or run one retained continuation; never promote a serving candidate."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prepare_parser = sub.add_parser("prepare")
    for name in ("capture", "output", "checkpoint", "checkout", "weights", "native", "deps"):
        prepare_parser.add_argument("--" + name, type=Path, required=True)
    sub.add_parser("run").add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "run":
        run(args.config)
    else:
        inputs = {name: getattr(args, name) for name in
                  ("checkpoint", "checkout", "weights", "native", "deps")}
        print(json.dumps(prepare(args.capture, args.output, inputs), indent=2))


if __name__ == "__main__":
    main()
