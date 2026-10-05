"""Guard qualification and publish complete verified Kolibri release artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

REPOSITORY = "kirillbilchenko/Kolibri-1-MXFP4-MxWave"


def _read(root: Path, name: str) -> dict:
    return json.loads((root / name).read_text())


def _require_qualified(root: Path, *, experimental: bool = False) -> None:
    for name, status in (
        ("conversion-result.json", "complete"), ("payload-verification.json", "passed"),
    ):
        if _read(root, name).get("status") != status:
            raise ValueError(f"Release prerequisite failed: {name}")
    for name in ("quality-comparison.json", "qualification-result.json"):
        status = _read(root, name).get("status")
        if status != "passed" and not (experimental and status == "failed"):
            raise ValueError(f"Release prerequisite failed: {name}")
    if experimental:
        quality = _read(root, "quality-comparison.json")
        gate = quality["gate"]
        mean_kl = quality["mean_next_token_kl"]
        if not math.isfinite(mean_kl) or mean_kl < 0:
            raise ValueError("Experimental release requires a finite nonnegative KL measurement")
        aggregate_delta = quality["domains"]["all"]["delta_mean_nll"]
        if not math.isfinite(aggregate_delta) or aggregate_delta > gate["max_mean_delta_nll"]:
            raise ValueError("Experimental release exceeds the frozen aggregate likelihood bound")
        domain_deltas = [quality["domains"][name]["delta_mean_nll"]
                         for name in ("en", "de", "code")]
        if any(not math.isfinite(delta) or delta > gate["max_domain_delta_nll"]
               for delta in domain_deltas):
            raise ValueError("Experimental release exceeds a frozen domain likelihood bound")
        expected_checks = {row["id"] for row in _read(root, "validation/protocol.json")["behavior_cases"]}
        checks = quality["behavior"]["mse"]
        if (not expected_checks or len(checks) != len(expected_checks)
                or {row["id"] for row in checks} != expected_checks
                or not all(row["passed"] for row in checks)):
            raise ValueError("Experimental release failed a behavior check")


def stage(root: Path) -> None:
    """Upload privately and verify every remote shard before changing public visibility."""
    from huggingface_hub import HfApi

    prepared = _read(root, "publication-prepared.json")
    _require_qualified(root, experimental=prepared["experimental"])
    manifest_bytes = (root / "checkpoint-mse/mxwave-manifest.json").read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != prepared["manifest_sha256"]:
        raise ValueError("Prepared checkpoint manifest changed")
    if hashlib.sha256((root / "quality-comparison.json").read_bytes()).hexdigest() != prepared["quality_sha256"]:
        raise ValueError("Prepared quality evidence changed")
    source = _read(root, "checkpoint-mse/validation/release-source.json")
    archive_sha = hashlib.sha256((root / "checkpoint-mse/mxwave-source.tar.gz").read_bytes()).hexdigest()
    if archive_sha != source["archive_sha256"]:
        raise ValueError("Release source archive does not match its provenance")
    candidate = _read(root, "measurements/mse.json")
    if source["frozen_evaluator_sha256"] != candidate["executed_source_sha256"]:
        raise ValueError("Archived evaluator differs from the executed quality screen")
    if "runtime_wrapper_sha256" in candidate:
        wrapper = source["source_files_sha256"]["scripts/evaluate_kolibri_mixed.py"]
        if wrapper != candidate["runtime_wrapper_sha256"]:
            raise ValueError("Archived mixed wrapper differs from the measured execution")
    api = HfApi()
    if api.whoami()["name"] != REPOSITORY.split("/")[0]:
        raise ValueError("Authenticated account does not match the release namespace")
    marker = root / "upload-repository.json"
    if marker.exists():
        if _read(root, marker.name) != prepared:
            raise ValueError("Existing upload intent belongs to another checkpoint")
    else:
        api.create_repo(REPOSITORY, repo_type="model", private=True, exist_ok=False)
        marker.write_text(json.dumps(prepared, indent=2) + "\n")
    api.upload_large_folder(
        REPOSITORY, folder_path=root / "checkpoint-mse", repo_type="model", num_workers=4,
        ignore_patterns=[".cache/**", ".mxwave*", "*.incomplete", "mxwave-run.json",
                         "mxwave-shard-integrity.json"],
    )
    info = api.model_info(REPOSITORY, files_metadata=True)
    manifest = json.loads(manifest_bytes)
    remote_files = {file.rfilename: file for file in info.siblings}
    for name, expected in manifest["shard_integrity"]["per_shard"].items():
        remote = remote_files[name]
        if remote.size != expected["bytes"] or remote.lfs is None or remote.lfs.sha256 != expected["sha256"]:
            raise ValueError(f"Remote shard size or SHA256 differs: {name}")
    record = {**prepared, "status": "staged-private", "revision": info.sha,
              "source_archive_sha256": archive_sha,
              "url": f"https://huggingface.co/{REPOSITORY}", "verified_shards": len(manifest["shard_integrity"]["per_shard"])}
    (root / "publication-staged.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


def publish(root: Path) -> None:
    """Make a fully uploaded and verified release public without overwriting another model."""
    from huggingface_hub import HfApi

    prepared = _read(root, "publication-prepared.json")
    _require_qualified(root, experimental=prepared["experimental"])
    staged = _read(root, "publication-staged.json")
    if staged["manifest_sha256"] != prepared["manifest_sha256"] or staged["verified_shards"] != 32:
        raise ValueError("The staged artifact is not the complete prepared checkpoint")
    api = HfApi()
    info = api.model_info(REPOSITORY)
    if info.sha != staged["revision"]:
        raise ValueError("The staged remote repository changed after hash verification")
    api.update_repo_settings(REPOSITORY, private=False)
    record = {**staged, "status": "published"}
    (root / "publication-result.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)
