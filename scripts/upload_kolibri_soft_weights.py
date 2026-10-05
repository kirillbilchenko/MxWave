"""Privately upload complete verified soft-RMS weights while serving qualification runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from publish_kolibri_release import REPOSITORY


def main() -> None:
    """Wait for the complete export, upload privately, and verify all remote shard hashes."""
    from huggingface_hub import HfApi

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    marker = root / "checkpoint-soft-rms-export.json"
    deadline = time.monotonic() + 3600
    while not marker.exists():
        if time.monotonic() > deadline:
            raise TimeoutError("Complete soft-RMS export was not produced within one hour")
        time.sleep(10)
    export = json.loads(marker.read_text())
    output = root / "checkpoint-soft-rms"
    raw = (output / "mxwave-manifest.json").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    manifest = json.loads(raw)
    if (
        export["status"] != "complete"
        or export["shards"] != 32
        or export["verified_payloads"] != 116303
        or digest != export["manifest_sha256"]
        or manifest["config_coverage_gaps"]
        or len(manifest["shard_integrity"]["per_shard"]) != 32
    ):
        raise ValueError("Private upload requires the complete verified soft-RMS export")
    publication = root / "soft-rms-publication"
    publication.mkdir(exist_ok=True)
    intent = publication / "private-weight-upload.json"
    identity = {
        "repository": REPOSITORY,
        "manifest_sha256": digest,
        "trial_specification_sha256": export["trial_specification_sha256"],
    }
    api = HfApi()
    if api.whoami()["name"] != REPOSITORY.split("/")[0]:
        raise ValueError("Authenticated account differs from the release namespace")
    if intent.exists():
        previous = json.loads(intent.read_text())
        if any(previous[key] != value for key, value in identity.items()):
            raise ValueError("Existing private upload belongs to another checkpoint")
        if not api.model_info(REPOSITORY).private:
            raise ValueError("Weight preupload must remain private")
    else:
        api.create_repo(REPOSITORY, repo_type="model", private=True, exist_ok=False)
        intent.write_text(json.dumps({**identity, "status": "uploading-private"}, indent=2) + "\n")
    print("Uploading 32 verified soft-RMS weight shards privately", flush=True)
    api.upload_large_folder(
        REPOSITORY,
        repo_type="model",
        folder_path=output,
        allow_patterns=["model-*-of-00032.safetensors"],
        ignore_patterns=[".cache/**", "*.incomplete"],
        num_workers=4,
    )
    info = api.model_info(REPOSITORY, files_metadata=True)
    if not info.private:
        raise ValueError("Private weight upload visibility changed unexpectedly")
    remote = {file.rfilename: file for file in info.siblings}
    for name, expected in manifest["shard_integrity"]["per_shard"].items():
        file = remote[name]
        if (
            file.size != expected["bytes"]
            or file.lfs is None
            or file.lfs.sha256 != expected["sha256"]
        ):
            raise ValueError(f"Uploaded weight shard differs: {name}")
    record = {
        **identity,
        "status": "weights-verified-private",
        "verified_shards": 32,
        "revision": info.sha,
        "published": False,
    }
    intent.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
