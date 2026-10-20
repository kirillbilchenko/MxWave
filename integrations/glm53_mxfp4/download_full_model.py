"""Download the pinned public checkpoint on Spark2 without CUDA or full-file buffers."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import time
from pathlib import Path
from typing import Any

REPOSITORY = "INCModel3/GLM-5.3-Flash-MXFP4-Mixed-CT-AutoRound"
REVISION = "35d1b0d5cb4a0db075fd2749c897e970dc15bfa5"


def file_sha256(path: Path) -> str:
    """Hash a source file with bounded host memory."""
    result = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024**2):
            result.update(chunk)
    return result.hexdigest()


def main() -> None:
    """Resume public downloads and verify all shard hashes against the audited revision."""
    from huggingface_hub import hf_hub_download, list_repo_files

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_bytes())
    if inventory["repository"] != REPOSITORY or inventory["revision"] != REVISION:
        raise ValueError("Full-model preparation requires the audited source revision")
    args.output.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "status": "downloading", "repository": REPOSITORY, "revision": REVISION,
        "inventory_sha256": file_sha256(args.inventory), "verified_shards": [],
        "verified_bytes": 0, "total_bytes": inventory["total_safetensors_file_bytes"],
        "started_epoch": time.time(),
    }

    def save() -> None:
        state["updated_epoch"] = time.time()
        target = args.output.parent / "download-state.json"
        temporary = target.with_suffix(".incomplete")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(target)

    def download(name: str) -> Path:
        return Path(hf_hub_download(REPOSITORY, name, revision=REVISION,
                                   token=False, local_dir=args.output))

    def shard(record: dict[str, Any]) -> dict[str, Any]:
        path = download(record["name"])
        if path.stat().st_size != record["bytes"] or file_sha256(path) != record["sha256"]:
            raise ValueError(f"Full-model shard differs from audited source: {path.name}")
        return record

    save()
    try:
        metadata = [n for n in list_repo_files(REPOSITORY, revision=REVISION, token=False)
                    if "/" not in n and not n.endswith(".safetensors")]
        for name in metadata:
            if name.endswith((".json", ".jinja", ".model", ".tiktoken", ".md", ".txt")):
                download(name)
        for name, expected in {
            "config.json": "194d290ebb0f19aedc60d24ea9e236e33c86616e460f4a108a842f6fd5b8b198",
            "model.safetensors.index.json":
                "6d1364a27eef78675390422b1b5230601817fb7460ce816951b2cd515934c85d",
        }.items():
            if file_sha256(args.output / name) != expected:
                raise ValueError(f"Pinned model metadata changed: {name}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(shard, x) for x in inventory["shards"]]
            for future in concurrent.futures.as_completed(futures):
                record = future.result()
                state["verified_shards"].append(record)
                state["verified_bytes"] += record["bytes"]
                save()
                print(json.dumps({"verified_shards": len(state["verified_shards"]),
                                  "verified_bytes": state["verified_bytes"]}), flush=True)
        if state["verified_bytes"] != state["total_bytes"]:
            raise ValueError("Verified shard total differs from the audited inventory")
        state["status"] = "complete"
    except Exception as exc:
        state.update(status="failed", error=repr(exc))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
