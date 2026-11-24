"""Read-only freeze of the user-selected live service; never export credentials."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from control_stock_trial import identity, write


def freeze(root: Path, output: Path, inventory: Path, rank: int) -> None:
    """Verify all target shards and snapshot runtime inputs without restarting serving."""
    output.mkdir(parents=True, exist_ok=False)
    (output / "payloads").mkdir()
    plan = json.loads((root / "prepared.json").read_bytes())
    entry = next(e for e in plan["owned"] if e["rank"] == rank)

    def inspect() -> dict[str, Any]:
        return json.loads(subprocess.check_output(["docker", "inspect", entry["name"]]))[0]

    container = inspect()
    if not container["State"]["Running"]:
        raise RuntimeError("The selected CT container is not running")
    before = identity(container)
    mounts = {m["Destination"]: Path(m["Source"]) for m in container["Mounts"]}
    env = dict(e.split("=", 1) for e in container["Config"]["Env"])
    files: dict[str, dict[str, Any]] = {}

    def retain(path: Path, expected: str | None = None, *, copy_small: bool = True) -> None:
        key = str(path)
        if key in files:
            return
        stat = path.stat()
        with path.open("rb") as stream:
            sha = hashlib.file_digest(stream, "sha256").hexdigest()
        after = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"Input changed during freeze: {path.name}")
        if expected is not None and sha != expected:
            raise RuntimeError(f"Checkpoint digest mismatch: {path.name}")
        saved = copy_small and stat.st_size <= 64 * 1024**2
        if saved and not (output / "payloads" / sha).exists():
            shutil.copyfile(path, output / "payloads" / sha)
        files[key] = {"sha256": sha, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                      "snapshot": "payloads/" + sha if saved else None}

    def source_tree(path: Path, *, all_files: bool = False) -> None:
        for p in sorted(path.rglob("*")):
            if (p.is_file() and not any(x in ("cache", "__pycache__", ".git", "logs")
                                        for x in p.relative_to(path).parts)
                    and (all_files or p.suffix in (".py", ".sh", ".so", ".json", ".h", ".cuh")
                         or p.name in ("METADATA", "entry_points.txt"))):
                retain(p)

    started = time.time()
    shards = json.loads(inventory.read_bytes())["shards"]
    for number, shard in enumerate(shards, 1):
        path = mounts["/model"] / shard["name"]
        if path.stat().st_size != shard["bytes"]:
            raise RuntimeError(f"Checkpoint size mismatch: {path.name}")
        retain(path, shard["sha256"], copy_small=False)
        write(output / "progress.json", {"rank": rank, "verified_shards": number,
                                         "total_shards": len(shards)})
    for p in mounts["/model"].iterdir():
        if p.is_file() and p.suffix != ".safetensors":
            retain(p)
    source_tree(mounts["/draft"], all_files=True)
    for destination in ("/overlay", "/trial", "/ct-code"):
        source_tree(mounts[destination])
    for p in mounts["/kda-runtime"].iterdir():
        if p.is_file() and (p.suffix == ".py" or p.name == "rounding-reference.json"):
            retain(p)
    # Record only artifact paths and PYTHONPATH inputs, never arbitrary environment values.
    values = [v for k, v in env.items() if k.startswith("MXWAVE_") and v.startswith("/")]
    values += env.get("PYTHONPATH", "").split(":")
    for value in values:
        for destination in sorted(mounts, key=len, reverse=True):
            if value == destination or value.startswith(destination + "/"):
                if destination in ("/model", "/reports"):
                    break
                path = mounts[destination] / value.removeprefix(destination).lstrip("/")
                if path.is_file():
                    retain(path)
                elif path.is_dir():
                    source_tree(path, all_files=path.name == "vocab-v2")
                break
    current = inspect()
    if identity(current) != before or not current["State"]["Running"]:
        raise RuntimeError("Service changed during read-only freeze")
    result = {"schema": 1, "status": "verified", "rank": rank,
              "created_epoch": time.time(), "seconds": time.time() - started,
              "identity": before, "container_name": entry["name"],
              "command": container["Config"]["Cmd"], "files": files,
              "target_shards_verified": len(shards), "credentials_exported": False,
              "service_restarted": False,
              "large_payloads": "Hash-verified in place; existing copies remain on both nodes"}
    write(output / "freeze.json", result)
    shutil.copyfile(inventory, output / "model-inventory.json")
    shutil.copyfile(root / "live-verification.json", output / "live-verification.json")
    print(json.dumps({k: v for k, v in result.items() if k not in ("files", "command")}))


def main() -> None:
    """Capture one host's immutable receipt and small runtime payload copies."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--rank", type=int, choices=(0, 1), required=True)
    args = p.parse_args()
    freeze(args.root, args.output, args.inventory, args.rank)


if __name__ == "__main__":
    main()
