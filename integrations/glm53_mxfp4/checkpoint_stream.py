"""Hash-verified safetensors iteration with at most one temporary source shard."""

from __future__ import annotations

import hashlib
import re
import tempfile
import urllib.request
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


def checked_shards(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    """Reject ambiguous names, hashes, duplicates and inconsistent byte totals."""
    shards = inventory["shards"]
    names = set()
    for shard in shards:
        name = shard["name"]
        if (not re.fullmatch(r"model-\d{5}-of-\d{5}\.safetensors", name)
                or not re.fullmatch(r"[0-9a-f]{64}", shard["sha256"])
                or not isinstance(shard["bytes"], int) or shard["bytes"] <= 0 or name in names):
            raise ValueError("Invalid or duplicate checkpoint shard")
        names.add(name)
    if (len(shards) != inventory["source_shards"]
            or sum(x["bytes"] for x in shards) != inventory["total_safetensors_file_bytes"]):
        raise ValueError("Checkpoint inventory totals differ")
    return shards


def verify_file(path: Path, record: dict[str, Any]) -> None:
    """Verify the complete file before exposing any tensor to a model loader."""
    result = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024**2):
            result.update(chunk)
    if path.stat().st_size != record["bytes"] or result.hexdigest() != record["sha256"]:
        raise ValueError(f"Checkpoint size/hash mismatch: {record['name']}")


def iter_checkpoint(
    inventory: dict[str, Any], local_source: Path, scratch: Path, *,
    remote_base: str | None = None,
    after_shard: Callable[[], None] | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield exact CPU tensors from a local checkpoint or one verified HTTP shard at a time.

    Only this iterator's temporary directory is removed. Source checkpoint files
    are never changed. The caller must finish asynchronous copies in after_shard
    before the mmap is closed and its temporary file released.
    """
    shards = checked_shards(inventory)
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mxwave-source-", dir=scratch) as temporary:
        for record in shards:
            path = local_source / record["name"]
            staged = not path.is_file()
            if staged:
                if remote_base is None:
                    raise FileNotFoundError(f"Missing local checkpoint shard: {path.name}")
                path = Path(temporary) / record["name"]
                request = urllib.request.Request(remote_base.rstrip("/") + "/" + path.name)
                with urllib.request.urlopen(request, timeout=180) as response, path.open("wb") as f:
                    while chunk := response.read(8 * 1024**2):
                        f.write(chunk)
                        if f.tell() > record["bytes"]:
                            raise ValueError("Remote checkpoint exceeds its audited byte count")
            verify_file(path, record)
            tensors = 0
            with safe_open(path, framework="pt", device="cpu") as handle:
                for name in handle.keys():  # noqa: SIM118 — safe_open is not a dictionary
                    tensor = handle.get_tensor(name)
                    yield name, tensor
                    del tensor
                    tensors += 1
                if after_shard is not None:
                    after_shard()
            if staged:
                path.unlink()
            if progress is not None:
                progress({"shard": record["name"], "bytes": record["bytes"],
                          "sha256": record["sha256"], "tensors": tensors})
