"""Safetensors shard discovery and streaming reads.

The streaming model loads one shard at a time, quantizes its targeted tensors
on-device, writes the result, and frees memory before moving to the next shard
— so models larger than any single machine can be quantized without loading
the full checkpoint into RAM/VRAM.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from safetensors import safe_open

__all__ = [
    "ShardFile",
    "discover_shards",
    "iter_tensors",
]


@dataclass(frozen=True)
class ShardFile:
    """A single safetensors shard file with its tensor name mapping."""

    path: Path
    weight_map: dict[str, str]  # tensor name -> shard filename (for the index)


def discover_shards(model_dir: str | Path) -> tuple[list[ShardFile], dict[str, str] | None]:
    """Discover shard files, preferring the safetensors index when present.

    Returns:
        (shards, weight_map): shards is the ordered list of shard files;
        weight_map is the full tensor->shard map from the index (or None if no
        index exists, in which case a per-shard map is built lazily).
    """
    model_dir_p = Path(model_dir)
    index_path = model_dir_p / "model.safetensors.index.json"
    weight_map: dict[str, str] | None = None

    if index_path.exists():
        index = json.loads(index_path.read_text())
        weight_map = index.get("weight_map", {})
        # Order shards by their first appearance in the map for determinism
        filenames = list(dict.fromkeys(weight_map.values()))
        shards = [ShardFile(model_dir_p / f, weight_map) for f in filenames]
    else:
        files = sorted(model_dir_p.glob("*.safetensors"))
        shards = [ShardFile(p, {}) for p in files]

    if not shards:
        raise FileNotFoundError(
            f"No safetensors shards found in {model_dir_p} "
            "(expected *.safetensors or model.safetensors.index.json)"
        )
    return shards, weight_map


def iter_tensors(
    shard: ShardFile,
    *,
    device: torch.device | str = "cpu",
) -> list[tuple[str, torch.Tensor]]:
    """Load all tensors from a shard into memory (for small shards).

    For true streaming of very large shards, prefer ``safe_open`` directly with
    ``get_tensor`` per tensor — this helper returns everything at once for
    simplicity and is suitable when a shard fits in device memory.
    """
    out: list[tuple[str, torch.Tensor]] = []
    with safe_open(str(shard.path), framework="pt", device=str(device)) as sf:
        for key in sf:
            out.append((key, sf.get_tensor(key)))
    return out


def shard_tensor_keys(shard: ShardFile) -> list[str]:
    """Return the tensor keys in a shard without loading data (header only)."""
    with safe_open(str(shard.path), framework="pt", device="cpu") as sf:
        return list(sf.keys())


def read_tensor(
    shard: ShardFile,
    key: str,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Read a single tensor from a shard (lazy — data pulled on access)."""
    with safe_open(str(shard.path), framework="pt", device=str(device)) as sf:
        return cast(torch.Tensor, sf.get_tensor(key))
