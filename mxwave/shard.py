"""Safetensors shard discovery and streaming reads.

The streaming model inspects one shard at a time and reads large target matrices
by bounded row ranges. It writes the completed output shard before advancing,
so neither the full checkpoint nor a full target matrix must enter device memory.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from safetensors import safe_open

__all__ = [
    "ShardFile",
    "TensorInfo",
    "discover_shards",
    "iter_tensors",
    "read_tensor",
    "read_tensor_row_range",
    "read_tensor_rows",
    "shard_tensor_info",
    "tensor_payload_sha256",
]


@dataclass(frozen=True)
class ShardFile:
    """A single safetensors shard file with its tensor name mapping."""

    path: Path
    weight_map: dict[str, str]  # tensor name -> shard filename (for the index)


@dataclass(frozen=True)
class TensorInfo:
    """Header-only metadata for one safetensors tensor."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    data_offsets: tuple[int, int]


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


def shard_tensor_info(shard: ShardFile) -> dict[str, TensorInfo]:
    """Read shapes, dtypes, and offsets from a shard without loading tensor data."""
    with shard.path.open("rb") as stream:
        raw_length = stream.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"Invalid safetensors header in {shard.path}")
        header_length = struct.unpack("<Q", raw_length)[0]
        if header_length > 100_000_000:
            raise ValueError(f"Unreasonably large safetensors header in {shard.path}")
        raw_header = stream.read(header_length)
    try:
        header: Any = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid safetensors header JSON in {shard.path}") from exc
    if not isinstance(header, dict):
        raise TypeError(f"Invalid safetensors header object in {shard.path}")

    result: dict[str, TensorInfo] = {}
    for name, raw_info in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not isinstance(raw_info, dict):
            raise TypeError(f"Invalid tensor entry in {shard.path}")
        raw_shape = raw_info.get("shape")
        raw_dtype = raw_info.get("dtype")
        raw_offsets = raw_info.get("data_offsets")
        if (
            not isinstance(raw_shape, list)
            or not all(isinstance(dim, int) and dim >= 0 for dim in raw_shape)
            or not isinstance(raw_dtype, str)
            or not isinstance(raw_offsets, list)
            or len(raw_offsets) != 2
            or not all(isinstance(offset, int) and offset >= 0 for offset in raw_offsets)
            or raw_offsets[0] > raw_offsets[1]
        ):
            raise ValueError(f"Invalid metadata for tensor {name!r} in {shard.path}")
        result[name] = TensorInfo(
            name=name,
            shape=tuple(raw_shape),
            dtype=raw_dtype,
            data_offsets=(raw_offsets[0], raw_offsets[1]),
        )
    maximum_end = max((info.data_offsets[1] for info in result.values()), default=0)
    expected_size = 8 + header_length + maximum_end
    if shard.path.stat().st_size != expected_size:
        raise ValueError(
            f"Safetensors size mismatch in {shard.path}: expected {expected_size}, "
            f"found {shard.path.stat().st_size}"
        )
    return result


def tensor_payload_sha256(
    shard: ShardFile,
    key: str,
    *,
    chunk_size_bytes: int = 8 * 1024**2,
) -> str:
    """Hash one tensor's raw safetensors payload with bounded host memory.

    Safetensors offsets are relative to the start of the data section, so the
    header length is included when seeking to the tensor.  Hashing raw bytes
    verifies bitwise preservation, including floating-point NaN payloads, and
    avoids materializing the tensor through PyTorch.
    """
    if not isinstance(chunk_size_bytes, int) or chunk_size_bytes <= 0:
        raise ValueError("chunk_size_bytes must be a positive integer")
    info = shard_tensor_info(shard).get(key)
    if info is None:
        raise KeyError(f"Tensor {key!r} is absent from {shard.path}")

    digest = hashlib.sha256()
    with shard.path.open("rb") as stream:
        raw_length = stream.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"Invalid safetensors header in {shard.path}")
        header_length = struct.unpack("<Q", raw_length)[0]
        start, stop = info.data_offsets
        stream.seek(8 + header_length + start)
        remaining = stop - start
        while remaining:
            chunk = stream.read(min(remaining, chunk_size_bytes))
            if not chunk:
                raise ValueError(
                    f"Unexpected end of safetensors payload for {key!r} in {shard.path}"
                )
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def read_tensor(
    shard: ShardFile,
    key: str,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Read a single tensor from a shard (lazy — data pulled on access)."""
    with safe_open(str(shard.path), framework="pt", device=str(device)) as sf:
        return cast(torch.Tensor, sf.get_tensor(key))


def read_tensor_rows(
    shard: ShardFile,
    key: str,
    rows: int,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Read only the leading rows of a matrix from a safetensors shard."""
    if rows <= 0:
        raise ValueError("rows must be positive")
    return read_tensor_row_range(shard, key, 0, rows, device=device)


def read_tensor_row_range(
    shard: ShardFile,
    key: str,
    start: int,
    stop: int,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Read the half-open row range ``[start, stop)`` of a matrix."""
    if start < 0 or stop <= start:
        raise ValueError("row range must satisfy 0 <= start < stop")
    with safe_open(str(shard.path), framework="pt", device="cpu") as sf:
        sliced = cast(torch.Tensor, sf.get_slice(key)[start:stop])
    if sliced.ndim != 2 or sliced.shape[0] != stop - start:
        raise ValueError(
            f"Tensor {key!r} cannot provide requested matrix rows [{start}, {stop})"
        )
    return sliced.to(device)
