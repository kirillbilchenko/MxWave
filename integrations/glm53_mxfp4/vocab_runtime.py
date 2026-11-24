"""Lossless TP2 vocabulary storage with stock vLLM communication outside each method."""

from __future__ import annotations

import hashlib
import json
import os
import weakref
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from prepare_bf16_attention import tensor_digest
from safetensors.torch import load_file
from vocab_policy import VocabTarget, validate_layer
from zipserv_contract import require_qualified_exponent_base

from mxwave.lossless_bf16_rows import read_from
from mxwave.lossless_bf16_rows_gpu import GpuBf16Rows, gather_rows, upload_rows

ROW_BUFFERS = ("base", "row_offsets", "block_offsets", "sign_mantissa", "planes", "exceptions")
_ENTRIES: list[dict[str, Any]] = []
_OWNERS: list[weakref.ReferenceType[torch.nn.Module]] = []


def file_digest(path: Path) -> str:
    """Hash artifacts in bounded chunks rather than allocating a full second payload."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def manifest() -> dict[str, Any]:
    """Require an explicit hash-bound manifest and all four source-shard records."""
    root = Path(os.environ["MXWAVE_GLM53_VOCAB_ARTIFACTS"])
    path = root / "prepared.json"
    if file_digest(path) != os.environ["MXWAVE_GLM53_VOCAB_MANIFEST_SHA256"]:
        raise ValueError("Vocabulary preparation manifest changed")
    result = json.loads(path.read_bytes())
    if (result.get("status") != "packed_gpu_verification_pending"
            or {(row["kind"], row["rank"]) for row in result["cases"]}
            != {(kind, rank) for kind in ("head", "embedding") for rank in (0, 1)}
            or len(result["cases"]) != 4):
        raise ValueError("Require all four prepared vocabulary shards")
    for row in result["cases"]:
        if row["kind"] == "head":
            metadata = row.get("metadata")
            if not isinstance(metadata, list) or len(metadata) != 5:
                raise ValueError("Require complete native head metadata before loading")
            require_qualified_exponent_base(metadata[4], where=f"vocabulary head r{row['rank']}")
    return result


def packed_head(layer: torch.nn.Module) -> Any:
    """Return a non-owning view of the native head buffers."""
    from mxwave_zipserv import runtime

    require_qualified_exponent_base(layer.mxwave_vocab_metadata[4], where="loaded vocabulary head")
    return runtime.PackedWeight(tuple(getattr(layer, "mxwave_vocab_" + n)
                                     for n in runtime.BUFFER_NAMES), *layer.mxwave_vocab_metadata)


def packed_embedding(layer: torch.nn.Module) -> GpuBf16Rows:
    """Return a non-owning view of the row-addressable embedding buffers."""
    return GpuBf16Rows(*layer.mxwave_vocab_metadata, "0" * 64,
                      *(getattr(layer, "mxwave_vocab_" + n) for n in ROW_BUFFERS))


@torch.library.custom_op("mxwave_glm53::embedding_rows", mutates_args=(), device_types="cuda")
def embedding_rows(ids: torch.Tensor, buffers: list[torch.Tensor],
                   rows: int, width: int, block: int) -> torch.Tensor:
    """Decode local ids after the unchanged stock TP2 mask has substituted row zero."""
    packed = GpuBf16Rows(rows, width, block, "0" * 64, *buffers)
    return gather_rows(packed, ids, validate_range=False)


@embedding_rows.register_fake
def _fake_embedding(ids: torch.Tensor, buffers: list[torch.Tensor],
                    rows: int, width: int, block: int) -> torch.Tensor:
    return torch.empty((*ids.shape, width), device=ids.device, dtype=torch.bfloat16)


def prepare(layer: torch.nn.Module, target: VocabTarget) -> None:
    """Bind the actual loaded TP shard to its immutable CPU-prepared compressed artifact."""
    from kda_runtime import validate_runtime
    from mxwave_zipserv import runtime

    validate_runtime()
    validate_layer(layer, target)
    if (getattr(layer, "mxwave_vocab_metadata", None) is not None
            or not layer.mxwave_vocab_source_loaded or layer.weight.device.type != "cpu"
            or layer.weight.dtype != torch.bfloat16):
        raise ValueError("Vocabulary preparation requires one fully loaded CPU BF16 shard")
    record = next(row for row in manifest()["cases"]
                  if row["kind"] == target.kind and row["rank"] == layer.tp_rank)
    if (record["prefix"] != target.prefix or record["source_key"] != target.source_key
            or record["shape"] != [target.rows, target.width]
            or list(layer.weight.shape) != record["shape"]
            or tensor_digest(layer.weight.detach()) != record["source_sha256"]):
        raise ValueError("Loaded TP vocabulary shard differs from the prepared source")
    path = Path(os.environ["MXWAVE_GLM53_VOCAB_ARTIFACTS"]) / record["artifact"]
    if path.name != record["artifact"] or file_digest(path) != record["artifact_sha256"]:
        raise ValueError("Vocabulary artifact changed or uses a nonlocal path")
    device = torch.device("cuda", torch.cuda.current_device())
    if target.kind == "head":
        cpu_buffers = load_file(path, device="cpu")
        if set(cpu_buffers) != set(runtime.BUFFER_NAMES):
            raise ValueError("Head artifact must contain exactly the seven native buffers")
        packed = runtime.PackedWeight(tuple(cpu_buffers[n] for n in runtime.BUFFER_NAMES),
                                      *record["metadata"]).to(device)
        if (packed.m, packed.k) != (target.rows, target.width):
            raise ValueError("Prepared head dimensions changed")
        buffers, names = packed.buffers, runtime.BUFFER_NAMES
        retained = packed.storage_bytes
    else:
        cpu = read_from(path, verify_hashes=True)
        if ((cpu.rows, cpu.width, cpu.block_size) != tuple(record["metadata"])
                or cpu.source_sha256 != record["source_sha256"]):
            raise ValueError("Embedding artifact source binding changed")
        packed = upload_rows(cpu, device=device)
        buffers, names = packed.tensors(), ROW_BUFFERS
        retained = packed.resident_bytes
    if retained != record["retained_bytes"] or retained >= record["raw_bytes"]:
        raise ValueError("Vocabulary retained-byte gate failed")
    for name, buffer in zip(names, buffers, strict=True):
        layer.register_buffer("mxwave_vocab_" + name, buffer)
    layer.mxwave_vocab_metadata = tuple(record["metadata"])
    layer.mxwave_vocab_kind = target.kind
    layer.mxwave_vocab_original_parameter = weakref.ref(layer.weight)
    layer.register_parameter("weight", None)
    entry = {**record, "loaded_source_bits_exact": True, "dense_parameter_removed": True}
    layer.mxwave_vocab_qualification = entry
    _ENTRIES.append(entry)
    _OWNERS.append(weakref.ref(layer))


def apply_head(layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Use the pinned split-K=1 fused decoder with no persistent expanded head."""
    from mxwave_zipserv import runtime

    return runtime.linear(x, packed_head(layer), split_k=1)


def report() -> dict[str, Any]:
    """Account only live compressed representations and original-parameter ownership."""
    live = [(entry, owner()) for entry, owner in zip(_ENTRIES, _OWNERS, strict=True)
            if owner() is not None]
    return {"layers": [entry for entry, _ in live], "resident_layers": len(live),
        "raw_bytes": sum(entry["raw_bytes"] for entry, _ in live),
        "retained_bytes": sum(entry["retained_bytes"] for entry, _ in live),
        "dense_parameter_references_alive": sum(
            layer.weight is not None or layer.mxwave_vocab_original_parameter() is not None
            for _, layer in live)}
