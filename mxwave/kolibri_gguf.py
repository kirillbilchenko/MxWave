"""Stream Kolibri's mixed MXFP4/FP8 checkpoint into a native MXFP4 GGUF.

Expert codes and E8M0 scales are repacked without requantization. Block-FP8
attention and shared experts are dequantized to BF16. Routers and norms are
stored in F32. The external, licensed Kolibri llama.cpp port supplies metadata
and tokenizer conversion; stock llama.cpp does not yet support this architecture.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import os
import shutil
import time
from collections.abc import Iterator, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Literal

import numpy as np
import torch

from .checkpoint import _atomic_write_text, locked_output_directory
from .gguf import _validate_quantization_group, repack_mxfp4_blocks
from .incremental_safetensors import source_data_start
from .shard import TensorInfo, discover_shards, shard_tensor_info
from .verify import verify_config_coverage

Encoding = Literal["mxfp4", "fp8-bf16", "bf16", "f32"]
_CHUNK_BYTES = 8 * 1024**2


@dataclass(frozen=True)
class KolibriTensorSource:
    """One raw source payload, located without reading tensor data."""

    path: Path
    info: TensorInfo
    offset: int


@dataclass(frozen=True)
class KolibriGGUFTensor:
    """One GGUF tensor and its sources in numeric expert order."""

    name: str
    shape: tuple[int, ...]
    encoding: Encoding
    parts: tuple[tuple[KolibriTensorSource, ...], ...]

    @property
    def nbytes(self) -> int:
        """Return the exact unpadded payload size."""
        elements = math.prod(self.shape)
        if self.encoding == "mxfp4":
            return elements // 32 * 17
        return elements * (4 if self.encoding == "f32" else 2)


@dataclass(frozen=True)
class KolibriGGUFPlan:
    """A complete, header-validated conversion plan."""

    config: dict[str, Any]
    tensors: tuple[KolibriGGUFTensor, ...]
    source_payloads: int

    @property
    def nbytes(self) -> int:
        """Return the total unpadded output payload size."""
        return sum(tensor.nbytes for tensor in self.tensors)


def _quantization_groups(config: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    quant = config.get("quantization_config", {})
    if (
        quant.get("quant_method") != "compressed-tensors"
        or quant.get("format") != "mixed-precision"
    ):
        raise ValueError("Kolibri GGUF requires the declared mixed compressed-tensors format")
    groups = quant.get("config_groups", {})
    mx = [group for group in groups.values() if group.get("format") == "mxfp4-pack-quantized"]
    fp = [group for group in groups.values() if group.get("format") == "float-quantized"]
    if len(groups) != 2 or len(mx) != 1 or len(fp) != 1:
        raise ValueError("Expected one MXFP4 expert group and one block-FP8 backbone group")
    _validate_quantization_group(mx[0])
    if mx[0].get("input_activations") is not None:
        raise ValueError("MXFP4 experts must use unquantized activations")
    weights = fp[0].get("weights", {})
    expected = {
        "num_bits": 8,
        "type": "float",
        "symmetric": True,
        "dynamic": False,
        "strategy": "block",
        "block_structure": [128, 128],
    }
    if any(weights.get(key) != value for key, value in expected.items()):
        raise ValueError("Backbone must declare symmetric static 128x128 block-FP8 weights")
    return mx[0], fp[0]


def plan_kolibri_gguf(model_dir: Path) -> KolibriGGUFPlan:
    """Validate every source payload and construct all 903 full-model GGUF tensors."""
    config = json.loads((model_dir / "config.json").read_text())
    if (
        config.get("architectures") != ["Kolibri1ForCausalLM"]
        or config.get("model_type") != "kolibri1"
    ):
        raise ValueError("Expected a Kolibri1 checkpoint")
    mx, fp = _quantization_groups(config)
    layers, experts = int(config["num_hidden_layers"]), int(config["num_experts"])
    hidden, ff = int(config["hidden_size"]), int(config["moe_intermediate_size"])
    shared, vocab = int(config["shared_expert_intermediate_size"]), int(config["vocab_size"])
    head_dim = int(config["head_dim"])
    q, kv = (
        int(config["num_attention_heads"]) * head_dim,
        int(config["num_key_value_heads"]) * head_dim,
    )
    if min(layers, experts, hidden, ff, shared, vocab, head_dim, q, kv) <= 0:
        raise ValueError("Kolibri dimensions must be positive")
    if hidden % 32 or ff % 32:
        raise ValueError("Kolibri expert input dimensions must be multiples of 32")
    if (
        config.get("attention_bias")
        or config.get("tie_word_embeddings")
        or config.get("hidden_act") != "silu"
    ):
        raise ValueError("Unsupported Kolibri attention, embedding, or activation configuration")
    layer_types = config.get("layer_types", [])
    if len(layer_types) != layers or any(
        kind not in ("full_attention", "sliding_attention") for kind in layer_types
    ):
        raise ValueError("Expected an explicit full/sliding attention type for every layer")
    inventory: dict[str, KolibriTensorSource] = {}
    shards, index = discover_shards(model_dir)
    for shard in shards:
        if shard.path.parent != model_dir or shard.path.name in (".", ".."):
            raise ValueError("Shard index must reference files inside the checkpoint")
        with shard.path.open("rb") as stream:
            start = source_data_start(stream, shard.path)
        for name, info in shard_tensor_info(shard).items():
            if name in inventory or (index is not None and index.get(name) != shard.path.name):
                raise ValueError(f"Duplicate or incorrectly indexed source: {name}")
            width = {"U8": 1, "F8_E4M3": 1, "BF16": 2, "F32": 4}.get(info.dtype)
            if (
                width is None
                or info.data_offsets[1] - info.data_offsets[0] != math.prod(info.shape) * width
            ):
                raise ValueError(f"Unsupported dtype or malformed source size: {name}")
            inventory[name] = KolibriTensorSource(shard.path, info, start + info.data_offsets[0])
    if index is not None and set(index) != set(inventory):
        raise ValueError("Shard index and tensor headers disagree")
    used: set[str] = set()
    mx_modules: list[str] = []
    fp_modules: list[str] = []
    ordinary_modules: list[str] = []
    plan: list[KolibriGGUFTensor] = []

    def source(name: str, shape: tuple[int, ...], dtypes: tuple[str, ...]) -> KolibriTensorSource:
        if name in used:
            raise ValueError(f"Source would be consumed twice: {name}")
        if name not in inventory:
            raise ValueError(f"Missing source payload: {name}")
        item = inventory[name]
        if item.info.shape != shape or item.info.dtype not in dtypes:
            raise ValueError(f"Unexpected source shape/dtype: {name}")
        used.add(name)
        return item

    def ordinary(name: str, target: str, shape: tuple[int, ...], *, fp8: bool = False) -> None:
        item = source(name, shape, ("F8_E4M3",) if fp8 else ("BF16", "F32"))
        parts: tuple[tuple[KolibriTensorSource, ...], ...]
        if fp8:
            scale = source(
                name.removesuffix(".weight") + ".weight_scale",
                tuple((dim + 127) // 128 for dim in shape),
                ("F32",),
            )
            parts = ((item, scale),)
            encoding: Encoding = "fp8-bf16"
            fp_modules.append(name.removesuffix(".weight"))
        else:
            parts = ((item,),)
            encoding = (
                "f32"
                if len(shape) == 1 or ".mlp.gate." in name or item.info.dtype == "F32"
                else "bf16"
            )
            if name.endswith(".weight"):
                ordinary_modules.append(name.removesuffix(".weight"))
        plan.append(KolibriGGUFTensor(target, shape, encoding, parts))

    ordinary("model.embed_tokens.weight", "token_embd.weight", (vocab, hidden))
    ordinary("model.norm.weight", "output_norm.weight", (hidden,))
    ordinary("lm_head.weight", "output.weight", (vocab, hidden))
    # Names and dimensions follow the public Kolibri HF and GGUF contracts.
    layer_fields = (
        ("self_attn.q_proj.weight", "attn_q.weight", (q, hidden), True),
        ("self_attn.k_proj.weight", "attn_k.weight", (kv, hidden), True),
        ("self_attn.v_proj.weight", "attn_v.weight", (kv, hidden), True),
        ("self_attn.o_proj.weight", "attn_output.weight", (hidden, q), True),
        ("self_attn.q_norm.weight", "attn_q_norm.weight", (head_dim,), False),
        ("self_attn.k_norm.weight", "attn_k_norm.weight", (head_dim,), False),
        ("input_layernorm.weight", "attn_norm.weight", (hidden,), False),
        ("post_attn_norm.weight", "post_attention_norm.weight", (hidden,), False),
        ("post_attention_layernorm.weight", "ffn_norm.weight", (hidden,), False),
        ("post_ffn_norm.weight", "post_ffw_norm.weight", (hidden,), False),
        ("mlp.gate.weight", "ffn_gate_inp.weight", (experts, hidden), False),
        ("moe.router.expert_bias", "exp_probs_b.bias", (experts,), False),
        ("mlp.shared_experts.gate_proj.weight", "ffn_gate_shexp.weight", (shared, hidden), True),
        ("mlp.shared_experts.up_proj.weight", "ffn_up_shexp.weight", (shared, hidden), True),
        ("mlp.shared_experts.down_proj.weight", "ffn_down_shexp.weight", (hidden, shared), True),
    )
    for layer in range(layers):
        prefix = f"model.layers.{layer}."
        for suffix, target, shape, is_fp8 in layer_fields:
            ordinary(prefix + suffix, f"blk.{layer}.{target}", shape, fp8=is_fp8)
        for projection, target in (
            ("gate_proj", "ffn_gate_exps"),
            ("up_proj", "ffn_up_exps"),
            ("down_proj", "ffn_down_exps"),
        ):
            rows, columns = (hidden, ff) if projection == "down_proj" else (ff, hidden)
            parts = []
            for expert in range(experts):
                module = prefix + f"mlp.experts.{expert}.{projection}"
                packed = source(module + ".weight_packed", (rows, columns // 2), ("U8",))
                scale = source(module + ".weight_scale", (rows, columns // 32), ("U8",))
                mx_modules.append(module)
                parts.append((packed, scale))
            plan.append(
                KolibriGGUFTensor(
                    f"blk.{layer}.{target}.weight", (experts, rows, columns), "mxfp4", tuple(parts)
                )
            )
    if used != set(inventory):
        raise ValueError(f"Unmapped source payloads: {sorted(set(inventory) - used)[:5]}")
    if verify_config_coverage(mx["targets"], [], mx_modules):
        raise ValueError("Declared MXFP4 targets do not cover every routed expert")
    if set(fp["targets"]) != set(fp_modules):
        raise ValueError("Declared FP8 targets do not match attention and shared experts")
    ignore = config["quantization_config"].get("ignore", [])
    if verify_config_coverage([], ignore, ordinary_modules):
        raise ValueError("Unquantized module is missing from the explicit ignore list")
    return KolibriGGUFPlan(config, tuple(plan), len(inventory))


def _read_payload(
    source: KolibriTensorSource, handles: Mapping[Path, BinaryIO], expected: Mapping[str, str]
) -> bytes:
    stream = handles[source.path]
    stream.seek(source.offset)
    size = source.info.data_offsets[1] - source.info.data_offsets[0]
    raw = stream.read(size)
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != expected[source.info.name]:
        raise ValueError(f"Source payload differs from its manifest: {source.info.name}")
    return raw


def _chunks(
    tensor: KolibriGGUFTensor, handles: Mapping[Path, BinaryIO], expected: Mapping[str, str]
) -> Iterator[bytes]:
    for part in tensor.parts:
        source = part[0]
        if tensor.encoding == "mxfp4":
            packed = torch.frombuffer(
                bytearray(_read_payload(source, handles, expected)), dtype=torch.uint8
            ).reshape(source.info.shape)
            scale_source = part[1]
            scales = torch.frombuffer(
                bytearray(_read_payload(scale_source, handles, expected)), dtype=torch.uint8
            ).reshape(scale_source.info.shape)
            yield repack_mxfp4_blocks(packed, scales).tobytes()
        elif tensor.encoding == "fp8-bf16":
            scale_source = part[1]
            scales = torch.frombuffer(
                bytearray(_read_payload(scale_source, handles, expected)), dtype=torch.float32
            ).reshape(scale_source.info.shape)
            if not torch.isfinite(scales).all() or not torch.all(scales > 0):
                raise ValueError(
                    f"Invalid block-FP8 dequantization multipliers: {scale_source.info.name}"
                )
            rows, columns = source.info.shape
            stream = handles[source.path]
            stream.seek(source.offset)
            digest = hashlib.sha256()
            for start in range(0, rows, 256):
                stop = min(start + 256, rows)
                raw = stream.read((stop - start) * columns)
                if len(raw) != (stop - start) * columns:
                    raise ValueError(f"Truncated FP8 source: {source.info.name}")
                digest.update(raw)
                weight = torch.frombuffer(bytearray(raw), dtype=torch.float8_e4m3fn).float()
                weight = weight.reshape(stop - start, columns)
                multipliers = scales[start // 128 : (stop + 127) // 128]
                multipliers = multipliers.repeat_interleave(128, 0).repeat_interleave(128, 1)
                restored = weight * multipliers[: stop - start, :columns]
                if not torch.isfinite(restored).all():
                    raise ValueError(f"Nonfinite dequantized FP8 tensor: {source.info.name}")
                yield restored.to(torch.bfloat16).view(torch.uint16).numpy().tobytes()
            if digest.hexdigest() != expected[source.info.name]:
                raise ValueError(f"FP8 source differs from its manifest: {source.info.name}")
        else:
            stream = handles[source.path]
            stream.seek(source.offset)
            remaining = source.info.data_offsets[1] - source.info.data_offsets[0]
            digest = hashlib.sha256()
            while remaining:
                raw = stream.read(min(remaining, _CHUNK_BYTES))
                if not raw:
                    raise ValueError(f"Truncated source: {source.info.name}")
                digest.update(raw)
                remaining -= len(raw)
                if tensor.encoding == "f32" and source.info.dtype == "BF16":
                    values = np.frombuffer(raw, dtype="<u2").astype("<u4") << 16
                    yield values.tobytes()
                else:
                    yield raw
            if digest.hexdigest() != expected[source.info.name]:
                raise ValueError(f"Source differs from its manifest: {source.info.name}")


def _writer(model_dir: Path, output: Path, plan: KolibriGGUFPlan, gguf: Any) -> Any:
    try:
        cls = importlib.import_module("conversion.kolibri1").Kolibri1Model
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            "Use the pinned, patched Kolibri llama.cpp converter on PYTHONPATH"
        ) from exc
    model = cls(
        model_dir, gguf.LlamaFileType.MOSTLY_MXFP4_MOE, output, hparams=copy.deepcopy(plan.config)
    )
    model.set_gguf_parameters()
    model.set_vocab()
    writer = model.gguf_writer
    writer.add_name("Kolibri-1 MxWave softer RMS MXFP4")
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_MXFP4_MOE)
    # Kolibri uses NeoX RoPE on sliding layers only. There is no Q/K row permutation.
    return writer


def _gguf_encoding(tensor: KolibriGGUFTensor, gguf: Any) -> tuple[Any, tuple[int, ...], Any]:
    if tensor.encoding == "mxfp4":
        byte_shape = (*tensor.shape[:-1], tensor.shape[-1] // 32 * 17)
        return gguf.GGMLQuantizationType.MXFP4, byte_shape, np.dtype("uint8")
    if tensor.encoding == "f32":
        return gguf.GGMLQuantizationType.F32, tensor.shape, np.dtype("float32")
    return gguf.GGMLQuantizationType.BF16, tensor.shape, np.dtype("uint16")


def _write_tensors(
    plan: KolibriGGUFPlan, writer: Any, output: Path, gguf: Any, expected: Mapping[str, str]
) -> dict[str, str]:
    for tensor in plan.tensors:
        dtype, shape, numpy_dtype = _gguf_encoding(tensor, gguf)
        writer.add_tensor_info(tensor.name, shape, numpy_dtype, tensor.nbytes, raw_dtype=dtype)
    writer.write_header_to_file(path=output)
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    if len(writer.fout) != 1:
        raise ValueError("Kolibri streaming export expects one GGUF output")
    output_stream = writer.fout[0]
    paths = {source.path for tensor in plan.tensors for part in tensor.parts for source in part}
    hashes: dict[str, str] = {}
    with ExitStack() as stack:
        handles = {path: stack.enter_context(path.open("rb")) for path in paths}
        completed = 0
        for index, tensor in enumerate(plan.tensors):
            writer.write_padding(output_stream, output_stream.tell())
            digest = hashlib.sha256()
            byte_count = 0
            for raw in _chunks(tensor, handles, expected):
                output_stream.write(raw)
                digest.update(raw)
                byte_count += len(raw)
            if byte_count != tensor.nbytes:
                raise ValueError(f"Incorrect emitted payload size: {tensor.name}")
            hashes[tensor.name] = digest.hexdigest()
            writer.write_padding(output_stream, byte_count)
            completed += byte_count
            if index % 18 == 0 or index + 1 == len(plan.tensors):
                print(
                    json.dumps(
                        {
                            "status": "converting",
                            "tensor": tensor.name,
                            "completed_bytes": completed,
                            "total_bytes": plan.nbytes,
                        }
                    ),
                    flush=True,
                )
        output_stream.flush()
        os.fsync(output_stream.fileno())
    return hashes


def _verify_output(path: Path, plan: KolibriGGUFPlan, hashes: Mapping[str, str], gguf: Any) -> None:
    reader = gguf.GGUFReader(path)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    if set(tensors) != set(hashes):
        raise ValueError("Emitted GGUF tensor inventory differs from the conversion plan")
    for item in plan.tensors:
        tensor = tensors[item.name]
        dtype = _gguf_encoding(item, gguf)[0]
        if (
            tensor.tensor_type != dtype
            or tuple(int(dim) for dim in tensor.shape) != item.shape[::-1]
            or tensor.n_bytes != item.nbytes
        ):
            raise ValueError(f"Emitted GGUF metadata differs: {item.name}")
        raw = tensor.data.view(np.uint8).reshape(-1)
        digest = hashlib.sha256()
        for start in range(0, raw.size, _CHUNK_BYTES):
            digest.update(raw[start : start + _CHUNK_BYTES].tobytes())
        if digest.hexdigest() != hashes[item.name]:
            raise ValueError(f"Independent GGUF payload verification failed: {item.name}")


def export_kolibri_gguf(model_dir: Path, output: Path, *, dry_run: bool = False) -> dict[str, Any]:
    """Stream a manifest-bound checkpoint, then independently verify every GGUF tensor."""
    started = time.monotonic()
    model_dir = model_dir.resolve()
    plan = plan_kolibri_gguf(model_dir)
    report: dict[str, Any] = {
        "status": "planned",
        "source_payloads": plan.source_payloads,
        "gguf_tensors": len(plan.tensors),
        "payload_bytes": plan.nbytes,
        "expert_conversion": "lossless MXFP4 repack; original E8M0 scales retained",
        "backbone_conversion": "FP8 dequantized to BF16; F32 routers and norms",
        "quality_qualification": "pending; vLLM metrics do not qualify a different backend",
    }
    if dry_run:
        return report
    gguf = importlib.import_module("gguf")
    manifest_path = model_dir / "mxwave-manifest.json"
    manifest_raw = manifest_path.read_bytes()
    expected = json.loads(manifest_raw)["payload_sha256"]
    names = {
        source.info.name for tensor in plan.tensors for part in tensor.parts for source in part
    }
    if set(expected) != names:
        raise ValueError("Source manifest payload inventory differs from the complete checkpoint")
    output = output.resolve()
    partial = output.with_suffix(output.suffix + ".partial")
    with locked_output_directory(output.parent, source_dir=model_dir):
        if output.exists() or partial.exists():
            raise FileExistsError("GGUF output or incomplete output already exists")
        if shutil.disk_usage(output.parent).free < plan.nbytes + 3 * 1024**3:
            raise OSError("GGUF export needs its planned size plus a 3 GiB reserve")
        writer = _writer(model_dir, partial, plan, gguf)
        writer.add_string("mxwave.source_manifest_sha256", hashlib.sha256(manifest_raw).hexdigest())
        writer.add_string("mxwave.conversion", "lossless expert MXFP4 repack; FP8 backbone to BF16")
        try:
            hashes = _write_tensors(plan, writer, partial, gguf, expected)
        finally:
            writer.close()
        _verify_output(partial, plan, hashes, gguf)
        with partial.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        partial.rename(output)
        report.update(
            status="complete",
            output=output.name,
            output_bytes=output.stat().st_size,
            output_sha256=digest,
            source_manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
            source_config_sha256=hashlib.sha256(
                (model_dir / "config.json").read_bytes()
            ).hexdigest(),
            verified_source_payloads=len(names),
            verified_gguf_tensors=len(hashes),
            gguf_payload_sha256=hashes,
            seconds=time.monotonic() - started,
        )
        _atomic_write_text(output.with_suffix(".json"), json.dumps(report, indent=2) + "\n")
    return report
