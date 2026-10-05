"""Assemble verified MXFP4 experts with the official block-FP8 Kolibri backbone."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import time
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path
from typing import BinaryIO

from mxwave.format import detect_input_format, load_config_json
from mxwave.incremental_safetensors import IncrementalSafeTensorsWriter, source_data_start
from mxwave.shard import ShardFile, TensorInfo, discover_shards, shard_tensor_info
from mxwave.verify import verify_config_coverage


@dataclass(frozen=True)
class Payload:
    """One unchanged source payload and its explicit destination name."""

    shard: Path
    info: TensorInfo
    name: str
    category: str


def _inventory(directory: Path) -> dict[str, tuple[Path, TensorInfo]]:
    result = {}
    for shard in discover_shards(directory)[0]:
        for name, info in shard_tensor_info(shard).items():
            if name in result:
                raise ValueError(f"Duplicate tensor {name}")
            result[name] = (shard.path, info)
    return result


def plan(experts: Path, reference: Path) -> tuple[dict, list[Payload]]:
    """Validate declared input formats, exact expert layout, FP8 scales, and coverage."""
    config = load_config_json(experts)
    reference_config = load_config_json(reference)
    if (detect_input_format(experts).kind != "mxfp4"
            or detect_input_format(reference).kind != "fp8_block"):
        raise ValueError("Assembly requires declared MXFP4 and native block-FP8 checkpoints")
    if reference_config["quantization_config"] != {
        "quant_method": "fp8", "activation_scheme": "dynamic", "weight_block_size": [128, 128],
        "modules_to_not_convert": reference_config["quantization_config"].get(
            "modules_to_not_convert", []),
    }:
        raise ValueError("Unsupported native FP8 configuration")
    for key in ("model_type", "architectures", "hidden_size", "moe_intermediate_size",
                "num_hidden_layers", "num_experts", "num_experts_per_tok", "layer_types",
                "vocab_size", "num_attention_heads", "num_key_value_heads", "head_dim"):
        if config.get(key) != reference_config.get(key):
            raise ValueError(f"Checkpoint architecture mismatch: {key}")
    if config.get("model_type") != "kolibri1":
        raise ValueError("This assembler only supports Kolibri1")
    layers, count = config["num_hidden_layers"], config["num_experts"]
    hidden, intermediate = config["hidden_size"], config["moe_intermediate_size"]
    original, gold = _inventory(experts), _inventory(reference)
    group = config["quantization_config"]["config_groups"]
    if len(group) != 1:
        raise ValueError("Expected a single weight-only MXFP4 expert scheme")
    mx_group = copy.deepcopy(next(iter(group.values())))
    if (mx_group.get("format") != "mxfp4-pack-quantized"
            or mx_group.get("input_activations") is not None
            or mx_group["weights"].get("group_size") != 32
            or mx_group["weights"].get("num_bits") != 4):
        raise ValueError("Expert scheme must be weight-only MXFP4 with block size 32")
    expected = {}
    for layer in range(layers):
        for expert in range(count):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                rows, columns = ((hidden, intermediate) if projection == "down_proj"
                                 else (intermediate, hidden))
                module = f"model.layers.{layer}.mlp.experts.{expert}.{projection}"
                expected[f"{module}.weight_packed"] = (rows, columns // 2)
                expected[f"{module}.weight_scale"] = (rows, columns // 32)
    actual = {name for name in original if ".mlp.experts." in name}
    if actual != set(expected):
        raise ValueError("Incomplete or foreign expert payload layout")
    payloads = []
    for name, shape in expected.items():
        path, info = original[name]
        if info.dtype != "U8" or info.shape != shape:
            raise ValueError(f"Invalid packed expert shape/dtype: {name}")
        payloads.append(Payload(path, info, name, "mxfp4_expert"))
    backbone_modules = {
        f"model.layers.{layer}.{kind}.{projection}"
        for layer in range(layers)
        for kind, projections in (
            ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
            ("mlp.shared_experts", ("gate_proj", "up_proj", "down_proj")),
        )
        for projection in projections
    }
    wanted_reference = set()
    for name, (_old_path, old_info) in original.items():
        if name in expected:
            continue
        if name not in gold:
            raise ValueError(f"Official reference is missing {name}")
        path, info = gold[name]
        if old_info.shape != info.shape:
            raise ValueError(f"Backbone shape mismatch: {name}")
        module = name.removesuffix(".weight")
        if module in backbone_modules:
            if info.dtype != "F8_E4M3" or len(info.shape) != 2:
                raise ValueError(f"Expected block-FP8 projection: {name}")
            scale_name = f"{module}.weight_scale_inv"
            scale_path, scale = gold[scale_name]
            if (scale.dtype != "F32"
                    or scale.shape != tuple((dim + 127) // 128 for dim in info.shape)):
                raise ValueError(f"FP8 scale shape/dtype mismatch: {scale_name}")
            # Native FP8 and compressed-tensors both store float32 dequantization
            # multipliers here. Only the schema's parameter name changes.
            payloads.append(Payload(scale_path, scale, f"{module}.weight_scale", "fp8_scale"))
            wanted_reference.add(scale_name)
        elif info.dtype != old_info.dtype:
            raise ValueError(f"Unexpected nonprojection precision change: {name}")
        payloads.append(Payload(path, info, name, "fp8_backbone"))
        wanted_reference.add(name)
    if wanted_reference != {name for name in gold if ".mlp.experts." not in name}:
        raise ValueError("Official reference contains unexpected or missing backbone tensors")
    found_backbone = {p.name.removesuffix(".weight") for p in payloads
                      if p.category == "fp8_backbone" and p.info.dtype == "F8_E4M3"}
    if found_backbone != backbone_modules:
        raise ValueError("Incomplete FP8 attention/shared-expert coverage")
    fp8_group = {
        "format": "float-quantized", "targets": sorted(backbone_modules),
        "weights": {"num_bits": 8, "type": "float", "symmetric": True,
                    "dynamic": False, "strategy": "block", "block_structure": [128, 128]},
        "input_activations": {"num_bits": 8, "type": "float", "symmetric": True,
                              "dynamic": True, "strategy": "token"},
    }
    qconfig = config["quantization_config"]
    qconfig["format"] = "mixed-precision"
    qconfig["config_groups"] = {"mxfp4_experts": mx_group, "fp8_backbone": fp8_group}
    ignored = [name for name in qconfig["ignore"] if name not in backbone_modules]
    targets = mx_group["targets"] + fp8_group["targets"]
    modules = sorted({p.name.removesuffix(".weight") for p in payloads
                      if p.name.endswith(".weight")} | {
                          name.removesuffix(".weight_packed") for name in expected
                          if name.endswith(".weight_packed")})
    if verify_config_coverage(targets, ignored, modules):
        raise ValueError("Mixed quantization config leaves uncovered modules")
    qconfig["ignore"] = ignored
    qconfig.pop("global_compression_ratio", None)
    return config, payloads


def _payload_hash(stream: BinaryIO, start: int, info: TensorInfo) -> str:
    stream.seek(start + info.data_offsets[0])
    remaining = info.data_offsets[1] - info.data_offsets[0]
    digest = hashlib.sha256()
    while remaining:
        value = stream.read(min(remaining, 8 * 1024**2))
        if not value:
            raise ValueError(f"Truncated payload: {info.name}")
        digest.update(value)
        remaining -= len(value)
    return digest.hexdigest()


def assemble(experts: Path, reference: Path, output: Path) -> dict:
    """Copy in bounded chunks and independently verify every output payload's SHA256."""
    started = time.monotonic()
    config, payloads = plan(experts, reference)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    total_bytes = sum(p.info.data_offsets[1] - p.info.data_offsets[0] for p in payloads)
    if shutil.disk_usage(output.parent).free < total_bytes + 3 * 1024**3:
        raise OSError("Insufficient space to assemble with a 3 GiB reserve")
    output.mkdir()
    grouped = defaultdict(list)
    for payload in payloads:
        # Preserve the original expert shard distribution; backbone tensors may
        # reside in different reference shards, so use their original MXFP4 slot.
        original_name = payload.info.name.replace(".weight_scale_inv", ".weight")
        slot = (payload.shard.name if payload.category == "mxfp4_expert"
                else _inventory_cached_slot(experts, original_name))
        grouped[slot].append(payload)
    weight_map, per_shard, hashes = {}, {}, {}
    for number, (filename, items) in enumerate(sorted(grouped.items()), 1):
        destination = output / filename
        with ExitStack() as stack:
            streams = {path: stack.enter_context(path.open("rb"))
                       for path in {p.shard for p in items}}
            starts = {path: source_data_start(stream, path) for path, stream in streams.items()}
            stream = stack.enter_context(destination.open("w+b"))
            writer = IncrementalSafeTensorsWriter(
                stream, {p.name: (p.info.shape, p.info.dtype) for p in items},
            )
            for payload in items:
                source = streams[payload.shard]
                hashes[payload.name] = _payload_hash(source, starts[payload.shard], payload.info)
                writer.copy_tensor_payload(
                    payload.name, source, payload.shard, replace(payload.info, name=payload.name),
                    source_payload_start=starts[payload.shard],
                )
                weight_map[payload.name] = filename
            writer.finish()
        infos = shard_tensor_info(ShardFile(destination, {}))
        with destination.open("rb") as stream:
            start = source_data_start(stream, destination)
            for name, info in infos.items():
                if _payload_hash(stream, start, info) != hashes[name]:
                    raise ValueError(f"Output payload verification failed: {name}")
            stream.seek(0)
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        per_shard[filename] = {"bytes": destination.stat().st_size, "sha256": digest}
        print(f"Assembled and verified shard {number}/{len(grouped)}: {filename}", flush=True)
    for path in experts.iterdir():
        if (path.is_file() and path.suffix != ".safetensors" and path.name not in
                {"config.json", "model.safetensors.index.json", "mxwave-manifest.json", "README.md"}
                and not path.name.startswith(("mxwave-", "."))):
            shutil.copyfile(path, output / path.name)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": total_bytes}, "weight_map": dict(sorted(weight_map.items())),
    }, indent=2) + "\n")
    manifest = {
        "format": "mixed-precision", "expert_format": "mxfp4-pack-quantized",
        "backbone_format": "float-quantized block FP8", "fp8_weight_block_size": [128, 128],
        "source_repository": "Aleph-Alpha/Kolibri-1-BF16",
        "source_revision": "7a8f290e7858825c3cf5e4c447ba68345de9f1d3",
        "backbone_repository": "Aleph-Alpha/Kolibri-1",
        "backbone_revision": "e52eb4627d11516b0c01de49210ab5a4e4061444",
        "expert_manifest_sha256": hashlib.sha256(
            (experts / "mxwave-manifest.json").read_bytes()).hexdigest()
        if (experts / "mxwave-manifest.json").is_file() else None,
        "assembly_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "activation_calibration": "none", "backbone_activation_quantization": "dynamic FP8",
        "expert_weights": "unchanged previously quantized BF16-master MXFP4 MSE clip-depth 4",
        "payload_verification": "all source/output SHA256 values match, including renamed scales",
        "payload_counts": dict(Counter(p.category for p in payloads)),
        "payload_sha256": hashes, "config_coverage_gaps": [],
        "shard_integrity": {"per_shard": per_shard},
        "quality_qualification": "pending",
    }
    (output / "mxwave-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    result = {
        "status": "complete", "seconds": time.monotonic() - started,
        "payload_bytes": total_bytes, "weight_files_bytes": sum(p["bytes"] for p in per_shard.values()),
        "payload_counts": manifest["payload_counts"], "shards": len(per_shard),
        "manifest_sha256": hashlib.sha256((output / "mxwave-manifest.json").read_bytes()).hexdigest(),
        "quality_qualification": "pending",
    }
    (output.parent / f"{output.name}-assembly.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


_SLOTS: dict[Path, dict[str, str]] = {}


def _inventory_cached_slot(directory: Path, name: str) -> str:
    if directory not in _SLOTS:
        _SLOTS[directory] = {key: path.name for key, (path, _info) in _inventory(directory).items()}
    return _SLOTS[directory][name]


def main() -> None:
    """Assemble a new candidate without modifying either verified input checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experts", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(assemble(args.experts, args.reference, args.output)), flush=True)


if __name__ == "__main__":
    main()
