"""Output-side helpers: quantization config assembly and coverage validation.

Builds a ``compressed-tensors`` ``mxfp4-pack-quantized`` quantization_config
for the emitted checkpoint, and validates that the targets/ignore cover every
real module (no Linear silently left unquantized).
"""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path
from typing import Any

from .shard import ShardFile, shard_tensor_keys
from .verify import verify_config_coverage

__all__ = [
    "assemble_output_dir",
    "build_quantization_config",
    "verify_emitted_config",
]

# A default MXFP4 group for compressed-tensors.
_MXFP4_GROUP: dict[str, Any] = {
    "format": "mxfp4-pack-quantized",
    "input_activations": {
        "actorder": None,
        "block_structure": None,
        "dynamic": True,
        "group_size": 32,
        "num_bits": 4,
        "observer": None,
        "observer_kwargs": {},
        "scale_dtype": "torch.uint8",
        "strategy": "group",
        "symmetric": True,
        "type": "float",
        "zp_dtype": None,
    },
    "output_activations": None,
    "weights": {
        "actorder": None,
        "block_structure": None,
        "dynamic": False,
        "group_size": 32,
        "num_bits": 4,
        "observer": "memoryless_minmax",
        "observer_kwargs": {},
        "scale_dtype": "torch.uint8",
        "strategy": "group",
        "symmetric": True,
        "type": "float",
        "zp_dtype": None,
    },
}


def _module_list_from_keys(keys: list[str]) -> list[str]:
    """Derive concrete module names (e.g. ``layers.0.mlp.gate_proj``) from tensor keys.

    Strips the ``.weight*`` suffix and the leading ``model.`` prefix so the
    result matches the module paths vLLM resolves against.
    """
    modules: list[str] = []
    for key in keys:
        base = re.sub(r"\.weight(_packed|_scale)?$", "", key)
        module = re.sub(r"^model\.", "", base)
        # Skip embeddings / lm_head / norms (not Linear modules).
        if re.search(r"(embed_tokens|lm_head|layernorm|_norm|router)", module):
            continue
        if module not in modules:
            modules.append(module)
    return modules


def _module_to_regex(module: str) -> str:
    """Turn a concrete module path into a layer-index-agnostic ``re:`` target."""
    parts = (r"\d+" if p.isdigit() else re.escape(p) for p in module.split("."))
    return "re:.*" + r"\.".join(parts) + "$"


def build_quantization_config(
    real_modules: list[str],
    *,
    transform_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compressed-tensors quantization_config that covers every Linear.

    Args:
        real_modules: Concrete module names derived from the checkpoint (e.g.
            ``layers.0.mlp.gate_proj``). Each is turned into an ``re:`` target,
            so one config covers every layer.
        transform_config: Optional rotation transform (e.g. ``{"type": "hadamard"}``).

    Returns:
        The ``quantization_config`` payload to merge into ``config.json``.
    """
    targets: list[str] = []
    for module in real_modules:
        target = _module_to_regex(module)
        if target not in targets:
            targets.append(target)

    # Anything not a Linear (embeddings, lm_head, norms, router) is ignored.
    ignore: list[str] = ["lm_head", "embed_tokens"]

    return {
        "config_groups": {
            "group_0": {
                **_MXFP4_GROUP,
                "targets": targets,
            }
        },
        "format": "mxfp4-pack-quantized",
        "global_compression_ratio": None,  # filled in assemble_output_dir
        "ignore": ignore,
        "kv_cache_scheme": None,
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
        "sparsity_config": {},
        "transform_config": transform_config or {},
        "version": "0.18.1",
    }


def _shard_data_size(path: Path) -> int:
    """Return the total tensor-data byte size of a safetensors shard."""
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        f.seek(8 + header_len)  # skip header JSON
        return len(f.read())


def _build_index(
    shards: list[ShardFile],
    output_dir: Path,
) -> dict[str, Any]:
    """Rebuild ``model.safetensors.index.json`` for the emitted output shards.

    Maps each tensor name to its output shard filename and sums the real data
    sizes for ``metadata.total_size``.
    """
    weight_map: dict[str, str] = {}
    total_size = 0
    for shard in shards:
        out_name = shard.path.name
        for key in shard_tensor_keys(shard):
            weight_map[key] = out_name
        total_size += _shard_data_size(output_dir / out_name)
    return {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }


def assemble_output_dir(
    model_dir: str | Path,
    output_dir: str | Path,
    shards: list[ShardFile],
    *,
    transform_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the output directory into a drop-in checkpoint.

    Copies ``config.json``, merges the quantization_config, and writes a rebuilt
    safetensors index. Returns the quantization_config that was written.
    """
    src = Path(model_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Copy config.json from source if present.
    src_cfg = src / "config.json"
    cfg: dict[str, Any] = {}
    if src_cfg.exists():
        cfg = json.loads(src_cfg.read_text())
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    # 2. Derive real modules from the emitted shards (union of all keys).
    all_keys: list[str] = []
    for shard in shards:
        for key in shard_tensor_keys(shard):
            if key not in all_keys:
                all_keys.append(key)
    real_modules = _module_list_from_keys(all_keys)

    qcfg = build_quantization_config(real_modules, transform_config=transform_config)

    # 3. Global compression ratio = source bytes / output bytes.
    src_bytes = sum(p.stat().st_size for p in src.glob("*.safetensors"))
    out_bytes = sum(p.stat().st_size for p in out.glob("*.safetensors"))
    if out_bytes > 0 and src_bytes > 0:
        qcfg["global_compression_ratio"] = round(src_bytes / out_bytes, 4)

    cfg["quantization_config"] = qcfg
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    # 4. Rebuild the index.
    index = _build_index(shards, out)
    (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))

    return qcfg


def verify_emitted_config(
    output_dir: str | Path,
    real_modules: list[str],
) -> list[str]:
    """Verify the emitted config covers all real modules; return uncovered gaps."""
    out = Path(output_dir)
    cfg_path = out / "config.json"
    if not cfg_path.exists():
        return real_modules
    cfg = json.loads(cfg_path.read_text())
    qcfg = cfg.get("quantization_config", {})
    groups = qcfg.get("config_groups", {})
    targets: list[str] = []
    for group in groups.values():
        targets.extend(group.get("targets", []))
    ignore = qcfg.get("ignore", [])
    return verify_config_coverage(targets, ignore, real_modules)
