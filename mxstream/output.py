"""Output-side helpers: quantization config assembly and coverage validation.

Builds a ``compressed-tensors`` ``mxfp4-pack-quantized`` quantization_config
for the emitted checkpoint, and validates that the targets/ignore cover every
real module (no Linear silently left unquantized).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .verify import verify_config_coverage

__all__ = [
    "build_quantization_config",
    "update_config",
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


def build_quantization_config(
    targets: list[str],
    ignore: list[str],
    *,
    transform_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compressed-tensors quantization_config for MXFP4."""
    return {
        "config_groups": {
            "group_0": {
                **_MXFP4_GROUP,
                "targets": targets,
            }
        },
        "format": "mxfp4-pack-quantized",
        "global_compression_ratio": None,
        "ignore": ignore,
        "kv_cache_scheme": None,
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
        "sparsity_config": {},
        "transform_config": transform_config or {},
        "version": "0.18.1",
    }


def update_config(
    output_dir: str | Path,
    quantization_config: dict[str, Any],
) -> None:
    """Merge quantization_config into the output config.json."""
    out = Path(output_dir)
    cfg_path = out / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
    else:
        cfg = {}
    cfg["quantization_config"] = quantization_config
    cfg_path.write_text(json.dumps(cfg, indent=2))


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
