"""Bounded loading of packed MXFP4 checkpoints into PyTorch modules."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import torch
from safetensors import safe_open

from .core import BLOCK_SIZE, dequant_mxfp4
from .shard import discover_shards, shard_tensor_keys

__all__ = ["checkpoint_tensor_files", "load_mxfp4_module"]


def checkpoint_tensor_files(model_dir: str | Path) -> dict[str, Path]:
    """Return the validated tensor-to-shard mapping for a safetensors checkpoint."""
    shards, weight_map = discover_shards(model_dir)
    if weight_map is not None:
        paths = {shard.path.name: shard.path for shard in shards}
        missing_files = sorted(set(weight_map.values()).difference(paths))
        if missing_files:
            raise FileNotFoundError(
                f"Checkpoint index references missing shard files: {missing_files[:5]}"
            )
        return {name: paths[filename] for name, filename in weight_map.items()}

    result: dict[str, Path] = {}
    for shard in shards:
        for name in shard_tensor_keys(shard):
            if name in result:
                raise ValueError(f"Checkpoint contains duplicate tensor {name!r}")
            result[name] = shard.path
    return result


def load_mxfp4_module(
    module: torch.nn.Module,
    module_prefix: str,
    checkpoint_files: Mapping[str, Path],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    """Load one module from an ``mxfp4-pack-quantized`` checkpoint."""
    expected = tuple(module.state_dict().items())
    if not expected:
        return module.to(device)

    direct_by_local: dict[str, str] = {}
    packed_by_local: dict[str, tuple[str, str, tuple[int, ...]]] = {}
    keys_by_file: dict[Path, set[str]] = defaultdict(set)
    for local_name, template in expected:
        checkpoint_name = f"{module_prefix}.{local_name}"
        direct_path = checkpoint_files.get(checkpoint_name)
        packed_name = f"{checkpoint_name.removesuffix('.weight')}.weight_packed"
        scale_name = f"{checkpoint_name.removesuffix('.weight')}.weight_scale"
        packed_path = checkpoint_files.get(packed_name) if checkpoint_name.endswith(".weight") else None
        scale_path = checkpoint_files.get(scale_name) if checkpoint_name.endswith(".weight") else None

        if direct_path is not None and (packed_path is not None or scale_path is not None):
            raise ValueError(
                f"MXFP4 checkpoint has both direct and packed forms for {checkpoint_name!r}"
            )
        if direct_path is not None:
            direct_by_local[local_name] = checkpoint_name
            keys_by_file[direct_path].add(checkpoint_name)
            continue
        if (packed_path is None) != (scale_path is None):
            raise ValueError(f"MXFP4 checkpoint has an incomplete packed form for {checkpoint_name!r}")
        if packed_path is None or scale_path is None:
            raise ValueError(f"MXFP4 checkpoint is missing tensor {checkpoint_name!r}")
        shape = tuple(template.shape)
        if len(shape) != 2 or shape[-1] % BLOCK_SIZE != 0:
            raise ValueError(
                f"Packed MXFP4 tensor {checkpoint_name!r} has unsupported logical shape {shape}"
            )
        packed_by_local[local_name] = (packed_name, scale_name, shape)
        keys_by_file[packed_path].add(packed_name)
        keys_by_file[scale_path].add(scale_name)

    raw: dict[str, torch.Tensor] = {}
    for path, names in keys_by_file.items():
        with safe_open(str(path), framework="pt", device=str(device)) as source:
            available = frozenset(source.keys())
            for name in sorted(names):
                if name not in available:
                    raise ValueError(f"Checkpoint shard {path} is missing indexed tensor {name!r}")
                raw[name] = cast(torch.Tensor, source.get_tensor(name))

    state: dict[str, torch.Tensor] = {}
    for local_name, _template in expected:
        direct_name = direct_by_local.get(local_name)
        if direct_name is not None:
            value = raw[direct_name]
            if value.is_floating_point() and value.dtype != dtype:
                value = value.to(dtype=dtype)
            state[local_name] = value
            continue

        packed_name, scale_name, shape = packed_by_local[local_name]
        packed = raw[packed_name]
        scales = raw[scale_name]
        expected_packed_shape = (*shape[:-1], shape[-1] // 2)
        expected_scale_shape = (*shape[:-1], shape[-1] // BLOCK_SIZE)
        if packed.dtype != torch.uint8 or tuple(packed.shape) != expected_packed_shape:
            raise ValueError(
                f"Packed tensor {packed_name!r} must be uint8 with shape "
                f"{expected_packed_shape}, got {packed.dtype} {tuple(packed.shape)}"
            )
        if scales.dtype != torch.uint8 or tuple(scales.shape) != expected_scale_shape:
            raise ValueError(
                f"Scale tensor {scale_name!r} must be uint8 with shape "
                f"{expected_scale_shape}, got {scales.dtype} {tuple(scales.shape)}"
            )
        state[local_name] = dequant_mxfp4(packed, scales, shape).to(dtype=dtype)

    module.load_state_dict(state, strict=True, assign=True)
    for name, value in module.state_dict().items():
        if value.device.type == "meta":
            raise ValueError(f"MXFP4 module retained an unloaded meta tensor: {name}")
    return module
