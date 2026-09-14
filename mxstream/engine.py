"""GPU-streaming quantization engine.

Streams safetensors shards one at a time: load a shard's targeted tensors into
device memory, quantize on-device (MXFP4, optionally activation-aware or
rotation-folded), write the compressed result to the output directory, free
memory, and advance to the next shard. Models larger than any single machine
are handled because only one shard (or a few tensors) is resident at a time.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from safetensors.torch import save_file

from .core import quantize_mxfp4
from .format import InputFormat, detect_input_format
from .rotate import hadamard_matrix
from .shard import ShardFile, discover_shards, read_tensor, shard_tensor_keys

__all__ = [
    "QuantizeConfig",
    "quantize_model",
    "quantize_shard",
]

# Tensors that are never quantized (copied verbatim or excluded).
_DEFAULT_EXCLUDE = re.compile(
    r"(.*self_attn.*|.*lm_head.*|.*embed_tokens.*|.*\.router.*|.*norm.*|.*bias.*)"
)

# MXFP4 is applied to weight tensors; identify by the `.weight` suffix.
_WEIGHT_RE = re.compile(r"\.weight$")


@dataclass
class QuantizeConfig:
    """Configuration for a quantization run."""

    model_dir: str | Path = ""
    output_dir: str | Path = ""
    device: torch.device | str = "cuda"
    scale_percentile: float = 99.5
    gamma: torch.Tensor | None = None  # per-channel activation magnitudes
    hessian: torch.Tensor | None = None
    rotation: str = "none"  # "hadamard" | "none"
    exclude: str = _DEFAULT_EXCLUDE.pattern
    workers: int = 1
    verbose: bool = True


def _should_quantize(name: str, exclude: str) -> bool:
    """True if a tensor name should be MXFP4-quantized."""
    return _WEIGHT_RE.search(name) is not None and re.search(exclude, name) is None


def _rotation_matrix(in_features: int, device: torch.device) -> torch.Tensor:
    """Build a rotation matrix for the given in_features (Hadamard if possible)."""
    # Hadamard requires power-of-two; fall back to a random orthogonal matrix
    # that also has power-of-two-compatible construction via padding is complex,
    # so we use a random orthogonal rotation for non-power-of-two sizes.
    if in_features & (in_features - 1) == 0:
        return hadamard_matrix(in_features, device=device)
    # Random orthogonal via QR — deterministic-free, fine for quantization
    q, _ = torch.linalg.qr(torch.randn(in_features, in_features, device=device))
    return cast(torch.Tensor, q)


def quantize_shard(
    shard: ShardFile,
    cfg: QuantizeConfig,
    *,
    rotation: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Quantize a single shard's targeted tensors, returning {name: tensor}.

    Non-targeted tensors are returned verbatim (to be copied/passthrough).
    """
    from .rotate import fold_rotation

    dev = torch.device(cfg.device)
    result: dict[str, torch.Tensor] = {}
    keys = shard_tensor_keys(shard)

    for name in keys:
        if _should_quantize(name, cfg.exclude):
            w = read_tensor(shard, name, device=dev)
            # Fold rotation into the weight's input dimension (in_features)
            if rotation is not None:
                in_features = w.shape[-1]
                w = fold_rotation(w, rotation[:in_features, :in_features], dim=-1)
            packed, scales = quantize_mxfp4(
                w,
                scale_percentile=cfg.scale_percentile,
                gamma=cfg.gamma,
                hessian=cfg.hessian,
            )
            # Emit compressed-tensors style packed keys
            base = name.removesuffix(".weight")
            result[f"{base}.weight_packed"] = packed.cpu()
            result[f"{base}.weight_scale"] = scales.cpu()
        else:
            # Passthrough: copy verbatim (kept on CPU for the output write)
            w = read_tensor(shard, name, device="cpu")
            result[name] = w

    return result


def _emit_index(weight_map: dict[str, str], output_dir: Path) -> None:
    """Write a model.safetensors.index.json if shards were split out."""
    # For simplicity, when we emit per-shard output we reuse the input map.
    total = 0
    index = {
        "metadata": {"total_size": total},
        "weight_map": weight_map,
    }
    (output_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))


def quantize_model(cfg: QuantizeConfig) -> int:
    """Run the full streaming quantization of a model directory.

    Returns the number of shards processed.
    """
    model_dir = Path(cfg.model_dir)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shards, _ = discover_shards(model_dir)
    dev = torch.device(cfg.device)

    # Build a single shared rotation matrix for all layers (QuaRot-style).
    rotation: torch.Tensor | None = None
    if cfg.rotation == "hadamard":
        # Determine a common hidden size from the first weight tensor.
        probe = None
        for shard in shards:
            for name in shard_tensor_keys(shard):
                if _should_quantize(name, cfg.exclude):
                    probe = read_tensor(shard, name, device="cpu")
                    break
            if probe is not None:
                break
        if probe is not None:
            rotation = _rotation_matrix(probe.shape[-1], dev)

    if cfg.verbose:
        fmt: InputFormat = detect_input_format(model_dir)
        print(f"[mxstream] input format: {fmt.kind}")
        print(f"[mxstream] shards: {len(shards)}, device: {dev}")

    processed = 0
    for shard in shards:
        if cfg.verbose:
            print(f"[mxstream] processing {shard.path.name} ...")
        tensors = quantize_shard(shard, cfg, rotation=rotation)
        # Write one output shard per input shard.
        out_name = shard.path.name
        out_path = output_dir / out_name
        save_file(tensors, str(out_path))
        processed += 1

    return processed
