"""Lossless GGUF export support for MxWave MXFP4 checkpoints.

The pure helpers in this module validate and repack MxWave's
``compressed-tensors`` representation.  :func:`install_llama_cpp_exporter`
adds that representation to a pinned llama.cpp converter at runtime, leaving
llama.cpp responsible for architecture metadata, tokenizer conversion, and
ordinary floating-point tensors.  Projector-only and MTP-only exports contain
no MxWave-compressed tensors in the current Qwen3.5 checkpoint contract and
are delegated to llama.cpp with an explicit fail-closed check.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from .core import BLOCK_SIZE

__all__ = [
    "Qwen35LinearAttentionConfig",
    "install_llama_cpp_exporter",
    "repack_mxfp4_blocks",
    "transform_qwen35_mxfp4",
]

_EXPECTED_FORMAT = "mxfp4-pack-quantized"
_EXPECTED_METHOD = "compressed-tensors"
_HOOK_MARKER = "_mxwave_gguf_hook_installed"


def _mxfp4_quantization_config(model: Any) -> Mapping[str, Any] | None:
    """Return an MxWave config from either the active or global model config.

    llama.cpp replaces ``hparams`` with ``vision_config`` for ``--mmproj``, but
    retains the complete Hugging Face configuration as ``global_config``.  Text
    and MTP exports keep the quantization config in ``hparams``.
    """
    candidates = (
        getattr(model, "hparams", {}).get("quantization_config"),
        getattr(model, "global_config", {}).get("quantization_config"),
    )
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        if (
            candidate.get("quant_method") == _EXPECTED_METHOD
            and candidate.get("format") == _EXPECTED_FORMAT
        ):
            return candidate
    return None


def _prepare_float_only_export(
    model: Any,
    original_prepare_tensors: Callable[[Any], None],
    mode: str,
) -> None:
    """Delegate a projector or MTP-only export after excluding MXFP4 payloads."""
    packed = sorted(name for name in model.model_tensors if name.endswith(".weight_packed"))
    if packed:
        raise ValueError(
            f"MxWave {mode} export contains unsupported MXFP4 tensors: {packed[:5]} "
            f"({len(packed)} total)"
        )

    # Upstream llama.cpp does not understand MxWave's compressed-tensors
    # format.  It does understand every BF16/F32 tensor selected by its
    # --mmproj and --mtp filters, so hide only the irrelevant quantization
    # descriptor for the duration of the delegated conversion.
    hparams = model.hparams
    had_quantization = "quantization_config" in hparams
    raw_quantization = hparams.get("quantization_config")
    if had_quantization:
        hparams["quantization_config"] = None
    try:
        original_prepare_tensors(model)
    finally:
        if had_quantization:
            hparams["quantization_config"] = raw_quantization


@dataclass(frozen=True)
class Qwen35LinearAttentionConfig:
    """Dimensions needed for Qwen3.5's grouped-to-tiled V-head reorder."""

    num_key_heads: int
    num_value_heads: int
    key_head_dim: int
    value_head_dim: int

    def __post_init__(self) -> None:
        values = (
            self.num_key_heads,
            self.num_value_heads,
            self.key_head_dim,
            self.value_head_dim,
        )
        if any(value <= 0 for value in values):
            raise ValueError("Qwen3.5 linear-attention dimensions must be positive")
        if self.num_value_heads % self.num_key_heads != 0:
            raise ValueError("num_value_heads must be divisible by num_key_heads")


def _validate_mxfp4_pair(name: str, packed: Tensor, scale: Tensor) -> tuple[int, int]:
    if packed.dtype is not torch.uint8 or scale.dtype is not torch.uint8:
        raise TypeError(f"MXFP4 tensor {name} must use uint8 packed weights and scales")
    if packed.ndim != 2 or scale.ndim != 2:
        raise ValueError(f"MXFP4 tensor {name} must have two-dimensional weights and scales")

    rows, packed_columns = (int(value) for value in packed.shape)
    logical_columns = packed_columns * 2
    if logical_columns % BLOCK_SIZE != 0:
        raise ValueError(
            f"MXFP4 tensor {name} has {logical_columns} columns; "
            f"expected a multiple of {BLOCK_SIZE}"
        )
    blocks = logical_columns // BLOCK_SIZE
    expected_scale_shape = (rows, blocks)
    if tuple(int(value) for value in scale.shape) != expected_scale_shape:
        raise ValueError(
            f"MXFP4 scale for {name} has shape {tuple(scale.shape)}, "
            f"expected {expected_scale_shape}"
        )
    return rows, logical_columns


def repack_mxfp4_blocks(packed: Tensor, scale: Tensor) -> NDArray[np.uint8]:
    """Repack compressed-tensors MXFP4 into ggml ``block_mxfp4`` bytes.

    This operation is lossless: it moves the existing four-bit codes and E8M0
    scale bytes without dequantizing or selecting new scales.
    """
    rows, logical_columns = _validate_mxfp4_pair("weight", packed, scale)
    block_count = logical_columns // BLOCK_SIZE

    source = packed.contiguous().reshape(rows, block_count, BLOCK_SIZE // 2)
    low = source & 0x0F
    high = (source >> 4) & 0x0F
    codes = torch.stack((low, high), dim=-1).reshape(rows, block_count, BLOCK_SIZE)
    destination_codes = codes[:, :, : BLOCK_SIZE // 2] | (codes[:, :, BLOCK_SIZE // 2 :] << 4)
    raw = torch.cat((scale.contiguous().unsqueeze(-1), destination_codes), dim=-1)
    return cast(NDArray[np.uint8], raw.reshape(rows, block_count * 17).cpu().numpy())


def _unpack_nibbles(packed: Tensor) -> Tensor:
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    return torch.stack((low, high), dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)


def _pack_nibbles(codes: Tensor) -> Tensor:
    if codes.shape[-1] % 2 != 0:
        raise ValueError("MXFP4 code count must be even")
    pairs = codes.reshape(*codes.shape[:-1], codes.shape[-1] // 2, 2)
    return (pairs[..., 0] | (pairs[..., 1] << 4)).contiguous()


def _v_head_permutation(
    config: Qwen35LinearAttentionConfig,
    head_dim: int,
) -> Tensor:
    values_per_key = config.num_value_heads // config.num_key_heads
    return (
        torch.arange(config.num_value_heads * head_dim, dtype=torch.long)
        .reshape(config.num_key_heads, values_per_key, head_dim)
        .permute(1, 0, 2)
        .contiguous()
        .reshape(-1)
    )


def _reorder_rows(
    packed: Tensor,
    scale: Tensor,
    config: Qwen35LinearAttentionConfig,
    head_dim: int,
) -> tuple[Tensor, Tensor]:
    permutation = _v_head_permutation(config, head_dim)
    expected_rows = config.num_value_heads * head_dim
    if packed.shape[0] != expected_rows:
        raise ValueError(
            f"Qwen3.5 V projection has {packed.shape[0]} rows, expected {expected_rows}"
        )
    return (
        packed.index_select(0, permutation.to(device=packed.device)),
        scale.index_select(0, permutation.to(device=scale.device)),
    )


def transform_qwen35_mxfp4(
    name: str,
    packed: Tensor,
    scale: Tensor,
    config: Qwen35LinearAttentionConfig,
) -> tuple[Tensor, Tensor]:
    """Apply Qwen3.5's lossless grouped-to-tiled V-head permutation."""
    _, logical_columns = _validate_mxfp4_pair(name, packed, scale)
    relevant_suffixes = (
        ".linear_attn.in_proj_qkv.weight",
        ".linear_attn.in_proj_z.weight",
        ".linear_attn.in_proj_a.weight",
        ".linear_attn.in_proj_b.weight",
        ".linear_attn.out_proj.weight",
    )
    if not name.endswith(relevant_suffixes):
        return packed, scale

    if name.endswith(".linear_attn.in_proj_qkv.weight"):
        q_rows = config.key_head_dim * config.num_key_heads
        k_rows = config.key_head_dim * config.num_key_heads
        v_rows = config.value_head_dim * config.num_value_heads
        expected_rows = q_rows + k_rows + v_rows
        if packed.shape[0] != expected_rows:
            raise ValueError(
                f"Qwen3.5 QKV projection has {packed.shape[0]} rows, expected {expected_rows}"
            )
        v_packed, v_scale = _reorder_rows(
            packed[q_rows + k_rows :],
            scale[q_rows + k_rows :],
            config,
            config.value_head_dim,
        )
        return (
            torch.cat((packed[: q_rows + k_rows], v_packed), dim=0),
            torch.cat((scale[: q_rows + k_rows], v_scale), dim=0),
        )

    if name.endswith(".linear_attn.in_proj_z.weight"):
        return _reorder_rows(packed, scale, config, config.value_head_dim)
    if name.endswith(
        (
            ".linear_attn.in_proj_a.weight",
            ".linear_attn.in_proj_b.weight",
        )
    ):
        return _reorder_rows(packed, scale, config, 1)

    expected_columns = config.num_value_heads * config.value_head_dim
    if logical_columns != expected_columns:
        raise ValueError(
            f"Qwen3.5 output projection has {logical_columns} columns, expected {expected_columns}"
        )
    column_permutation = _v_head_permutation(config, config.value_head_dim)
    grouped_columns = column_permutation.reshape(-1, BLOCK_SIZE)
    group_starts = grouped_columns[:, 0]
    expected_groups = group_starts.unsqueeze(1) + torch.arange(
        BLOCK_SIZE, dtype=column_permutation.dtype
    )
    if not torch.equal(grouped_columns, expected_groups) or not bool(
        torch.all(group_starts % BLOCK_SIZE == 0)
    ):
        raise ValueError("Qwen3.5 V-head permutation is not aligned to MXFP4 blocks")

    group_permutation = (group_starts // BLOCK_SIZE).to(dtype=torch.long)
    expected_block_ids = torch.arange(scale.shape[-1], dtype=torch.long)
    if group_permutation.numel() != scale.shape[-1] or not torch.equal(
        torch.sort(group_permutation).values,
        expected_block_ids,
    ):
        raise ValueError("Qwen3.5 V-head permutation does not cover every MXFP4 block")

    codes = _unpack_nibbles(packed)
    codes = codes.index_select(-1, column_permutation.to(device=packed.device))
    return (
        _pack_nibbles(codes),
        scale.index_select(-1, group_permutation.to(device=scale.device)),
    )


def _normalise_target(model: Any, target: str) -> str:
    item = model.filter_tensors((f"{target}.weight_packed", lambda: torch.empty(0)))
    if item is None:
        raise ValueError(f"MxWave target was filtered unexpectedly: {target}")
    name, _ = item
    return cast(str, name).removesuffix(".weight_packed")


def _validate_quantization_group(group: Mapping[str, Any]) -> list[str]:
    # Early MxWave artifacts declared the format only at the top level. Newer
    # output repeats it on the group, as compressed-tensors now does.
    if group.get("format") not in (None, _EXPECTED_FORMAT):
        raise ValueError(f"MxWave GGUF export requires group format {_EXPECTED_FORMAT!r}")
    weights = group.get("weights")
    if not isinstance(weights, Mapping):
        raise TypeError("MxWave GGUF export requires a weights scheme")
    expected_fields: dict[str, object] = {
        "num_bits": 4,
        "type": "float",
        "symmetric": True,
        "group_size": BLOCK_SIZE,
        "strategy": "group",
        "dynamic": False,
        "scale_dtype": "torch.uint8",
    }
    for key, expected in expected_fields.items():
        if weights.get(key) != expected:
            raise ValueError(f"MxWave GGUF export requires weights.{key}={expected!r}")
    targets = group.get("targets")
    if (
        not isinstance(targets, list)
        or not targets
        or not all(isinstance(target, str) for target in targets)
    ):
        raise TypeError("MxWave GGUF export requires a non-empty string target list")
    return cast(list[str], targets)


def _qwen_config(model: Any) -> Qwen35LinearAttentionConfig:
    return Qwen35LinearAttentionConfig(
        num_key_heads=int(model.hparams["linear_num_key_heads"]),
        num_value_heads=int(model.hparams["linear_num_value_heads"]),
        key_head_dim=int(model.hparams["linear_key_head_dim"]),
        value_head_dim=int(model.hparams["linear_value_head_dim"]),
    )


def _lazy_mxfp4_tensor(
    model: Any,
    name: str,
    packed_loader: Callable[[], Tensor],
    scale_loader: Callable[[], Tensor],
    gguf_module: Any,
    lazy_torch_tensor: Any,
) -> Any:
    packed_meta = packed_loader()
    scale_meta = scale_loader()
    rows, logical_columns = _validate_mxfp4_pair(name, packed_meta, scale_meta)
    byte_shape = (rows, logical_columns // BLOCK_SIZE * 17)
    config = _qwen_config(model)

    def load(
        loaders: tuple[Callable[[], Tensor], Callable[[], Tensor]],
    ) -> NDArray[np.uint8]:
        current_packed = lazy_torch_tensor.to_eager(loaders[0]())
        current_scale = lazy_torch_tensor.to_eager(loaders[1]())
        current_packed, current_scale = transform_qwen35_mxfp4(
            name,
            current_packed,
            current_scale,
            config,
        )
        return repack_mxfp4_blocks(current_packed, current_scale)

    return gguf_module.LazyNumpyTensor(
        meta=gguf_module.LazyNumpyTensor.meta_with_dtype_and_shape(np.uint8, byte_shape),
        args=((packed_loader, scale_loader),),
        func=load,
    )


def _write_compressed_mxfp4(
    model: Any,
    quantization_config: Mapping[str, Any],
    gguf_module: Any,
    lazy_torch_tensor: Any,
    logger: Any,
) -> None:
    groups = quantization_config.get("config_groups")
    if not isinstance(groups, Mapping) or set(groups) != {"group_0"}:
        raise ValueError("MxWave GGUF export requires exactly config_groups.group_0")
    raw_group = groups["group_0"]
    if not isinstance(raw_group, Mapping):
        raise TypeError("MxWave config_groups.group_0 must be an object")
    targets = _validate_quantization_group(raw_group)
    expected = {_normalise_target(model, target) for target in targets}
    if len(expected) != len(targets):
        raise ValueError("MxWave GGUF targets must be unique after normalization")
    actual = {
        name.removesuffix(".weight_packed")
        for name in model.model_tensors
        if name.endswith(".weight_packed")
    }
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            "MxWave target coverage mismatch: "
            f"missing={missing[:5]} ({len(missing)}), "
            f"extra={extra[:5]} ({len(extra)})"
        )

    consumed: list[str] = []
    for base_name in sorted(actual):
        packed_name = f"{base_name}.weight_packed"
        scale_name = f"{base_name}.weight_scale"
        if scale_name not in model.model_tensors:
            raise KeyError(f"Missing MXFP4 scale tensor: {scale_name}")
        weight_name = f"{base_name}.weight"
        mapped_name = model.map_tensor_name(weight_name)
        data = _lazy_mxfp4_tensor(
            model,
            weight_name,
            model.model_tensors[packed_name],
            model.model_tensors[scale_name],
            gguf_module,
            lazy_torch_tensor,
        )
        shape = gguf_module.quant_shape_from_byte_shape(
            data.shape,
            gguf_module.GGMLQuantizationType.MXFP4,
        )
        logger.info(
            "%s: losslessly repacked MxWave MXFP4, shape = {%s}",
            mapped_name,
            ", ".join(str(value) for value in reversed(shape)),
        )
        model.gguf_writer.add_tensor(
            mapped_name,
            data,
            raw_dtype=gguf_module.GGMLQuantizationType.MXFP4,
        )
        model._prec_a4[mapped_name] = False
        consumed.extend((packed_name, scale_name))

    for name in consumed:
        del model.model_tensors[name]
    logger.info("Prepared %d lossless MxWave MXFP4 tensors", len(actual))


def install_llama_cpp_exporter() -> None:
    """Teach the active, pinned llama.cpp converter about MxWave checkpoints."""
    gguf_module = importlib.import_module("gguf")
    base_module = importlib.import_module("conversion.base")
    model_base = base_module.ModelBase
    if bool(getattr(model_base, _HOOK_MARKER, False)):
        return

    original_prepare_tensors = model_base.prepare_tensors

    def prepare_tensors_with_mxwave(model: Any) -> None:
        raw_quantization = _mxfp4_quantization_config(model)
        if raw_quantization is None:
            original_prepare_tensors(model)
            return

        # Both auxiliary modes select only original floating-point tensors:
        # Qwen3.5's vision tower is emitted by the MMPROJ model, while --mtp
        # filters out the quantized target layers and retains the BF16 MTP head
        # plus its shared embedding/output tensors.  Do not require the target
        # coverage contract in either mode, but fail rather than silently lose
        # data if a future checkpoint quantizes an auxiliary tensor.
        if model.model_arch == gguf_module.MODEL_ARCH.MMPROJ:
            _prepare_float_only_export(model, original_prepare_tensors, "projector")
            return
        if bool(getattr(model, "mtp_only", False)):
            _prepare_float_only_export(model, original_prepare_tensors, "MTP-only")
            return

        # llama.cpp's multimodal config loader selects the registered model
        # class from top-level ``architectures`` and then exposes only the
        # nested text config as ``hparams``. The converter's enum is therefore
        # the authoritative runtime architecture identity.
        expected_architecture = gguf_module.MODEL_ARCH.QWEN35
        if model.model_arch != expected_architecture:
            raise ValueError(
                f"Unsupported MxWave GGUF architecture: {model.model_arch!r}; "
                f"expected {expected_architecture!r}"
            )

        _write_compressed_mxfp4(
            model,
            raw_quantization,
            gguf_module,
            base_module.LazyTorchTensor,
            base_module.logger,
        )

        model.hparams["quantization_config"] = None
        try:
            original_prepare_tensors(model)
        finally:
            model.hparams["quantization_config"] = raw_quantization
        model._is_mxfp4 = True

    model_base.prepare_tensors = prepare_tensors_with_mxwave
    setattr(model_base, _HOOK_MARKER, True)
