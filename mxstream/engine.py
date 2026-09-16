"""GPU-streaming, architecture-aware MXFP4 checkpoint conversion."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from . import __version__
from .calibration import CalibrationData, CalibrationObjective, load_calibration_data
from .core import QuantizationMethod, dequant_mxfp4, quantize_mxfp4
from .format import InputFormat, detect_input_format
from .output import assemble_output_dir, verify_emitted_config
from .policy import (
    PolicyName,
    QuantizationPolicy,
    expected_target_count,
    resolve_policy,
    validate_selected_tensor,
)
from .shard import (
    ShardFile,
    TensorInfo,
    discover_shards,
    read_tensor,
    read_tensor_row_range,
    read_tensor_rows,
    shard_tensor_info,
)
from .verify import block_hessian_weighted_sqnr, channel_weighted_sqnr, sqnr

__all__ = [
    "ModelPlan",
    "PlannedTensor",
    "QuantizeConfig",
    "plan_model",
    "quantize_model",
    "quantize_shard",
]


@dataclass
class QuantizeConfig:
    """Configuration for a streaming quantization run."""

    model_dir: str | Path = ""
    output_dir: str | Path = ""
    device: torch.device | str = "cuda"
    policy: PolicyName = "auto"
    method: QuantizationMethod = "mse"
    scale_percentile: float = 99.5
    mse_clip_depth: int = 1
    hessian_rounding_sweeps: int = 0
    hessian_error_feedback: bool = False
    hessian_feedback_damp_percent: float = 1.0
    hessian_feedback_activation_order: bool = True
    hessian_feedback_max_mse_ratio: float | None = None
    feedback_selection_stats: str | Path | None = None
    tensor_row_chunk_size: int = 1024
    activation_stats: str | Path | None = None
    calibration_objective: CalibrationObjective = "mean-abs"
    gamma_proxy: bool = True
    gamma: torch.Tensor | None = None
    hessian: torch.Tensor | None = None
    rotation: str = "none"
    workers: int = 1
    resume: bool = False
    verify_sqnr: bool = False
    sqnr_rows: int = 16
    source_repository: str | None = None
    source_revision: str | None = None
    verbose: bool = True


@dataclass(frozen=True)
class PlannedTensor:
    """One source tensor and its planned output representation."""

    info: TensorInfo
    shard_name: str
    quantized: bool

    def output_specs(self) -> dict[str, tuple[tuple[int, ...], str]]:
        """Return emitted key -> (shape, safetensors dtype) for this tensor."""
        if not self.quantized:
            return {self.info.name: (self.info.shape, self.info.dtype)}
        rows, columns = self.info.shape
        module = self.info.name.removesuffix(".weight")
        return {
            f"{module}.weight_packed": ((rows, columns // 2), "U8"),
            f"{module}.weight_scale": ((rows, columns // 32), "U8"),
        }


@dataclass(frozen=True)
class ModelPlan:
    """Header-only conversion plan produced before tensors enter device memory."""

    input_format: InputFormat
    policy: QuantizationPolicy
    shards: tuple[ShardFile, ...]
    tensors: tuple[PlannedTensor, ...]
    gamma_proxy_sources: tuple[tuple[str, str], ...]
    source_data_bytes: int
    projected_output_data_bytes: int

    @property
    def target_names(self) -> frozenset[str]:
        """Source tensor names selected for MXFP4."""
        return frozenset(item.info.name for item in self.tensors if item.quantized)

    @property
    def target_modules(self) -> list[str]:
        """Concrete module names selected for MXFP4."""
        return sorted(name.removesuffix(".weight") for name in self.target_names)

    @property
    def real_modules(self) -> list[str]:
        """All checkpoint modules carrying a weight tensor."""
        return sorted(
            item.info.name.removesuffix(".weight")
            for item in self.tensors
            if item.info.name.endswith(".weight")
        )

    @property
    def ignored_modules(self) -> list[str]:
        """Concrete weighted modules intentionally kept unquantized."""
        targets = set(self.target_modules)
        return [module for module in self.real_modules if module not in targets]

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serializable dry-run summary."""
        target_source_bytes = sum(
            _tensor_data_bytes(item.info) for item in self.tensors if item.quantized
        )
        return {
            "input_format": self.input_format.kind,
            "policy": self.policy.name,
            "policy_description": self.policy.description,
            "shards": len(self.shards),
            "source_tensors": len(self.tensors),
            "target_tensors": len(self.target_names),
            "gamma_proxy_targets": len(self.gamma_proxy_sources),
            "passthrough_tensors": len(self.tensors) - len(self.target_names),
            "source_data_bytes": self.source_data_bytes,
            "target_source_bytes": target_source_bytes,
            "projected_output_data_bytes": self.projected_output_data_bytes,
            "projected_compression_ratio": round(
                self.source_data_bytes / self.projected_output_data_bytes, 4
            ),
        }


_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}
_FLOAT_DTYPES = frozenset({"BF16", "F16", "F32"})


def _tensor_data_bytes(info: TensorInfo) -> int:
    offset_bytes = info.data_offsets[1] - info.data_offsets[0]
    dtype_bytes = _DTYPE_BYTES.get(info.dtype)
    if dtype_bytes is None:
        return offset_bytes
    expected = math.prod(info.shape) * dtype_bytes
    if expected != offset_bytes:
        raise ValueError(
            f"Tensor {info.name!r} has inconsistent shape/dtype/data offsets in its header"
        )
    return expected


def _projected_tensor_bytes(item: PlannedTensor) -> int:
    if not item.quantized:
        return _tensor_data_bytes(item.info)
    rows, columns = item.info.shape
    return rows * (columns // 2) + rows * (columns // 32)


def _validate_run_options(cfg: QuantizeConfig) -> None:
    if cfg.rotation != "none":
        raise ValueError(
            "Rotation is disabled: the current implementation does not fold the transform "
            "through every required producer/consumer module"
        )
    if cfg.gamma is not None or cfg.hessian is not None:
        raise ValueError(
            "Global calibration tensors are unsafe; calibration must be keyed per target module"
        )
    if cfg.workers != 1:
        raise ValueError("Only one GPU streaming worker is currently supported")
    if cfg.sqnr_rows <= 0:
        raise ValueError("sqnr_rows must be positive")
    if not isinstance(cfg.mse_clip_depth, int) or not 0 <= cfg.mse_clip_depth <= 8:
        raise ValueError("mse_clip_depth must be an integer in [0, 8]")
    if not isinstance(cfg.hessian_rounding_sweeps, int) or not (
        0 <= cfg.hessian_rounding_sweeps <= 4
    ):
        raise ValueError("hessian_rounding_sweeps must be an integer in [0, 4]")
    if not isinstance(cfg.hessian_error_feedback, bool):
        raise TypeError("hessian_error_feedback must be a boolean")
    if not isinstance(cfg.hessian_feedback_activation_order, bool):
        raise TypeError("hessian_feedback_activation_order must be a boolean")
    if not math.isfinite(cfg.hessian_feedback_damp_percent) or not (
        0.0 < cfg.hessian_feedback_damp_percent <= 100.0
    ):
        raise ValueError("hessian_feedback_damp_percent must be in (0, 100]")
    if cfg.hessian_feedback_max_mse_ratio is not None and (
        not math.isfinite(cfg.hessian_feedback_max_mse_ratio)
        or not 1.0 <= cfg.hessian_feedback_max_mse_ratio <= 4.0
    ):
        raise ValueError("hessian_feedback_max_mse_ratio must be in [1, 4] or None")
    if not isinstance(cfg.tensor_row_chunk_size, int) or cfg.tensor_row_chunk_size <= 0:
        raise ValueError("tensor_row_chunk_size must be a positive integer")
    if cfg.activation_stats is not None and cfg.method != "mse":
        raise ValueError("Activation calibration can only be used with method='mse'")
    if cfg.hessian_rounding_sweeps and (
        cfg.method != "mse"
        or cfg.activation_stats is None
        or cfg.calibration_objective != "block-hessian"
    ):
        raise ValueError(
            "Hessian rounding requires method='mse', --activation-stats, and "
            "calibration_objective='block-hessian'"
        )
    if cfg.hessian_error_feedback and (
        cfg.method != "mse"
        or cfg.activation_stats is None
        or cfg.calibration_objective != "block-hessian"
    ):
        raise ValueError(
            "Hessian error feedback requires method='mse', --activation-stats, and "
            "calibration_objective='block-hessian'"
        )
    if cfg.hessian_error_feedback and cfg.hessian_rounding_sweeps:
        raise ValueError(
            "Hessian coordinate rounding and error feedback are mutually exclusive"
        )
    if cfg.feedback_selection_stats is not None and not cfg.hessian_error_feedback:
        raise ValueError("feedback_selection_stats requires Hessian error feedback")


def plan_model(cfg: QuantizeConfig) -> ModelPlan:
    """Inspect a checkpoint header-only and return a validated conversion plan."""
    _validate_run_options(cfg)
    model_dir = Path(cfg.model_dir)
    input_format = detect_input_format(model_dir)
    if input_format.kind != "fp16":
        raise ValueError(
            f"Input checkpoint is {input_format.kind!r}; only plain BF16/FP16/FP32 is supported"
        )
    policy = resolve_policy(model_dir, cfg.policy)
    shards, source_weight_map = discover_shards(model_dir)

    planned: list[PlannedTensor] = []
    seen: set[str] = set()
    for shard in shards:
        if not shard.path.is_file():
            raise FileNotFoundError(f"Missing source shard: {shard.path}")
        for info in shard_tensor_info(shard).values():
            if info.name in seen:
                raise ValueError(f"Duplicate source tensor key: {info.name}")
            seen.add(info.name)
            if source_weight_map is not None:
                mapped_shard = source_weight_map.get(info.name)
                if mapped_shard != shard.path.name:
                    raise ValueError(
                        f"Source index maps {info.name!r} to {mapped_shard!r}, "
                        f"but it is stored in {shard.path.name!r}"
                    )
            quantized = policy.selects(info)
            if policy.matches_name(info.name):
                validate_selected_tensor(info, policy)
            planned.append(
                PlannedTensor(info=info, shard_name=shard.path.name, quantized=quantized)
            )

    if source_weight_map is not None:
        missing = sorted(set(source_weight_map).difference(seen))
        extra = sorted(seen.difference(source_weight_map))
        if missing or extra:
            raise ValueError(
                "Source index and shard headers disagree: "
                f"missing={missing[:5]}, unindexed={extra[:5]}"
            )

    target_count = sum(item.quantized for item in planned)
    if target_count == 0:
        raise ValueError(f"Policy {policy.name!r} selected no tensors")
    expected = expected_target_count(model_dir, policy)
    if expected is not None and target_count != expected:
        raise ValueError(
            f"Policy {policy.name!r} expected {expected} tensors from the architecture config, "
            f"but selected {target_count}"
        )

    tensor_info = {item.info.name: item.info for item in planned}
    gamma_proxy_sources: list[tuple[str, str]] = []
    if cfg.method == "mse" and cfg.gamma_proxy and cfg.activation_stats is None:
        for target in sorted(item.info.name for item in planned if item.quantized):
            source_name = policy.gamma_proxy_source(target)
            if source_name is None:
                continue
            source = tensor_info.get(source_name)
            if source is None:
                raise ValueError(
                    f"Gamma proxy source {source_name!r} is missing for target {target!r}"
                )
            target_info = tensor_info[target]
            expected_shape = (target_info.shape[-1],)
            if source.shape != expected_shape or source.dtype not in _FLOAT_DTYPES:
                raise ValueError(
                    f"Gamma proxy source {source_name!r} is {source.shape}/{source.dtype}, "
                    f"expected {expected_shape} with a floating dtype"
                )
            gamma_proxy_sources.append((target, source_name))

    source_data_bytes = sum(_tensor_data_bytes(item.info) for item in planned)
    projected_output_data_bytes = sum(_projected_tensor_bytes(item) for item in planned)
    return ModelPlan(
        input_format=input_format,
        policy=policy,
        shards=tuple(shards),
        tensors=tuple(planned),
        gamma_proxy_sources=tuple(gamma_proxy_sources),
        source_data_bytes=source_data_bytes,
        projected_output_data_bytes=projected_output_data_bytes,
    )


def _should_quantize(name: str, target_names: frozenset[str]) -> bool:
    """Return whether a source tensor name is in the validated target set."""
    return name in target_names


def quantize_shard(
    shard: ShardFile,
    cfg: QuantizeConfig,
    *,
    target_names: frozenset[str] | None = None,
    gamma_by_target: Mapping[str, torch.Tensor] | None = None,
    hessian_by_target: Mapping[str, torch.Tensor] | None = None,
    feedback_selection_hessian_by_target: Mapping[str, torch.Tensor] | None = None,
    sqnr_results: dict[str, float] | None = None,
    calibration_weighted_sqnr_results: dict[str, float] | None = None,
    feedback_selection_weighted_sqnr_results: dict[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    """Quantize selected weights in bounded row chunks and preserve all other tensors."""
    infos = shard_tensor_info(shard)
    if target_names is None:
        policy = resolve_policy(cfg.model_dir, cfg.policy)
        target_names = frozenset(info.name for info in infos.values() if policy.selects(info))

    device = torch.device(cfg.device)
    result: dict[str, torch.Tensor] = {}
    for name, info in infos.items():
        if _should_quantize(name, target_names):
            gamma = gamma_by_target.get(name) if gamma_by_target is not None else None
            hessian = hessian_by_target.get(name) if hessian_by_target is not None else None
            feedback_selection_hessian = (
                feedback_selection_hessian_by_target.get(name)
                if feedback_selection_hessian_by_target is not None
                else None
            )
            gamma_device = gamma.to(device=device) if gamma is not None else None
            hessian_device = hessian.to(device=device) if hessian is not None else None
            feedback_selection_hessian_device = (
                feedback_selection_hessian.to(device=device)
                if feedback_selection_hessian is not None
                else None
            )
            total_rows, columns = info.shape
            packed_parts: list[torch.Tensor] = []
            scale_parts: list[torch.Tensor] = []
            for start in range(0, total_rows, cfg.tensor_row_chunk_size):
                stop = min(start + cfg.tensor_row_chunk_size, total_rows)
                weight_chunk = read_tensor_row_range(
                    shard,
                    name,
                    start,
                    stop,
                    device=device,
                )
                if not torch.isfinite(weight_chunk).all():
                    raise ValueError(
                        f"Target tensor contains NaN or infinity in rows [{start}, {stop}): {name}"
                    )
                packed_chunk, scale_chunk = quantize_mxfp4(
                    weight_chunk,
                    scale_percentile=cfg.scale_percentile,
                    gamma=gamma_device,
                    hessian=hessian_device,
                    method=cfg.method,
                    mse_clip_depth=cfg.mse_clip_depth,
                    hessian_rounding_sweeps=cfg.hessian_rounding_sweeps,
                    hessian_error_feedback=cfg.hessian_error_feedback,
                    hessian_feedback_damp_percent=cfg.hessian_feedback_damp_percent,
                    hessian_feedback_activation_order=(
                        cfg.hessian_feedback_activation_order
                    ),
                    hessian_feedback_max_mse_ratio=cfg.hessian_feedback_max_mse_ratio,
                    feedback_selection_hessian=feedback_selection_hessian_device,
                )
                packed_parts.append(packed_chunk.cpu().contiguous())
                scale_parts.append(scale_chunk.cpu().contiguous())
                del weight_chunk, packed_chunk, scale_chunk

            packed = torch.cat(packed_parts, dim=0)
            scales = torch.cat(scale_parts, dim=0)
            module = name.removesuffix(".weight")
            result[f"{module}.weight_packed"] = packed
            result[f"{module}.weight_scale"] = scales

            if sqnr_results is not None:
                rows = min(cfg.sqnr_rows, total_rows)
                original = read_tensor_rows(shard, name, rows, device=device)
                reconstructed = dequant_mxfp4(
                    packed[:rows].to(device),
                    scales[:rows].to(device),
                    (rows, columns),
                )
                sqnr_results[name] = sqnr(original, reconstructed)
                if gamma is not None and calibration_weighted_sqnr_results is not None:
                    assert gamma_device is not None
                    calibration_weighted_sqnr_results[name] = channel_weighted_sqnr(
                        original, reconstructed, gamma_device
                    )
                elif hessian is not None and calibration_weighted_sqnr_results is not None:
                    assert hessian_device is not None
                    calibration_weighted_sqnr_results[name] = block_hessian_weighted_sqnr(
                        original, reconstructed, hessian_device
                    )
                if (
                    feedback_selection_hessian is not None
                    and feedback_selection_weighted_sqnr_results is not None
                ):
                    assert feedback_selection_hessian_device is not None
                    feedback_selection_weighted_sqnr_results[name] = (
                        block_hessian_weighted_sqnr(
                            original,
                            reconstructed,
                            feedback_selection_hessian_device,
                        )
                    )
                del original, reconstructed
            del (
                packed_parts,
                scale_parts,
                gamma_device,
                hessian_device,
                feedback_selection_hessian_device,
            )
        else:
            result[name] = read_tensor(shard, name, device="cpu").contiguous()
    return result


def _load_gamma_proxies(
    plan: ModelPlan,
) -> dict[str, torch.Tensor]:
    """Load small, validated RMSNorm proxies once and key them by target tensor."""
    if not plan.gamma_proxy_sources:
        return {}
    planned_by_name = {item.info.name: item for item in plan.tensors}
    shards_by_name = {shard.path.name: shard for shard in plan.shards}
    source_cache: dict[str, torch.Tensor] = {}
    result: dict[str, torch.Tensor] = {}
    for target_name, source_name in plan.gamma_proxy_sources:
        gamma = source_cache.get(source_name)
        if gamma is None:
            source = planned_by_name[source_name]
            gamma = read_tensor(
                shards_by_name[source.shard_name], source_name, device="cpu"
            ).float()
            if not torch.isfinite(gamma).all():
                raise ValueError(f"Gamma proxy contains NaN or infinity: {source_name}")
            gamma = gamma.abs().contiguous()
            source_cache[source_name] = gamma
        result[target_name] = gamma
    return result


def _load_activation_calibration(
    cfg: QuantizeConfig,
    plan: ModelPlan,
) -> CalibrationData | None:
    """Load exact per-target activation statistics when explicitly supplied."""
    if cfg.activation_stats is None:
        return None
    expected_widths = {
        item.info.name: item.info.shape[-1]
        for item in plan.tensors
        if item.quantized
    }
    return load_calibration_data(
        cfg.activation_stats,
        cfg.calibration_objective,
        expected_widths,
        expected_policy=plan.policy.name,
        expected_source_repository=cfg.source_repository,
        expected_source_revision=cfg.source_revision,
    )


def _load_feedback_selection_calibration(
    cfg: QuantizeConfig,
    plan: ModelPlan,
    training: CalibrationData | None,
) -> CalibrationData | None:
    """Load an independent block Hessian used only for feedback selection."""
    if cfg.feedback_selection_stats is None:
        return None
    if training is None or training.objective != "block-hessian":
        raise ValueError("Feedback selection requires block-Hessian training calibration")
    expected_widths = {
        item.info.name: item.info.shape[-1]
        for item in plan.tensors
        if item.quantized
    }
    selection = load_calibration_data(
        cfg.feedback_selection_stats,
        "block-hessian",
        expected_widths,
        expected_policy=plan.policy.name,
        expected_source_repository=cfg.source_repository,
        expected_source_revision=cfg.source_revision,
    )
    if selection.file_sha256 == training.file_sha256:
        raise ValueError("Feedback selection statistics must differ from training statistics")
    training_tokens = training.metadata.get("token_ids_sha256")
    selection_tokens = selection.metadata.get("token_ids_sha256")
    if training_tokens and training_tokens == selection_tokens:
        raise ValueError("Feedback selection and training statistics use identical token IDs")

    training_corpus = training.metadata.get("corpus_sha256")
    selection_corpus = selection.metadata.get("corpus_sha256")
    if training_corpus and training_corpus == selection_corpus:
        training_length = int(training.metadata["sequence_length"])
        selection_length = int(selection.metadata["sequence_length"])
        training_offset = int(training.metadata.get("sequence_offset", "0"))
        selection_offset = int(selection.metadata.get("sequence_offset", "0"))
        training_interval = (
            training_offset * training_length,
            (training_offset + int(training.metadata["num_sequences"])) * training_length,
        )
        selection_interval = (
            selection_offset * selection_length,
            (selection_offset + int(selection.metadata["num_sequences"])) * selection_length,
        )
        intervals_overlap = max(training_interval[0], selection_interval[0]) < min(
            training_interval[1], selection_interval[1]
        )
        if intervals_overlap:
            raise ValueError(
                "Feedback selection and training calibration token ranges overlap: "
                f"training={training_interval}, selection={selection_interval}"
            )
    return selection


def _expected_shard_specs(
    plan: ModelPlan,
    shard_name: str,
) -> dict[str, tuple[tuple[int, ...], str]]:
    specs: dict[str, tuple[tuple[int, ...], str]] = {}
    for item in plan.tensors:
        if item.shard_name == shard_name:
            specs.update(item.output_specs())
    return specs


def _verify_output_shard(path: Path, expected: dict[str, tuple[tuple[int, ...], str]]) -> None:
    shard = ShardFile(path=path, weight_map={})
    actual = shard_tensor_info(shard)
    if set(actual) != set(expected):
        missing = sorted(set(expected).difference(actual))
        extra = sorted(set(actual).difference(expected))
        raise ValueError(
            f"Emitted shard {path.name} has wrong keys: missing={missing[:5]}, extra={extra[:5]}"
        )
    for name, (shape, dtype) in expected.items():
        actual_info = actual[name]
        if actual_info.shape != shape or actual_info.dtype != dtype:
            raise ValueError(
                f"Emitted tensor {name!r} is {actual_info.shape}/{actual_info.dtype}, "
                f"expected {shape}/{dtype}"
            )


def _sample_existing_shard_sqnr(
    source_shard: ShardFile,
    output_path: Path,
    plan: ModelPlan,
    cfg: QuantizeConfig,
    results: dict[str, float],
    gamma_by_target: Mapping[str, torch.Tensor],
    hessian_by_target: Mapping[str, torch.Tensor],
    calibration_weighted_results: dict[str, float],
    feedback_selection_hessian_by_target: Mapping[str, torch.Tensor],
    feedback_selection_weighted_results: dict[str, float],
) -> None:
    """Recompute bounded SQNR samples for a structurally valid resumed shard."""
    output_shard = ShardFile(path=output_path, weight_map={})
    device = torch.device(cfg.device)
    for item in plan.tensors:
        if item.shard_name != source_shard.path.name or not item.quantized:
            continue
        name = item.info.name
        rows = min(cfg.sqnr_rows, item.info.shape[0])
        module = name.removesuffix(".weight")
        original = read_tensor_rows(source_shard, name, rows, device=device)
        packed = read_tensor_rows(output_shard, f"{module}.weight_packed", rows, device=device)
        scales = read_tensor_rows(output_shard, f"{module}.weight_scale", rows, device=device)
        reconstructed = dequant_mxfp4(
            packed,
            scales,
            (rows, item.info.shape[1]),
        )
        results[name] = sqnr(original, reconstructed)
        gamma = gamma_by_target.get(name)
        if gamma is not None:
            calibration_weighted_results[name] = channel_weighted_sqnr(
                original, reconstructed, gamma
            )
        hessian = hessian_by_target.get(name)
        if hessian is not None:
            calibration_weighted_results[name] = block_hessian_weighted_sqnr(
                original, reconstructed, hessian
            )
        feedback_selection_hessian = feedback_selection_hessian_by_target.get(name)
        if feedback_selection_hessian is not None:
            feedback_selection_weighted_results[name] = block_hessian_weighted_sqnr(
                original,
                reconstructed,
                feedback_selection_hessian,
            )
        del original, packed, scales, reconstructed


def _atomic_save_shard(tensors: dict[str, torch.Tensor], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.incomplete")
    temporary.unlink(missing_ok=True)
    try:
        save_file(tensors, str(temporary))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_output_path(model_dir: Path, output_dir: Path, resume: bool) -> None:
    source = model_dir.resolve()
    output = output_dir.resolve()
    if source == output or output.is_relative_to(source):
        raise ValueError("output_dir must not equal or be nested inside model_dir")
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}; use --resume for a partial run"
        )


def _run_identity(
    cfg: QuantizeConfig,
    plan: ModelPlan,
    activation_calibration: CalibrationData | None,
    feedback_selection_calibration: CalibrationData | None,
) -> dict[str, Any]:
    """Build the shard-producing identity that a resumed run must exactly match."""
    target_description = [
        {
            "name": item.info.name,
            "shape": list(item.info.shape),
            "dtype": item.info.dtype,
        }
        for item in sorted(plan.tensors, key=lambda value: value.info.name)
        if item.quantized
    ]
    target_digest = hashlib.sha256(
        json.dumps(target_description, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    if activation_calibration is not None:
        calibration_identity: dict[str, Any] = {
            "kind": "activation-stats",
            "objective": activation_calibration.objective,
            "file_sha256": activation_calibration.file_sha256,
        }
    elif plan.gamma_proxy_sources:
        calibration_identity = {
            "kind": "layernorm-gamma-proxy",
            "sources": list(plan.gamma_proxy_sources),
        }
    else:
        calibration_identity = {"kind": "none"}
    return {
        "schema_version": 1,
        "producer": {"name": "mxstream", "version": __version__},
        "policy": plan.policy.name,
        "method": cfg.method,
        "scale_percentile": cfg.scale_percentile if cfg.method == "mse" else None,
        "mse_clip_depth": cfg.mse_clip_depth if cfg.method == "mse" else None,
        "hessian_rounding_sweeps": (
            cfg.hessian_rounding_sweeps if cfg.method == "mse" else None
        ),
        "hessian_error_feedback": cfg.hessian_error_feedback if cfg.method == "mse" else None,
        "hessian_feedback_damp_percent": (
            cfg.hessian_feedback_damp_percent if cfg.hessian_error_feedback else None
        ),
        "hessian_feedback_activation_order": (
            cfg.hessian_feedback_activation_order if cfg.hessian_error_feedback else None
        ),
        "hessian_feedback_max_mse_ratio": (
            cfg.hessian_feedback_max_mse_ratio if cfg.hessian_error_feedback else None
        ),
        "feedback_selection_calibration": (
            {
                "kind": "activation-stats",
                "objective": feedback_selection_calibration.objective,
                "file_sha256": feedback_selection_calibration.file_sha256,
            }
            if feedback_selection_calibration is not None
            else None
        ),
        "calibration": calibration_identity,
        "target_plan_sha256": target_digest,
        "source": {
            "repository": cfg.source_repository,
            "revision": cfg.source_revision,
            "shards": [
                {
                    "name": shard.path.name,
                    "size": shard.path.stat().st_size,
                    "mtime_ns": shard.path.stat().st_mtime_ns,
                }
                for shard in plan.shards
            ],
        },
    }


def _prepare_run_marker(output_dir: Path, identity: dict[str, Any], resume: bool) -> None:
    """Create or validate the run marker before any output shard is reused."""
    marker_path = output_dir / "mxstream-run.json"
    if resume and any(output_dir.iterdir()):
        if not marker_path.is_file():
            raise ValueError(
                "Cannot safely resume: mxstream-run.json is missing from the non-empty output"
            )
        try:
            recorded = json.loads(marker_path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError("Cannot safely resume: mxstream-run.json is invalid") from exc
        if recorded != identity:
            raise ValueError(
                "Cannot safely resume: quantization settings, calibration, targets, or source changed"
            )
        return

    temporary = marker_path.with_name(f".{marker_path.name}.{os.getpid()}.incomplete")
    temporary.unlink(missing_ok=True)
    try:
        temporary.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
        temporary.replace(marker_path)
    finally:
        temporary.unlink(missing_ok=True)


def _manifest(
    cfg: QuantizeConfig,
    plan: ModelPlan,
    sqnr_results: dict[str, float],
    calibration_weighted_sqnr_results: dict[str, float],
    feedback_selection_weighted_sqnr_results: dict[str, float],
    activation_calibration: CalibrationData | None,
    feedback_selection_calibration: CalibrationData | None,
) -> dict[str, Any]:
    summary = plan.summary()
    values = list(sqnr_results.values())
    calibration_weighted_values = list(calibration_weighted_sqnr_results.values())
    feedback_selection_weighted_values = list(
        feedback_selection_weighted_sqnr_results.values()
    )
    gamma_proxy_source_kinds = sorted(
        {".".join(source_name.rsplit(".", 2)[-2:]) for _, source_name in plan.gamma_proxy_sources}
    )
    activation_summary: dict[str, Any] | None = None
    if activation_calibration is not None:
        metadata = activation_calibration.metadata
        activation_summary = {
            "objective": activation_calibration.objective,
            "statistic": {
                "mean-abs": "mean-absolute-module-input",
                "rms": "root-mean-square-module-input",
                "block-hessian": "block32-input-second-moment",
            }[activation_calibration.objective],
            "file_sha256": activation_calibration.file_sha256,
            "weighted_tensors": len(activation_calibration.tensors),
            "unweighted_tensors": len(plan.target_names) - len(activation_calibration.tensors),
            "source_repository": metadata.get("source_repository") or None,
            "source_revision": metadata.get("source_revision") or None,
            "num_sequences": int(metadata["num_sequences"]),
            "sequence_offset": int(metadata.get("sequence_offset", "0")),
            "sequence_length": int(metadata["sequence_length"]),
            "num_tokens": int(metadata["num_tokens"]),
            "corpus_sha256": metadata.get("corpus_sha256"),
            "token_ids_sha256": metadata.get("token_ids_sha256"),
            "hessian_damp": (
                float(metadata["hessian_damp"])
                if activation_calibration.objective == "block-hessian"
                else None
            ),
        }
    feedback_selection_summary: dict[str, Any] | None = None
    if feedback_selection_calibration is not None:
        metadata = feedback_selection_calibration.metadata
        feedback_selection_summary = {
            "objective": feedback_selection_calibration.objective,
            "statistic": "block32-input-second-moment",
            "role": "candidate-selection-only",
            "file_sha256": feedback_selection_calibration.file_sha256,
            "weighted_tensors": len(feedback_selection_calibration.tensors),
            "source_repository": metadata.get("source_repository") or None,
            "source_revision": metadata.get("source_revision") or None,
            "num_sequences": int(metadata["num_sequences"]),
            "sequence_offset": int(metadata.get("sequence_offset", "0")),
            "sequence_length": int(metadata["sequence_length"]),
            "num_tokens": int(metadata["num_tokens"]),
            "corpus_sha256": metadata.get("corpus_sha256"),
            "token_ids_sha256": metadata.get("token_ids_sha256"),
            "hessian_damp": float(metadata["hessian_damp"]),
        }
    weighted_sqnr_summary = (
        {
            "objective": (
                activation_calibration.objective
                if activation_calibration is not None
                else "layernorm-gamma-proxy"
            ),
            "count": len(calibration_weighted_values),
            "coverage": len(calibration_weighted_values)
            / (
                len(activation_calibration.tensors)
                if activation_calibration is not None
                else len(plan.gamma_proxy_sources)
            ),
            "minimum": min(calibration_weighted_values),
            "mean": sum(calibration_weighted_values) / len(calibration_weighted_values),
            "per_tensor": calibration_weighted_sqnr_results,
        }
        if calibration_weighted_values
        else None
    )
    feedback_selection_sqnr_summary = (
        {
            "objective": "heldout-block-hessian",
            "count": len(feedback_selection_weighted_values),
            "coverage": len(feedback_selection_weighted_values)
            / len(feedback_selection_calibration.tensors),
            "minimum": min(feedback_selection_weighted_values),
            "mean": sum(feedback_selection_weighted_values)
            / len(feedback_selection_weighted_values),
            "per_tensor": feedback_selection_weighted_sqnr_results,
        }
        if feedback_selection_weighted_values
        and feedback_selection_calibration is not None
        else None
    )
    summary.update(
        {
            "manifest_version": 1,
            "producer": {"name": "mxstream", "version": __version__},
            "source": {
                "repository": cfg.source_repository,
                "revision": cfg.source_revision,
            },
            "method": cfg.method,
            "scale_percentile": cfg.scale_percentile if cfg.method == "mse" else None,
            "mse_clip_depth": cfg.mse_clip_depth if cfg.method == "mse" else None,
            "hessian_rounding_sweeps": (
                cfg.hessian_rounding_sweeps if cfg.method == "mse" else None
            ),
            "hessian_error_feedback": (
                cfg.hessian_error_feedback if cfg.method == "mse" else None
            ),
            "hessian_feedback_damp_percent": (
                cfg.hessian_feedback_damp_percent if cfg.hessian_error_feedback else None
            ),
            "hessian_feedback_activation_order": (
                cfg.hessian_feedback_activation_order if cfg.hessian_error_feedback else None
            ),
            "hessian_feedback_max_mse_ratio": (
                cfg.hessian_feedback_max_mse_ratio if cfg.hessian_error_feedback else None
            ),
            "feedback_selection_calibration": feedback_selection_summary,
            "tensor_row_chunk_size": cfg.tensor_row_chunk_size,
            "weight_scale_selection": (
                "rtn-memoryless-minmax"
                if cfg.method == "rtn"
                else f"mse-activation-{activation_calibration.objective}"
                if activation_calibration is not None
                else "mse-layernorm-gamma-proxy"
                if plan.gamma_proxy_sources
                else "mse-unweighted"
            ),
            "activation_calibration": activation_summary,
            "gamma_proxy": {
                "sources": gamma_proxy_source_kinds,
                "weighted_tensors": len(plan.gamma_proxy_sources),
                "unweighted_tensors": len(plan.target_names) - len(plan.gamma_proxy_sources),
            }
            if plan.gamma_proxy_sources
            else None,
            "activation_quantization": "dynamic-mxfp4-group32",
            "rotation": "none",
            "target_modules": plan.target_modules,
            "ignored_modules": plan.ignored_modules,
            "sqnr_sample_rows": cfg.sqnr_rows if cfg.verify_sqnr else None,
            "sqnr_db": {
                "count": len(values),
                "coverage": len(values) / len(plan.target_names),
                "minimum": min(values),
                "mean": sum(values) / len(values),
                "per_tensor": sqnr_results,
            }
            if values
            else None,
            "calibration_weighted_sqnr_db": (
                weighted_sqnr_summary if activation_calibration is not None else None
            ),
            "feedback_selection_weighted_sqnr_db": feedback_selection_sqnr_summary,
            "gamma_weighted_sqnr_db": (
                weighted_sqnr_summary
                if activation_calibration is None and plan.gamma_proxy_sources
                else None
            ),
        }
    )
    return summary


def quantize_model(cfg: QuantizeConfig) -> int:
    """Convert a model into a structurally verified MXFP4 checkpoint."""
    plan = plan_model(cfg)
    model_dir = Path(cfg.model_dir)
    output_dir = Path(cfg.output_dir)
    _validate_output_path(model_dir, output_dir, cfg.resume)

    if cfg.verbose:
        print(json.dumps(plan.summary(), indent=2))
    activation_calibration = _load_activation_calibration(cfg, plan)
    feedback_selection_calibration = _load_feedback_selection_calibration(
        cfg,
        plan,
        activation_calibration,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _prepare_run_marker(
        output_dir,
        _run_identity(
            cfg,
            plan,
            activation_calibration,
            feedback_selection_calibration,
        ),
        cfg.resume,
    )

    sqnr_results: dict[str, float] = {}
    calibration_weighted_sqnr_results: dict[str, float] = {}
    feedback_selection_weighted_sqnr_results: dict[str, float] = {}
    if activation_calibration is None:
        gamma_by_target = _load_gamma_proxies(plan)
        hessian_by_target: Mapping[str, torch.Tensor] = {}
    elif activation_calibration.objective == "block-hessian":
        gamma_by_target = {}
        hessian_by_target = activation_calibration.tensors
    else:
        gamma_by_target = activation_calibration.tensors
        hessian_by_target = {}
    feedback_selection_hessian_by_target: Mapping[str, torch.Tensor] = (
        feedback_selection_calibration.tensors
        if feedback_selection_calibration is not None
        else {}
    )
    output_shards: list[Path] = []
    for index, shard in enumerate(plan.shards, start=1):
        output_path = output_dir / shard.path.name
        expected = _expected_shard_specs(plan, shard.path.name)
        if cfg.resume and output_path.exists():
            _verify_output_shard(output_path, expected)
            if cfg.verify_sqnr:
                _sample_existing_shard_sqnr(
                    shard,
                    output_path,
                    plan,
                    cfg,
                    sqnr_results,
                    gamma_by_target,
                    hessian_by_target,
                    calibration_weighted_sqnr_results,
                    feedback_selection_hessian_by_target,
                    feedback_selection_weighted_sqnr_results,
                )
            if cfg.verbose:
                print(f"[mxstream] [{index}/{len(plan.shards)}] resume {shard.path.name}")
        else:
            if output_path.exists():
                raise FileExistsError(f"Refusing to replace existing shard: {output_path}")
            if cfg.verbose:
                print(f"[mxstream] [{index}/{len(plan.shards)}] quantize {shard.path.name}")
            tensors = quantize_shard(
                shard,
                cfg,
                target_names=plan.target_names,
                gamma_by_target=gamma_by_target,
                hessian_by_target=hessian_by_target,
                feedback_selection_hessian_by_target=(
                    feedback_selection_hessian_by_target
                ),
                sqnr_results=sqnr_results if cfg.verify_sqnr else None,
                calibration_weighted_sqnr_results=(
                    calibration_weighted_sqnr_results if cfg.verify_sqnr else None
                ),
                feedback_selection_weighted_sqnr_results=(
                    feedback_selection_weighted_sqnr_results
                    if cfg.verify_sqnr
                    else None
                ),
            )
            _atomic_save_shard(tensors, output_path)
            del tensors
            _verify_output_shard(output_path, expected)
            if torch.device(cfg.device).type == "cuda":
                torch.cuda.empty_cache()
        output_shards.append(output_path)

    manifest = _manifest(
        cfg,
        plan,
        sqnr_results,
        calibration_weighted_sqnr_results,
        feedback_selection_weighted_sqnr_results,
        activation_calibration,
        feedback_selection_calibration,
    )
    assemble_output_dir(
        model_dir,
        output_dir,
        output_shards,
        target_modules=plan.target_modules,
        ignored_modules=plan.ignored_modules,
        real_modules=plan.real_modules,
        manifest=manifest,
    )
    gaps = verify_emitted_config(output_dir, plan.real_modules)
    if gaps:
        raise ValueError(f"Emitted config coverage failed: {gaps[:10]}")
    return len(plan.shards)
