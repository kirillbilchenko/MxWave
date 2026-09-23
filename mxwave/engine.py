"""GPU-streaming, architecture-aware MXFP4 checkpoint conversion."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from . import __version__
from .calibration import CalibrationArtifact, CalibrationObjective, open_calibration_data
from .core import QuantizationMethod, dequant_mxfp4, quantize_mxfp4
from .format import InputFormat, detect_input_format
from .incremental_safetensors import IncrementalSafeTensorsWriter, source_data_start
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
_INTEGRITY_SIDECAR_FILENAME = "mxwave-shard-integrity.json"
_INTEGRITY_SCHEMA_VERSION = 1
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class _ShardIntegrityRecord:
    """Cryptographic identity of one complete output shard file."""

    bytes: int
    sha256: str

    def as_json(self) -> dict[str, int | str]:
        """Return the canonical JSON representation persisted in the ledger."""
        return {"bytes": self.bytes, "sha256": self.sha256}


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
    if not isinstance(cfg.tensor_row_chunk_size, int) or cfg.tensor_row_chunk_size <= 0:
        raise ValueError("tensor_row_chunk_size must be a positive integer")
    if cfg.activation_stats is not None and cfg.method != "mse":
        raise ValueError("Activation calibration can only be used with method='mse'")


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
    sqnr_results: dict[str, float] | None = None,
    calibration_weighted_sqnr_results: dict[str, float] | None = None,
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
            gamma_device = gamma.to(device=device) if gamma is not None else None
            hessian_device = hessian.to(device=device) if hessian is not None else None
            total_rows, columns = info.shape
            # This compatibility helper necessarily returns a materialized
            # shard mapping, but fill each output tensor in place so row
            # chunking does not also retain a list plus torch.cat copy. The
            # production model path writes these chunks directly to disk.
            packed = torch.empty((total_rows, columns // 2), dtype=torch.uint8)
            scales = torch.empty((total_rows, columns // 32), dtype=torch.uint8)
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
                )
                packed[start:stop].copy_(packed_chunk, non_blocking=False)
                scales[start:stop].copy_(scale_chunk, non_blocking=False)
                del weight_chunk, packed_chunk, scale_chunk

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
                del original, reconstructed
            del gamma, hessian, gamma_device, hessian_device
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
) -> CalibrationArtifact | None:
    """Open exact per-target activation statistics for non-caching access."""
    if cfg.activation_stats is None:
        return None
    expected_widths = {
        item.info.name: item.info.shape[-1]
        for item in plan.tensors
        if item.quantized
    }
    artifact = open_calibration_data(
        cfg.activation_stats,
        cfg.calibration_objective,
        expected_widths,
        expected_policy=plan.policy.name,
        expected_source_repository=cfg.source_repository,
        expected_source_revision=cfg.source_revision,
    )
    # Fail before emitting any checkpoint shard while retaining only one
    # statistic tensor at a time.
    artifact.validate_all()
    return artifact


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
            gamma_device = gamma.to(device=device)
            calibration_weighted_results[name] = channel_weighted_sqnr(
                original, reconstructed, gamma_device
            )
            del gamma_device
        del gamma
        hessian = hessian_by_target.get(name)
        if hessian is not None:
            hessian_device = hessian.to(device=device)
            calibration_weighted_results[name] = block_hessian_weighted_sqnr(
                original, reconstructed, hessian_device
            )
            del hessian_device
        del hessian
        del original, packed, scales, reconstructed


def _fsync_directory(path: Path) -> None:
    """Persist a rename or unlink in one directory before returning."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_quantize_shard(
    shard: ShardFile,
    cfg: QuantizeConfig,
    path: Path,
    expected: Mapping[str, tuple[tuple[int, ...], str]],
    *,
    target_names: frozenset[str],
    gamma_by_target: Mapping[str, torch.Tensor],
    hessian_by_target: Mapping[str, torch.Tensor],
) -> None:
    """Quantize and atomically emit one shard without full host tensors."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.incomplete")
    temporary.unlink(missing_ok=True)
    try:
        infos = shard_tensor_info(shard)
        device = torch.device(cfg.device)
        with shard.path.open("rb") as source, temporary.open("w+b") as destination:
            source_payload_start = source_data_start(source, shard.path)
            writer = IncrementalSafeTensorsWriter(destination, expected)
            for name, info in infos.items():
                if not _should_quantize(name, target_names):
                    writer.copy_tensor_payload(
                        name,
                        source,
                        shard.path,
                        info,
                        source_payload_start=source_payload_start,
                    )
                    continue

                gamma = gamma_by_target.get(name)
                hessian = hessian_by_target.get(name)
                gamma_device = gamma.to(device=device) if gamma is not None else None
                hessian_device = hessian.to(device=device) if hessian is not None else None
                total_rows, columns = info.shape
                module = name.removesuffix(".weight")
                packed_name = f"{module}.weight_packed"
                scale_name = f"{module}.weight_scale"
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
                            "Target tensor contains NaN or infinity in rows "
                            f"[{start}, {stop}): {name}"
                        )
                    packed_chunk, scale_chunk = quantize_mxfp4(
                        weight_chunk,
                        scale_percentile=cfg.scale_percentile,
                        gamma=gamma_device,
                        hessian=hessian_device,
                        method=cfg.method,
                        mse_clip_depth=cfg.mse_clip_depth,
                    )
                    expected_packed_shape = (stop - start, columns // 2)
                    expected_scale_shape = (stop - start, columns // 32)
                    if tuple(packed_chunk.shape) != expected_packed_shape:
                        raise AssertionError(
                            f"Packed chunk for {name!r} is {tuple(packed_chunk.shape)}, "
                            f"expected {expected_packed_shape}"
                        )
                    if tuple(scale_chunk.shape) != expected_scale_shape:
                        raise AssertionError(
                            f"Scale chunk for {name!r} is {tuple(scale_chunk.shape)}, "
                            f"expected {expected_scale_shape}"
                        )
                    writer.write_u8_chunk(packed_name, packed_chunk)
                    writer.write_u8_chunk(scale_name, scale_chunk)
                    del weight_chunk, packed_chunk, scale_chunk
                del gamma, hessian, gamma_device, hessian_device
            writer.finish()
            os.fsync(destination.fileno())
        _verify_output_shard(temporary, dict(expected))
        temporary.replace(path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _stale_shard_temporary_names(expected_filenames: set[str]) -> tuple[re.Pattern[str], ...]:
    """Return exact patterns for temporary files created by the shard writer."""
    return tuple(
        re.compile(rf"\.{re.escape(name)}\.[0-9]+\.incomplete\Z")
        for name in sorted(expected_filenames)
    )


def _cleanup_stale_shard_temporaries(
    output_dir: Path,
    expected_filenames: set[str],
) -> None:
    """Remove only stale temporary files for shards in the current model plan."""
    if not output_dir.is_dir():
        return
    patterns = _stale_shard_temporary_names(expected_filenames)
    removed = False
    for candidate in output_dir.iterdir():
        if not any(pattern.fullmatch(candidate.name) for pattern in patterns):
            continue
        if candidate.is_dir() and not candidate.is_symlink():
            raise ValueError(
                f"Refusing to remove stale shard temporary directory: {candidate}"
            )
        candidate.unlink()
        removed = True
    if removed:
        _fsync_directory(output_dir)


def _validate_output_path(
    model_dir: Path,
    output_dir: Path,
    resume: bool,
    expected_filenames: set[str],
) -> None:
    source = model_dir.resolve()
    output = output_dir.resolve()
    if source == output or output.is_relative_to(source):
        raise ValueError("output_dir must not equal or be nested inside model_dir")
    _cleanup_stale_shard_temporaries(output_dir, expected_filenames)
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}; use --resume for a partial run"
        )


def _run_identity(
    cfg: QuantizeConfig,
    plan: ModelPlan,
    activation_calibration: CalibrationArtifact | None,
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
        "producer": {"name": "mxwave", "version": __version__},
        "policy": plan.policy.name,
        "method": cfg.method,
        "scale_percentile": cfg.scale_percentile if cfg.method == "mse" else None,
        "mse_clip_depth": cfg.mse_clip_depth if cfg.method == "mse" else None,
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


def _canonical_sha256(value: object) -> str:
    """Hash one JSON-compatible value using a stable canonical encoding."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _file_integrity(path: Path) -> _ShardIntegrityRecord:
    """Hash one shard in bounded chunks and return its exact file size."""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    stat_size = path.stat().st_size
    if size != stat_size:
        raise OSError(
            f"Shard {path.name} changed size while its integrity hash was computed: "
            f"read={size}, stat={stat_size}"
        )
    return _ShardIntegrityRecord(bytes=size, sha256=digest.hexdigest())


def _atomic_write_integrity_ledger(
    path: Path,
    *,
    run_identity_sha256: str,
    records: Mapping[str, _ShardIntegrityRecord],
) -> None:
    """Atomically and durably persist the complete per-shard integrity ledger."""
    document = {
        "schema_version": _INTEGRITY_SCHEMA_VERSION,
        "algorithm": "sha256",
        "run_identity_sha256": run_identity_sha256,
        "role": "resume-only metadata; final evidence is copied into mxwave-manifest.json",
        "shards": {
            name: record.as_json() for name, record in sorted(records.items())
        },
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.incomplete")
    temporary.unlink(missing_ok=True)
    try:
        with temporary.open("w") as stream:
            stream.write(json.dumps(document, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _parse_integrity_records(
    path: Path,
    *,
    run_identity_sha256: str,
    expected_filenames: set[str],
) -> dict[str, _ShardIntegrityRecord]:
    """Load and strictly validate one run-bound shard-integrity ledger."""
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Cannot safely resume: invalid shard-integrity ledger JSON"
        ) from exc
    if not isinstance(raw, dict):
        raise TypeError("Cannot safely resume: shard-integrity ledger must be an object")
    if raw.get("schema_version") != _INTEGRITY_SCHEMA_VERSION:
        raise ValueError("Cannot safely resume: unsupported shard-integrity ledger schema")
    if raw.get("algorithm") != "sha256":
        raise ValueError("Cannot safely resume: shard-integrity algorithm changed")
    if raw.get("run_identity_sha256") != run_identity_sha256:
        raise ValueError(
            "Cannot safely resume: shard-integrity ledger does not match run identity"
        )
    raw_records = raw.get("shards")
    if not isinstance(raw_records, dict):
        raise TypeError("Cannot safely resume: shard-integrity ledger has no shards object")

    records: dict[str, _ShardIntegrityRecord] = {}
    for name, raw_record in raw_records.items():
        if not isinstance(name, str) or name not in expected_filenames:
            raise ValueError(f"Cannot safely resume: unexpected integrity shard {name!r}")
        if not isinstance(raw_record, dict) or set(raw_record) != {"bytes", "sha256"}:
            raise TypeError(
                f"Cannot safely resume: invalid integrity record for {name!r}"
            )
        byte_count = raw_record.get("bytes")
        digest = raw_record.get("sha256")
        if (
            not isinstance(byte_count, int)
            or isinstance(byte_count, bool)
            or byte_count <= 0
            or not isinstance(digest, str)
            or _SHA256_PATTERN.fullmatch(digest) is None
        ):
            raise ValueError(
                f"Cannot safely resume: invalid size or SHA-256 for {name!r}"
            )
        records[name] = _ShardIntegrityRecord(bytes=byte_count, sha256=digest)
    return records


def _prepare_integrity_records(
    output_dir: Path,
    *,
    run_identity_sha256: str,
    expected_filenames: set[str],
    resume: bool,
) -> tuple[Path, dict[str, _ShardIntegrityRecord]]:
    """Create or load the run-bound ledger without trusting existing shards."""
    ledger_path = output_dir / _INTEGRITY_SIDECAR_FILENAME
    existing_shards = {
        path.name
        for path in output_dir.glob("*.safetensors")
        if path.name in expected_filenames
    }
    if resume:
        if ledger_path.is_file():
            records = _parse_integrity_records(
                ledger_path,
                run_identity_sha256=run_identity_sha256,
                expected_filenames=expected_filenames,
            )
            orphan_shards = sorted(existing_shards.difference(records))
            if orphan_shards:
                raise ValueError(
                    "Cannot safely resume: existing shard has no integrity record; this is an "
                    f"untrusted crash window: {orphan_shards[:3]}"
                )
            return ledger_path, records
        if existing_shards:
            raise ValueError(
                "Cannot safely resume: shard-integrity ledger is missing while output shards "
                f"exist ({sorted(existing_shards)[:3]}). This is an untrusted crash window."
            )
    records = {}
    _atomic_write_integrity_ledger(
        ledger_path,
        run_identity_sha256=run_identity_sha256,
        records=records,
    )
    return ledger_path, records


def _validate_resumed_shard_integrity(
    path: Path,
    records: Mapping[str, _ShardIntegrityRecord],
) -> None:
    """Require a matching recorded size and full-file SHA-256 before reuse."""
    recorded = records.get(path.name)
    if recorded is None:
        raise ValueError(
            f"Cannot safely resume: existing shard {path.name!r} has no integrity record. "
            "This is an untrusted crash window."
        )
    actual_size = path.stat().st_size
    if actual_size != recorded.bytes:
        raise ValueError(
            f"Cannot safely resume: shard {path.name!r} size mismatch "
            f"(recorded={recorded.bytes}, actual={actual_size})"
        )
    actual = _file_integrity(path)
    if actual.sha256 != recorded.sha256:
        raise ValueError(
            f"Cannot safely resume: shard {path.name!r} SHA-256 mismatch"
        )


def _prepare_run_marker(output_dir: Path, identity: dict[str, Any], resume: bool) -> None:
    """Create or validate the run marker before any output shard is reused."""
    marker_path = output_dir / "mxwave-run.json"
    if resume and any(output_dir.iterdir()):
        if not marker_path.is_file():
            raise ValueError(
                "Cannot safely resume: mxwave-run.json is missing from the non-empty output"
            )
        try:
            recorded = json.loads(marker_path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError("Cannot safely resume: mxwave-run.json is invalid") from exc
        if recorded != identity:
            raise ValueError(
                "Cannot safely resume: quantization settings, calibration, targets, or source changed"
            )
        return

    temporary = marker_path.with_name(f".{marker_path.name}.{os.getpid()}.incomplete")
    temporary.unlink(missing_ok=True)
    try:
        with temporary.open("w") as stream:
            stream.write(json.dumps(identity, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(marker_path)
        _fsync_directory(marker_path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _manifest(
    cfg: QuantizeConfig,
    plan: ModelPlan,
    sqnr_results: dict[str, float],
    calibration_weighted_sqnr_results: dict[str, float],
    activation_calibration: CalibrationArtifact | None,
    *,
    run_identity_sha256: str,
    shard_integrity: Mapping[str, _ShardIntegrityRecord],
) -> dict[str, Any]:
    summary = plan.summary()
    values = list(sqnr_results.values())
    calibration_weighted_values = list(calibration_weighted_sqnr_results.values())
    ordered_shard_integrity = {
        name: record.as_json() for name, record in sorted(shard_integrity.items())
    }
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
    summary.update(
        {
            "manifest_version": 1,
            "producer": {"name": "mxwave", "version": __version__},
            "source": {
                "repository": cfg.source_repository,
                "revision": cfg.source_revision,
            },
            "method": cfg.method,
            "scale_percentile": cfg.scale_percentile if cfg.method == "mse" else None,
            "mse_clip_depth": cfg.mse_clip_depth if cfg.method == "mse" else None,
            "tensor_row_chunk_size": cfg.tensor_row_chunk_size,
            "host_memory_contract": {
                "output_emission": "incremental-safetensors",
                "passthrough_copy_chunk_bytes": 8 * 1024**2,
                "calibration_tensor_loading": (
                    "on-demand-non-caching"
                    if activation_calibration is not None
                    else None
                ),
                "scope": (
                    "materialized tensor payloads; allocator, CUDA workspace, mmap, and "
                    "filesystem cache are additional"
                ),
            },
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
            "shard_integrity": {
                "algorithm": "sha256",
                "run_identity_sha256": run_identity_sha256,
                "complete": len(ordered_shard_integrity) == len(plan.shards),
                "shard_count": len(ordered_shard_integrity),
                "file_bytes": sum(record.bytes for record in shard_integrity.values()),
                "aggregate_sha256": _canonical_sha256(ordered_shard_integrity),
                "per_shard": ordered_shard_integrity,
            },
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
    expected_filenames = {shard.path.name for shard in plan.shards}
    _validate_output_path(
        model_dir,
        output_dir,
        cfg.resume,
        expected_filenames,
    )
    if cfg.resume and output_dir.is_dir():
        unexpected = sorted(
            path.name
            for path in output_dir.glob("*.safetensors")
            if path.name not in expected_filenames
        )
        if unexpected:
            raise ValueError(
                f"Cannot safely resume with unexpected output shards: {unexpected[:5]}"
            )

    if cfg.verbose:
        print(json.dumps(plan.summary(), indent=2))
    activation_calibration = _load_activation_calibration(cfg, plan)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_identity = _run_identity(
        cfg,
        plan,
        activation_calibration,
    )
    run_identity_sha256 = _canonical_sha256(run_identity)
    _prepare_run_marker(
        output_dir,
        run_identity,
        cfg.resume,
    )
    integrity_path, integrity_records = _prepare_integrity_records(
        output_dir,
        run_identity_sha256=run_identity_sha256,
        expected_filenames=expected_filenames,
        resume=cfg.resume,
    )

    sqnr_results: dict[str, float] = {}
    calibration_weighted_sqnr_results: dict[str, float] = {}
    gamma_by_target: Mapping[str, torch.Tensor]
    hessian_by_target: Mapping[str, torch.Tensor]
    if activation_calibration is None:
        gamma_by_target = _load_gamma_proxies(plan)
        hessian_by_target = {}
    elif activation_calibration.objective == "block-hessian":
        gamma_by_target = {}
        hessian_by_target = activation_calibration.tensors
    else:
        gamma_by_target = activation_calibration.tensors
        hessian_by_target = {}
    output_shards: list[Path] = []
    for index, shard in enumerate(plan.shards, start=1):
        output_path = output_dir / shard.path.name
        expected = _expected_shard_specs(plan, shard.path.name)
        if cfg.resume and output_path.exists():
            _validate_resumed_shard_integrity(output_path, integrity_records)
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
                )
            if cfg.verbose:
                print(f"[mxwave] [{index}/{len(plan.shards)}] resume {shard.path.name}")
        else:
            if output_path.exists():
                raise FileExistsError(f"Refusing to replace existing shard: {output_path}")
            if cfg.verbose:
                print(f"[mxwave] [{index}/{len(plan.shards)}] quantize {shard.path.name}")
            _atomic_quantize_shard(
                shard,
                cfg,
                output_path,
                expected,
                target_names=plan.target_names,
                gamma_by_target=gamma_by_target,
                hessian_by_target=hessian_by_target,
            )
            _verify_output_shard(output_path, expected)
            integrity_records[shard.path.name] = _file_integrity(output_path)
            _atomic_write_integrity_ledger(
                integrity_path,
                run_identity_sha256=run_identity_sha256,
                records=integrity_records,
            )
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
                )
            if torch.device(cfg.device).type == "cuda":
                torch.cuda.empty_cache()
        output_shards.append(output_path)

    if activation_calibration is not None:
        activation_calibration.verify_unchanged(verify_sha256=True)

    if set(integrity_records) != expected_filenames:
        missing = sorted(expected_filenames.difference(integrity_records))
        extra = sorted(set(integrity_records).difference(expected_filenames))
        raise AssertionError(
            "Shard integrity coverage failed before final assembly: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    manifest = _manifest(
        cfg,
        plan,
        sqnr_results,
        calibration_weighted_sqnr_results,
        activation_calibration,
        run_identity_sha256=run_identity_sha256,
        shard_integrity=integrity_records,
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
