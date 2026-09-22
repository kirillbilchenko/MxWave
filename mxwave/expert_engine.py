"""Bounded, adapter-driven MXFP4 emission for fused MoE expert banks.

This engine is intentionally separate from the calibrated dense converter. It
supports one narrow contract: unfold validated routed-expert banks, quantize
each logical two-dimensional matrix with memoryless RTN or unweighted MSE scale
search, and preserve all other tensors in their source representation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from . import __version__
from .adapters import resolve_expert_layout
from .core import QuantizationMethod, dequant_mxfp4, quantize_mxfp4
from .expert_ir import ExpertQuantizationLayout, LogicalExpertMatrix
from .format import InputFormat, detect_input_format, load_config_json
from .output import assemble_output_dir, verify_emitted_config
from .shard import (
    ShardFile,
    TensorInfo,
    discover_shards,
    read_tensor,
    shard_tensor_info,
    tensor_payload_sha256,
)
from .verify import sqnr

__all__ = [
    "ExpertQuantizationConfig",
    "ExpertQuantizationPlan",
    "ExpertQuantizationUnit",
    "PlannedExpertSource",
    "emit_expert_unit",
    "plan_expert_model",
    "quantize_expert_model",
]

_DEFAULT_HOST_TENSOR_CAP_BYTES = 1024**3
_DEFAULT_MSE_SCALE_PERCENTILE = 99.5
_DEFAULT_MSE_CLIP_DEPTH = 4
_INTEGRITY_SIDECAR_FILENAME = "mxwave-shard-integrity.json"
_INTEGRITY_SCHEMA_VERSION = 1
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
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
UnitKind = Literal["expert-bank", "passthrough"]
TensorSpec = tuple[tuple[int, ...], str]


@dataclass(frozen=True)
class _ShardIntegrityRecord:
    """Cryptographic identity of one complete output shard file."""

    bytes: int
    sha256: str

    def as_json(self) -> dict[str, int | str]:
        """Return the canonical JSON representation persisted in the sidecar."""
        return {"bytes": self.bytes, "sha256": self.sha256}


@dataclass
class ExpertQuantizationConfig:
    """Options for the experimental routed-expert converter."""

    model_dir: str | Path = ""
    output_dir: str | Path = ""
    device: torch.device | str = "cuda"
    method: QuantizationMethod = "rtn"
    scale_percentile: float = _DEFAULT_MSE_SCALE_PERCENTILE
    mse_clip_depth: int = _DEFAULT_MSE_CLIP_DEPTH
    tensor_row_chunk_size: int = 2048
    host_tensor_cap_bytes: int = _DEFAULT_HOST_TENSOR_CAP_BYTES
    resume: bool = False
    verify_sqnr: bool = False
    sqnr_rows: int = 16
    source_repository: str | None = None
    source_revision: str | None = None
    verbose: bool = True


@dataclass(frozen=True)
class PlannedExpertSource:
    """One source tensor and its exact emitted representation."""

    info: TensorInfo
    shard_name: str
    logical_matrices: tuple[LogicalExpertMatrix, ...] = ()

    @property
    def is_expert_bank(self) -> bool:
        """Return whether this source tensor is unfolded and quantized."""
        return bool(self.logical_matrices)

    def output_specs(self) -> dict[str, TensorSpec]:
        """Return all emitted keys, shapes, and safetensors dtypes."""
        if not self.is_expert_bank:
            return {self.info.name: (self.info.shape, self.info.dtype)}
        result: dict[str, TensorSpec] = {}
        for matrix in self.logical_matrices:
            for spec in matrix.output_specs():
                if spec.name in result:
                    raise ValueError(f"Repeated expert output tensor: {spec.name}")
                result[spec.name] = (spec.shape, spec.dtype)
        return result

    @property
    def projected_data_bytes(self) -> int:
        """Return emitted tensor-data bytes for this source tensor."""
        if not self.is_expert_bank:
            return _tensor_data_bytes(self.info)
        return sum(math.prod(shape) * _dtype_bytes(dtype) for shape, dtype in self.output_specs().values())


@dataclass(frozen=True)
class ExpertQuantizationUnit:
    """One independently resumable, cap-bounded output safetensors shard."""

    filename: str
    kind: UnitKind
    sources: tuple[PlannedExpertSource, ...]
    projected_data_bytes: int

    def __post_init__(self) -> None:
        """Validate unit identity and source grouping."""
        if not self.filename.endswith(".safetensors"):
            raise ValueError(f"Invalid output shard filename: {self.filename}")
        if not self.sources or self.projected_data_bytes <= 0:
            raise ValueError("An emission unit must contain source tensor data")
        selected = sum(source.is_expert_bank for source in self.sources)
        if self.kind == "expert-bank" and (len(self.sources) != 1 or selected != 1):
            raise ValueError("An expert-bank unit must contain exactly one fused bank")
        if self.kind == "passthrough" and selected:
            raise ValueError("A passthrough unit cannot contain a selected expert bank")

    def expected_specs(self) -> dict[str, TensorSpec]:
        """Return the exact output contract for this shard."""
        result: dict[str, TensorSpec] = {}
        for source in self.sources:
            for name, spec in source.output_specs().items():
                if name in result:
                    raise ValueError(f"Repeated output tensor in {self.filename}: {name}")
                result[name] = spec
        return result


@dataclass(frozen=True)
class ExpertQuantizationPlan:
    """Header-only, exact-coverage plan for expert-only MXFP4 emission."""

    input_format: InputFormat
    layout: ExpertQuantizationLayout
    shards: tuple[ShardFile, ...]
    tensors: tuple[PlannedExpertSource, ...]
    units: tuple[ExpertQuantizationUnit, ...]
    source_data_bytes: int
    projected_output_data_bytes: int
    host_tensor_cap_bytes: int
    method: QuantizationMethod
    scale_percentile: float
    mse_clip_depth: int

    @property
    def target_modules(self) -> list[str]:
        """Return concrete routed-expert modules selected for MXFP4."""
        return sorted(
            matrix.output_module
            for source in self.tensors
            for matrix in source.logical_matrices
        )

    @property
    def real_modules(self) -> list[str]:
        """Return every runtime weighted module after bank unfolding."""
        passthrough = {
            source.info.name.removesuffix(".weight")
            for source in self.tensors
            if not source.is_expert_bank and source.info.name.endswith(".weight")
        }
        return sorted(passthrough.union(self.target_modules))

    @property
    def ignored_modules(self) -> list[str]:
        """Return concrete weighted modules intentionally left unquantized."""
        return sorted(set(self.real_modules).difference(self.target_modules))

    @property
    def config_target_patterns(self) -> list[str]:
        """Return adapter-declared selectors for validated runtime experts."""
        return list(self.layout.target_patterns)

    @property
    def config_ignored_patterns(self) -> list[str]:
        """Return concrete passthrough modules plus adapter-declared ignores."""
        return [*self.ignored_modules, *self.layout.ignored_patterns]

    @property
    def emitted_tensor_count(self) -> int:
        """Return the exact number of tensors in the output checkpoint."""
        return sum(len(unit.expected_specs()) for unit in self.units)

    @property
    def maximum_unit_bytes(self) -> int:
        """Return the largest planned resident tensor payload."""
        return max(unit.projected_data_bytes for unit in self.units)

    def summary(self) -> dict[str, Any]:
        """Return a compact JSON-serializable conversion summary."""
        passthrough_count = sum(not tensor.is_expert_bank for tensor in self.tensors)
        if self.method == "rtn":
            policy_description = "adapter-validated routed experts only; RTN baseline"
        else:
            policy_description = (
                "adapter-validated routed experts only; unweighted-MSE scale-search candidate"
            )
        summary: dict[str, Any] = {
            "input_format": self.input_format.kind,
            "policy": self.layout.policy_name,
            "policy_description": policy_description,
            "method": "rtn" if self.method == "rtn" else "unweighted-mse",
            "source_shards": len(self.shards),
            "output_shards": len(self.units),
            "source_tensors": len(self.tensors),
            "target_source_banks": self.layout.source_bank_count,
            "logical_matrices": self.layout.logical_matrix_count,
            "target_tensors": self.layout.logical_matrix_count,
            "quantized_output_tensors": self.layout.output_tensor_count,
            "passthrough_tensors": passthrough_count,
            "emitted_tensors": self.emitted_tensor_count,
            "source_data_bytes": self.source_data_bytes,
            "projected_output_data_bytes": self.projected_output_data_bytes,
            "projected_compression_ratio": round(
                self.source_data_bytes / self.projected_output_data_bytes, 4
            ),
            "host_tensor_cap_bytes": self.host_tensor_cap_bytes,
            "maximum_planned_unit_bytes": self.maximum_unit_bytes,
        }
        if self.method == "mse":
            summary.update(
                {
                    "scale_search": _scale_search_metadata(
                        self.scale_percentile,
                        self.mse_clip_depth,
                    ),
                }
            )
        return summary


def _dtype_bytes(dtype: str) -> int:
    try:
        return _DTYPE_BYTES[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported safetensors dtype: {dtype}") from exc


def _tensor_data_bytes(info: TensorInfo) -> int:
    size = info.data_offsets[1] - info.data_offsets[0]
    expected = math.prod(info.shape) * _dtype_bytes(info.dtype)
    if size != expected:
        raise ValueError(
            f"Tensor {info.name!r} has inconsistent shape/dtype/data offsets in its header"
        )
    return expected


def _scale_search_metadata(
    scale_percentile: float,
    mse_clip_depth: int,
) -> dict[str, float | int | bool]:
    """Describe the exact unweighted-MSE candidate set used by the core."""
    return {
        "scale_percentile": scale_percentile,
        "mse_clip_depth": mse_clip_depth,
        "includes_no_clipping_candidate": True,
    }


def _mse_selection_name(scale_percentile: float, mse_clip_depth: int) -> str:
    """Return the stable manifest label for one unweighted-MSE search."""
    percentile = format(scale_percentile, "g")
    return f"unweighted-mse-p{percentile}-depth{mse_clip_depth}-plus-no-clipping"


def _matches_config_pattern(pattern: str, module: str) -> bool:
    if pattern.startswith("re:"):
        return re.match(pattern[3:], module) is not None
    return pattern == module


def _validate_config_classification(plan: ExpertQuantizationPlan) -> None:
    targets = set(plan.target_modules)
    real_modules = plan.real_modules
    target_patterns = tuple(plan.config_target_patterns)
    ignored_patterns = tuple(plan.config_ignored_patterns)
    for module in real_modules:
        target_match = any(
            _matches_config_pattern(pattern, module)
            for pattern in target_patterns
        )
        ignore_match = any(
            _matches_config_pattern(pattern, module)
            for pattern in ignored_patterns
        )
        if target_match != (module in targets):
            raise ValueError(f"Compact expert target regex misclassifies {module!r}")
        if target_match and ignore_match:
            raise ValueError(f"Module is both targeted and ignored: {module!r}")
        if not target_match and not ignore_match:
            raise ValueError(f"Module is neither targeted nor ignored: {module!r}")


def _validate_options(cfg: ExpertQuantizationConfig) -> None:
    if cfg.method not in ("rtn", "mse"):
        raise ValueError(f"Unsupported quantization method: {cfg.method}")
    if (
        isinstance(cfg.scale_percentile, bool)
        or not isinstance(cfg.scale_percentile, (int, float))
        or not math.isfinite(cfg.scale_percentile)
        or not 0.0 < cfg.scale_percentile <= 100.0
    ):
        raise ValueError("scale_percentile must be finite and in (0, 100]")
    if (
        isinstance(cfg.mse_clip_depth, bool)
        or not isinstance(cfg.mse_clip_depth, int)
        or not 0 <= cfg.mse_clip_depth <= 8
    ):
        raise ValueError("mse_clip_depth must be an integer in [0, 8]")
    if not isinstance(cfg.tensor_row_chunk_size, int) or cfg.tensor_row_chunk_size <= 0:
        raise ValueError("tensor_row_chunk_size must be a positive integer")
    if not isinstance(cfg.host_tensor_cap_bytes, int) or cfg.host_tensor_cap_bytes <= 0:
        raise ValueError("host_tensor_cap_bytes must be a positive integer")
    if not isinstance(cfg.sqnr_rows, int) or cfg.sqnr_rows <= 0:
        raise ValueError("sqnr_rows must be a positive integer")


def _build_units(
    tensors: tuple[PlannedExpertSource, ...],
    layout: ExpertQuantizationLayout,
    cap_bytes: int,
) -> tuple[ExpertQuantizationUnit, ...]:
    by_name = {tensor.info.name: tensor for tensor in tensors}
    raw_units: list[tuple[UnitKind, tuple[PlannedExpertSource, ...]]] = []

    for bank in layout.banks:
        source = by_name[bank.source_name]
        if source.projected_data_bytes > cap_bytes:
            raise ValueError(
                f"Expert bank {bank.source_name!r} projects to "
                f"{source.projected_data_bytes} bytes, above host tensor cap {cap_bytes}"
            )
        raw_units.append(("expert-bank", (source,)))

    passthrough_group: list[PlannedExpertSource] = []
    passthrough_bytes = 0
    for source in tensors:
        if source.is_expert_bank:
            continue
        size = source.projected_data_bytes
        if size > cap_bytes:
            raise ValueError(
                f"Passthrough tensor {source.info.name!r} needs {size} bytes, "
                f"above host tensor cap {cap_bytes}; raise the cap explicitly"
            )
        if passthrough_group and passthrough_bytes + size > cap_bytes:
            raw_units.append(("passthrough", tuple(passthrough_group)))
            passthrough_group = []
            passthrough_bytes = 0
        passthrough_group.append(source)
        passthrough_bytes += size
    if passthrough_group:
        raw_units.append(("passthrough", tuple(passthrough_group)))

    total = len(raw_units)
    units = tuple(
        ExpertQuantizationUnit(
            filename=f"model-{index:05d}-of-{total:05d}.safetensors",
            kind=kind,
            sources=sources,
            projected_data_bytes=sum(source.projected_data_bytes for source in sources),
        )
        for index, (kind, sources) in enumerate(raw_units, start=1)
    )
    if any(unit.projected_data_bytes > cap_bytes for unit in units):
        raise AssertionError("Emission unit exceeded the validated host tensor cap")
    return units


def plan_expert_model(cfg: ExpertQuantizationConfig) -> ExpertQuantizationPlan:
    """Inspect a supported checkpoint header-only and plan exact expert emission."""
    _validate_options(cfg)
    model_dir = Path(cfg.model_dir)
    input_format = detect_input_format(model_dir)
    if input_format.kind != "fp16":
        raise ValueError(
            f"Input checkpoint is {input_format.kind!r}; expert conversion requires "
            "plain float weights"
        )
    config = load_config_json(model_dir)
    shards, source_weight_map = discover_shards(model_dir)

    infos: list[tuple[TensorInfo, str]] = []
    seen: set[str] = set()
    for shard in shards:
        if not shard.path.is_file():
            raise FileNotFoundError(f"Missing source shard: {shard.path}")
        for info in shard_tensor_info(shard).values():
            if info.name in seen:
                raise ValueError(f"Duplicate source tensor key: {info.name}")
            seen.add(info.name)
            if source_weight_map is not None and source_weight_map.get(info.name) != shard.path.name:
                raise ValueError(
                    f"Source index maps {info.name!r} inconsistently with {shard.path.name!r}"
                )
            infos.append((info, shard.path.name))
    if source_weight_map is not None:
        missing = sorted(set(source_weight_map).difference(seen))
        extra = sorted(seen.difference(source_weight_map))
        if missing or extra:
            raise ValueError(
                "Source index and shard headers disagree: "
                f"missing={missing[:5]}, unindexed={extra[:5]}"
            )

    layout = resolve_expert_layout(config, (info for info, _ in infos))
    matrices_by_source: dict[str, list[LogicalExpertMatrix]] = {
        name: [] for name in layout.source_names
    }
    for matrix in layout.iter_logical_matrices():
        matrices_by_source[matrix.source_name].append(matrix)
    planned = tuple(
        PlannedExpertSource(
            info=info,
            shard_name=shard_name,
            logical_matrices=tuple(matrices_by_source.get(info.name, ())),
        )
        for info, shard_name in infos
    )
    selected_names = {source.info.name for source in planned if source.is_expert_bank}
    if selected_names != layout.source_names:
        raise AssertionError("Expert source-bank coverage changed after layout validation")

    output_owners: dict[str, str] = {}
    for source in planned:
        for output_name in source.output_specs():
            previous = output_owners.setdefault(output_name, source.info.name)
            if previous != source.info.name:
                raise ValueError(
                    f"Output tensor {output_name!r} collides between {previous!r} "
                    f"and {source.info.name!r}"
                )

    units = _build_units(planned, layout, cfg.host_tensor_cap_bytes)
    covered_sources = [source.info.name for unit in units for source in unit.sources]
    if len(covered_sources) != len(planned) or set(covered_sources) != seen:
        raise AssertionError("Emission units do not cover each source tensor exactly once")

    source_bytes = sum(_tensor_data_bytes(source.info) for source in planned)
    projected_bytes = sum(unit.projected_data_bytes for unit in units)
    plan = ExpertQuantizationPlan(
        input_format=input_format,
        layout=layout,
        shards=tuple(shards),
        tensors=planned,
        units=units,
        source_data_bytes=source_bytes,
        projected_output_data_bytes=projected_bytes,
        host_tensor_cap_bytes=cfg.host_tensor_cap_bytes,
        method=cfg.method,
        scale_percentile=cfg.scale_percentile,
        mse_clip_depth=cfg.mse_clip_depth,
    )
    _validate_config_classification(plan)
    return plan


def _quantize_logical_matrix(
    source_slice: Any,
    matrix: LogicalExpertMatrix,
    cfg: ExpertQuantizationConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device(cfg.device)
    rows, columns = matrix.shape
    packed_parts: list[torch.Tensor] = []
    scale_parts: list[torch.Tensor] = []
    for start in range(0, rows, cfg.tensor_row_chunk_size):
        stop = min(start + cfg.tensor_row_chunk_size, rows)
        weight = _read_logical_rows(
            source_slice,
            matrix,
            start,
            stop,
            device=device,
        )
        if not torch.isfinite(weight).all():
            raise ValueError(
                f"Expert matrix {matrix.output_module!r} contains NaN or infinity "
                f"in rows [{start}, {stop})"
            )
        packed, scales = quantize_mxfp4(
            weight,
            method=cfg.method,
            scale_percentile=cfg.scale_percentile,
            mse_clip_depth=cfg.mse_clip_depth,
        )
        packed_parts.append(packed.cpu().contiguous())
        scale_parts.append(scales.cpu().contiguous())
        del weight, packed, scales
    packed_result = torch.cat(packed_parts, dim=0)
    scale_result = torch.cat(scale_parts, dim=0)
    if tuple(packed_result.shape) != (rows, columns // 2):
        raise AssertionError(f"Packed expert shape changed for {matrix.output_module!r}")
    if tuple(scale_result.shape) != (rows, columns // 32):
        raise AssertionError(f"Expert scale shape changed for {matrix.output_module!r}")
    return packed_result, scale_result


def _read_logical_rows(
    source_slice: Any,
    matrix: LogicalExpertMatrix,
    start: int,
    stop: int,
    *,
    device: torch.device | str,
) -> torch.Tensor:
    """Slice logical rows from an already-open fused bank handle."""
    if not 0 <= start < stop <= matrix.shape[0]:
        raise ValueError(
            f"Logical row range [{start}, {stop}) is outside {matrix.output_module!r}"
        )
    sliced = cast(
        torch.Tensor,
        source_slice[
            matrix.expert_index,
            matrix.row_start + start : matrix.row_start + stop,
            :,
        ],
    )
    expected = (stop - start, matrix.shape[1])
    if tuple(sliced.shape) != expected:
        raise ValueError(
            f"Expert bank returned {tuple(sliced.shape)} for {matrix.output_module!r}, "
            f"expected {expected}"
        )
    return sliced.to(device)


def emit_expert_unit(
    unit: ExpertQuantizationUnit,
    cfg: ExpertQuantizationConfig,
    *,
    shards_by_name: dict[str, ShardFile],
    sqnr_results: dict[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    """Build one cap-bounded output shard without loading a full expert bank."""
    result: dict[str, torch.Tensor] = {}
    for source in unit.sources:
        shard = shards_by_name[source.shard_name]
        if source.is_expert_bank:
            # Keep one safetensors handle open for the complete bank task.  The
            # PySafeSlice object performs bounded expert/row reads and never
            # materializes the rank-3 bank.
            with safe_open(str(shard.path), framework="pt", device="cpu") as handle:
                source_slice = handle.get_slice(source.info.name)
                if tuple(source_slice.get_shape()) != source.info.shape:
                    raise ValueError(
                        f"Expert bank {source.info.name!r} changed shape after planning"
                    )
                for matrix in source.logical_matrices:
                    packed, scales = _quantize_logical_matrix(
                        source_slice,
                        matrix,
                        cfg,
                    )
                    packed_key = f"{matrix.output_module}.weight_packed"
                    scale_key = f"{matrix.output_module}.weight_scale"
                    result[packed_key] = packed
                    result[scale_key] = scales
                    if sqnr_results is not None:
                        rows = min(cfg.sqnr_rows, matrix.shape[0])
                        original = _read_logical_rows(
                            source_slice,
                            matrix,
                            0,
                            rows,
                            device=cfg.device,
                        )
                        reconstructed = dequant_mxfp4(
                            packed[:rows].to(cfg.device),
                            scales[:rows].to(cfg.device),
                            (rows, matrix.shape[1]),
                        )
                        sqnr_results[matrix.output_module] = sqnr(original, reconstructed)
                        del original, reconstructed
        else:
            result[source.info.name] = read_tensor(
                shard, source.info.name, device="cpu"
            ).contiguous()
    resident_bytes = sum(tensor.numel() * tensor.element_size() for tensor in result.values())
    if resident_bytes > cfg.host_tensor_cap_bytes:
        raise AssertionError(
            f"Emission unit {unit.filename} materialized {resident_bytes} bytes above "
            f"the configured cap {cfg.host_tensor_cap_bytes}"
        )
    return result


def _atomic_save_shard(tensors: dict[str, torch.Tensor], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.incomplete")
    temporary.unlink(missing_ok=True)
    try:
        save_file(tensors, str(temporary))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


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


def _atomic_write_integrity_sidecar(
    path: Path,
    *,
    run_identity_sha256: str,
    records: dict[str, _ShardIntegrityRecord],
) -> None:
    """Atomically persist the complete per-shard resume-integrity ledger."""
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
    finally:
        temporary.unlink(missing_ok=True)


def _parse_integrity_records(
    path: Path,
    *,
    run_identity_sha256: str,
    expected_filenames: set[str],
) -> dict[str, _ShardIntegrityRecord]:
    """Load and strictly validate one run-bound integrity sidecar."""
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Cannot safely resume expert conversion: invalid shard-integrity sidecar JSON"
        ) from exc
    if not isinstance(raw, dict):
        raise TypeError(
            "Cannot safely resume expert conversion: integrity sidecar must be an object"
        )
    if raw.get("schema_version") != _INTEGRITY_SCHEMA_VERSION:
        raise ValueError(
            "Cannot safely resume expert conversion: unsupported integrity sidecar schema"
        )
    if raw.get("algorithm") != "sha256":
        raise ValueError(
            "Cannot safely resume expert conversion: integrity sidecar algorithm changed"
        )
    if raw.get("run_identity_sha256") != run_identity_sha256:
        raise ValueError(
            "Cannot safely resume expert conversion: integrity sidecar does not match run "
            "identity"
        )
    raw_records = raw.get("shards")
    if not isinstance(raw_records, dict):
        raise TypeError(
            "Cannot safely resume expert conversion: integrity sidecar has no shards object"
        )

    records: dict[str, _ShardIntegrityRecord] = {}
    for name, raw_record in raw_records.items():
        if not isinstance(name, str) or name not in expected_filenames:
            raise ValueError(
                f"Cannot safely resume expert conversion: unexpected integrity shard {name!r}"
            )
        if not isinstance(raw_record, dict) or set(raw_record) != {"bytes", "sha256"}:
            raise TypeError(
                f"Cannot safely resume expert conversion: invalid integrity record for {name!r}"
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
                f"Cannot safely resume expert conversion: invalid size or SHA-256 for {name!r}"
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
    sidecar = output_dir / _INTEGRITY_SIDECAR_FILENAME
    existing_shards = sorted(
        path.name for path in output_dir.glob("*.safetensors") if path.name in expected_filenames
    )
    if resume:
        if sidecar.is_file():
            return sidecar, _parse_integrity_records(
                sidecar,
                run_identity_sha256=run_identity_sha256,
                expected_filenames=expected_filenames,
            )
        if existing_shards:
            raise ValueError(
                "Cannot safely resume expert conversion: shard-integrity sidecar is missing "
                f"while output shards exist ({existing_shards[:3]}). This is an untrusted crash "
                "window; remove the orphan shard(s) before retrying --resume."
            )
    records: dict[str, _ShardIntegrityRecord] = {}
    _atomic_write_integrity_sidecar(
        sidecar,
        run_identity_sha256=run_identity_sha256,
        records=records,
    )
    return sidecar, records


def _validate_resumed_shard_integrity(
    path: Path,
    records: dict[str, _ShardIntegrityRecord],
) -> None:
    """Require a matching recorded size and full-file SHA-256 before reuse."""
    recorded = records.get(path.name)
    if recorded is None:
        raise ValueError(
            f"Cannot safely resume expert conversion: existing shard {path.name!r} has no "
            "integrity record. This is an untrusted crash window; remove that orphan shard before "
            "retrying --resume."
        )
    actual_size = path.stat().st_size
    if actual_size != recorded.bytes:
        raise ValueError(
            f"Cannot safely resume expert conversion: shard {path.name!r} size mismatch "
            f"(recorded={recorded.bytes}, actual={actual_size})"
        )
    actual = _file_integrity(path)
    if actual.sha256 != recorded.sha256:
        raise ValueError(
            f"Cannot safely resume expert conversion: shard {path.name!r} SHA-256 mismatch"
        )


def _verify_output_shard(path: Path, expected: dict[str, TensorSpec]) -> None:
    actual = shard_tensor_info(ShardFile(path=path, weight_map={}))
    if set(actual) != set(expected):
        missing = sorted(set(expected).difference(actual))
        extra = sorted(set(actual).difference(expected))
        raise ValueError(
            f"Emitted shard {path.name} has wrong keys: missing={missing[:5]}, extra={extra[:5]}"
        )
    for name, (shape, dtype) in expected.items():
        info = actual[name]
        if info.shape != shape or info.dtype != dtype:
            raise ValueError(
                f"Emitted tensor {name!r} is {info.shape}/{info.dtype}, "
                f"expected {shape}/{dtype}"
            )


def _verify_passthrough_payloads(
    unit: ExpertQuantizationUnit,
    output_path: Path,
    shards_by_name: dict[str, ShardFile],
    results: dict[str, str],
) -> None:
    """Prove that one passthrough unit preserves every raw payload bit."""
    if unit.kind != "passthrough":
        return
    output_shard = ShardFile(path=output_path, weight_map={})
    for source in unit.sources:
        if source.is_expert_bank:
            raise AssertionError("Passthrough verification received an expert bank")
        source_shard = shards_by_name[source.shard_name]
        source_digest = tensor_payload_sha256(source_shard, source.info.name)
        output_digest = tensor_payload_sha256(output_shard, source.info.name)
        if output_digest != source_digest:
            raise ValueError(
                f"Passthrough tensor {source.info.name!r} changed raw payload bytes"
            )
        results[source.info.name] = source_digest


def _sample_existing_unit_sqnr(
    unit: ExpertQuantizationUnit,
    output_path: Path,
    cfg: ExpertQuantizationConfig,
    shards_by_name: dict[str, ShardFile],
    results: dict[str, float],
) -> None:
    """Recompute bounded SQNR for a resumed expert-bank output shard."""
    if unit.kind != "expert-bank":
        return
    source = unit.sources[0]
    source_shard = shards_by_name[source.shard_name]
    with (
        safe_open(str(source_shard.path), framework="pt", device="cpu") as source_handle,
        safe_open(str(output_path), framework="pt", device="cpu") as output_handle,
    ):
        source_slice = source_handle.get_slice(source.info.name)
        for matrix in source.logical_matrices:
            rows = min(cfg.sqnr_rows, matrix.shape[0])
            original = _read_logical_rows(
                source_slice,
                matrix,
                0,
                rows,
                device=cfg.device,
            )
            packed = cast(
                torch.Tensor,
                output_handle.get_slice(
                    f"{matrix.output_module}.weight_packed"
                )[:rows],
            ).to(cfg.device)
            scales = cast(
                torch.Tensor,
                output_handle.get_slice(
                    f"{matrix.output_module}.weight_scale"
                )[:rows],
            ).to(cfg.device)
            reconstructed = dequant_mxfp4(
                packed,
                scales,
                (rows, matrix.shape[1]),
            )
            results[matrix.output_module] = sqnr(original, reconstructed)
            del original, packed, scales, reconstructed


def _validate_output_path(model_dir: Path, output_dir: Path, resume: bool) -> None:
    source = model_dir.resolve()
    output = output_dir.resolve()
    if source == output or output.is_relative_to(source):
        raise ValueError("output_dir must not equal or be nested inside model_dir")
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}; use --resume for a partial run"
        )


def _run_identity(cfg: ExpertQuantizationConfig, plan: ExpertQuantizationPlan) -> dict[str, Any]:
    plan_description = {
        "layout": plan.layout.summary(),
        "banks": [
            {"name": bank.source_name, "shape": list(bank.shape), "dtype": bank.dtype}
            for bank in plan.layout.banks
        ],
        "matrices": [
            {
                "source": matrix.source_name,
                "expert_index": matrix.expert_index,
                "row_start": matrix.row_start,
                "row_stop": matrix.row_stop,
                "output_module": matrix.output_module,
            }
            for matrix in plan.layout.iter_logical_matrices()
        ],
        "target_patterns": plan.config_target_patterns,
        "ignored_patterns": list(plan.layout.ignored_patterns),
        "units": [
            {
                "filename": unit.filename,
                "kind": unit.kind,
                "sources": [source.info.name for source in unit.sources],
            }
            for unit in plan.units
        ],
    }
    plan_sha256 = hashlib.sha256(
        json.dumps(plan_description, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    source_shards: dict[str, dict[str, int | str]] = {}
    for shard in sorted(plan.shards, key=lambda item: item.path.name):
        integrity = _file_integrity(shard.path)
        source_shards[shard.path.name] = integrity.as_json()
    model_dir = Path(cfg.model_dir)
    source_identity: dict[str, Any] = {
        "repository": cfg.source_repository,
        "revision": cfg.source_revision,
        "config_sha256": hashlib.sha256(
            (model_dir / "config.json").read_bytes()
        ).hexdigest(),
        "shards": source_shards,
    }
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        source_identity["weight_index_sha256"] = hashlib.sha256(
            index_path.read_bytes()
        ).hexdigest()
    identity: dict[str, Any] = {
        "schema_version": 2,
        "engine": "adapter-driven-expert-quantization-v1",
        "producer": {"name": "mxwave", "version": __version__},
        "layout_policy": plan.layout.policy_name,
        "method": "rtn" if cfg.method == "rtn" else "unweighted-mse",
        "calibration": {"kind": "none"},
        "tensor_row_chunk_size": cfg.tensor_row_chunk_size,
        "host_tensor_cap_bytes": cfg.host_tensor_cap_bytes,
        "target_plan_sha256": plan_sha256,
        "source": source_identity,
    }
    if cfg.method == "mse":
        identity.update(
            {
                "scale_search": _scale_search_metadata(
                    cfg.scale_percentile,
                    cfg.mse_clip_depth,
                ),
            }
        )
    return identity


def _prepare_run_marker(output_dir: Path, identity: dict[str, Any], resume: bool) -> None:
    marker_path = output_dir / "mxwave-run.json"
    if resume and any(output_dir.iterdir()):
        if not marker_path.is_file():
            raise ValueError(
                "Cannot safely resume expert conversion: mxwave-run.json is missing"
            )
        try:
            recorded = json.loads(marker_path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Cannot safely resume expert conversion: invalid run marker"
            ) from exc
        if recorded != identity:
            raise ValueError(
                "Cannot safely resume expert conversion: source, layout, quantization "
                "settings, chunking, or cap changed"
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
    cfg: ExpertQuantizationConfig,
    plan: ExpertQuantizationPlan,
    sqnr_results: dict[str, float],
    passthrough_payload_hashes: dict[str, str],
    *,
    source_identity: dict[str, Any],
    run_identity_sha256: str,
    shard_integrity: dict[str, _ShardIntegrityRecord],
) -> dict[str, Any]:
    summary = plan.summary()
    values = list(sqnr_results.values())
    ordered_payload_hashes = sorted(passthrough_payload_hashes.items())
    aggregate_payload_hash = hashlib.sha256(
        json.dumps(ordered_payload_hashes, separators=(",", ":")).encode()
    ).hexdigest()
    passthrough_bytes = sum(
        source.projected_data_bytes for source in plan.tensors if not source.is_expert_bank
    )
    ordered_shard_integrity = {
        name: record.as_json() for name, record in sorted(shard_integrity.items())
    }
    raw_source_shards = source_identity.get("shards")
    if not isinstance(raw_source_shards, dict):
        raise TypeError("Run identity source shards must be an object")
    source_shards = cast(dict[str, dict[str, int | str]], raw_source_shards)
    summary.update(
        {
            "manifest_version": 1,
            "producer": {"name": "mxwave", "version": __version__},
            "source": source_identity,
            "method": "rtn" if cfg.method == "rtn" else "unweighted-mse",
            "experimental": True,
            "experimental_scope": (
                "routed-expert RTN baseline; no activation calibration"
                if cfg.method == "rtn"
                else (
                    "routed-expert unweighted-MSE scale-search candidate; "
                    "no activation calibration"
                )
            ),
            "weight_scale_selection": (
                "rtn-memoryless-minmax"
                if cfg.method == "rtn"
                else _mse_selection_name(
                    cfg.scale_percentile,
                    cfg.mse_clip_depth,
                )
            ),
            "tensor_row_chunk_size": cfg.tensor_row_chunk_size,
            "activation_quantization": "none",
            "activation_calibration": None,
            "gamma_proxy": None,
            "required_backends": ["linear=marlin", "moe=marlin"],
            "target_modules": plan.target_modules,
            "ignored_modules": plan.ignored_modules,
            "config_target_patterns": plan.config_target_patterns,
            "config_ignored_patterns": plan.config_ignored_patterns,
            "host_memory_contract": {
                "resident_tensor_cap_bytes": cfg.host_tensor_cap_bytes,
                "maximum_planned_unit_bytes": plan.maximum_unit_bytes,
                "scope": (
                    "materialized tensor payload only; allocator, serializer, mmap, and "
                    "filesystem cache overhead are not included"
                ),
            },
            "passthrough_payload_verification": {
                "algorithm": "sha256",
                "canonical_aggregate": "sha256(JSON(sorted([tensor_name,payload_sha256])))",
                "source_output_match": True,
                "tensor_count": len(ordered_payload_hashes),
                "data_bytes": passthrough_bytes,
                "aggregate_sha256": aggregate_payload_hash,
                "per_tensor_sha256": dict(ordered_payload_hashes),
            },
            "source_weight_shard_integrity": {
                "algorithm": "sha256",
                "scope": "source safetensors files; copied model assets are excluded",
                "canonical_aggregate": "sha256(canonical-json(per_shard))",
                "shard_count": len(source_shards),
                "file_bytes": sum(
                    cast(int, record["bytes"]) for record in source_shards.values()
                ),
                "aggregate_sha256": _canonical_sha256(source_shards),
                "per_shard": source_shards,
            },
            "shard_integrity": {
                "algorithm": "sha256",
                "sidecar": _INTEGRITY_SIDECAR_FILENAME,
                "run_identity_sha256": run_identity_sha256,
                "canonical_aggregate": "sha256(canonical-json(per_shard))",
                "complete": len(ordered_shard_integrity) == len(plan.units),
                "shard_count": len(ordered_shard_integrity),
                "file_bytes": sum(record.bytes for record in shard_integrity.values()),
                "aggregate_sha256": _canonical_sha256(ordered_shard_integrity),
                "per_shard": ordered_shard_integrity,
            },
            "fingerprint_excluded_metadata_assets": [
                "mxwave-run.json",
                _INTEGRITY_SIDECAR_FILENAME,
            ],
            "sqnr_sample_rows": cfg.sqnr_rows if cfg.verify_sqnr else None,
            "sqnr_db": (
                {
                    "count": len(values),
                    "coverage": len(values) / plan.layout.logical_matrix_count,
                    "minimum": min(values),
                    "mean": sum(values) / len(values),
                    "per_tensor": sqnr_results,
                }
                if values
                else None
            ),
            "calibration_weighted_sqnr_db": None,
            "gamma_weighted_sqnr_db": None,
        }
    )
    return summary


def _verify_complete_index(output_dir: Path, plan: ExpertQuantizationPlan) -> None:
    raw = json.loads((output_dir / "model.safetensors.index.json").read_text())
    if not isinstance(raw, dict) or not isinstance(raw.get("weight_map"), dict):
        raise TypeError("Emitted checkpoint index has no weight_map object")
    weight_map: dict[str, object] = raw["weight_map"]
    expected: dict[str, str] = {}
    for unit in plan.units:
        expected.update({name: unit.filename for name in unit.expected_specs()})
    if weight_map != expected:
        missing = sorted(set(expected).difference(weight_map))
        extra = sorted(set(weight_map).difference(expected))
        raise ValueError(
            f"Emitted checkpoint index coverage failed: missing={missing[:5]}, extra={extra[:5]}"
        )


def quantize_expert_model(cfg: ExpertQuantizationConfig) -> int:
    """Emit and structurally verify an expert-only, weight-only MXFP4 checkpoint."""
    plan = plan_expert_model(cfg)
    model_dir = Path(cfg.model_dir)
    output_dir = Path(cfg.output_dir)
    _validate_output_path(model_dir, output_dir, cfg.resume)
    if cfg.verbose:
        print(json.dumps(plan.summary(), indent=2))

    output_dir.mkdir(parents=True, exist_ok=True)
    expected_filenames = {unit.filename for unit in plan.units}
    if cfg.resume:
        unexpected = sorted(
            path.name
            for path in output_dir.glob("*.safetensors")
            if path.name not in expected_filenames
        )
        if unexpected:
            raise ValueError(f"Cannot safely resume with unexpected output shards: {unexpected[:5]}")
    run_identity = _run_identity(cfg, plan)
    run_identity_sha256 = _canonical_sha256(run_identity)
    _prepare_run_marker(output_dir, run_identity, cfg.resume)
    integrity_path, integrity_records = _prepare_integrity_records(
        output_dir,
        run_identity_sha256=run_identity_sha256,
        expected_filenames=expected_filenames,
        resume=cfg.resume,
    )

    shards_by_name = {shard.path.name: shard for shard in plan.shards}
    sqnr_results: dict[str, float] = {}
    passthrough_payload_hashes: dict[str, str] = {}
    output_shards: list[Path] = []
    for index, unit in enumerate(plan.units, start=1):
        output_path = output_dir / unit.filename
        expected = unit.expected_specs()
        reused = cfg.resume and output_path.exists()
        if reused:
            _validate_resumed_shard_integrity(output_path, integrity_records)
            _verify_output_shard(output_path, expected)
            if cfg.verify_sqnr:
                _sample_existing_unit_sqnr(
                    unit,
                    output_path,
                    cfg,
                    shards_by_name,
                    sqnr_results,
                )
            if cfg.verbose:
                print(f"[mxwave] [{index}/{len(plan.units)}] resume {unit.filename}")
        else:
            if output_path.exists():
                raise FileExistsError(f"Refusing to replace existing shard: {output_path}")
            if cfg.verbose:
                print(
                    f"[mxwave] [{index}/{len(plan.units)}] emit {unit.kind} {unit.filename}"
                )
            tensors = emit_expert_unit(
                unit,
                cfg,
                shards_by_name=shards_by_name,
                sqnr_results=sqnr_results if cfg.verify_sqnr else None,
            )
            _atomic_save_shard(tensors, output_path)
            del tensors
            _verify_output_shard(output_path, expected)
            if torch.device(cfg.device).type == "cuda":
                torch.cuda.empty_cache()
        _verify_passthrough_payloads(
            unit,
            output_path,
            shards_by_name,
            passthrough_payload_hashes,
        )
        if not reused:
            integrity_records[unit.filename] = _file_integrity(output_path)
            _atomic_write_integrity_sidecar(
                integrity_path,
                run_identity_sha256=run_identity_sha256,
                records=integrity_records,
            )
        output_shards.append(output_path)

    expected_passthrough = {
        source.info.name for source in plan.tensors if not source.is_expert_bank
    }
    if set(passthrough_payload_hashes) != expected_passthrough:
        missing = sorted(expected_passthrough.difference(passthrough_payload_hashes))
        extra = sorted(set(passthrough_payload_hashes).difference(expected_passthrough))
        raise AssertionError(
            f"Passthrough payload verification coverage failed: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
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
        passthrough_payload_hashes,
        source_identity=cast(dict[str, Any], run_identity["source"]),
        run_identity_sha256=run_identity_sha256,
        shard_integrity=integrity_records,
    )
    assemble_output_dir(
        model_dir,
        output_dir,
        output_shards,
        target_modules=plan.config_target_patterns,
        ignored_modules=plan.config_ignored_patterns,
        real_modules=plan.real_modules,
        manifest=manifest,
        weight_only=True,
    )
    gaps = verify_emitted_config(output_dir, plan.real_modules)
    if gaps:
        raise ValueError(f"Emitted expert config coverage failed: {gaps[:10]}")
    _verify_complete_index(output_dir, plan)
    return len(plan.units)
