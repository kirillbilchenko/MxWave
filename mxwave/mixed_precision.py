"""Bounded composition of standard mixed MXFP4/FP8 checkpoints.

The primary checkpoint supplies already packed MXFP4 tensors. Explicitly selected
runtime groups are replaced with channel-wise FP8 weights read from a compatible
dense donor. Shards are processed one at a time and selected donor matrices enter
the accelerator one at a time.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .output import build_quantization_config, copy_model_assets, verify_emitted_config
from .runtime_adapters import resolve_runtime_graph
from .shard import ShardFile, TensorInfo, discover_shards, read_tensor, shard_tensor_info

__all__ = [
    "Fp8CompositionPlan",
    "MixedPrecisionSource",
    "build_fp8_composition_plan",
    "compose_fp8_checkpoint",
    "inspect_mxfp4_checkpoint",
    "quantize_fp8_channelwise",
]

_FLOAT_DTYPES = frozenset({"BF16", "F16", "F32"})
_FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
_METRIC_FIELDS = (
    "sqnr_db",
    "calibration_weighted_sqnr_db",
    "gamma_weighted_sqnr_db",
    "feedback_selection_weighted_sqnr_db",
)


@dataclass(frozen=True)
class _Checkpoint:
    """Header-only inventory of a safetensors checkpoint."""

    root: Path
    shards: tuple[ShardFile, ...]
    info_by_key: Mapping[str, TensorInfo]
    shard_by_key: Mapping[str, ShardFile]
    config: Mapping[str, Any]


@dataclass(frozen=True)
class MixedPrecisionSource:
    """Public header-only view of an MXFP4 checkpoint."""

    root: Path
    config: Mapping[str, Any]
    tensor_names: tuple[str, ...]
    target_modules: tuple[str, ...]
    ignored_modules: tuple[str, ...]

    @property
    def real_modules(self) -> tuple[str, ...]:
        """Return every module covered by the compressed-tensors config."""
        return tuple(sorted(set(self.target_modules).union(self.ignored_modules)))


@dataclass(frozen=True)
class Fp8CompositionPlan:
    """Validated tensor and byte plan for one mixed-precision candidate."""

    primary: _Checkpoint
    dense: _Checkpoint
    selected_modules: tuple[str, ...]
    mxfp4_modules: tuple[str, ...]
    ignored_modules: tuple[str, ...]
    real_modules: tuple[str, ...]
    additions_by_shard: Mapping[str, tuple[str, ...]]
    removed_keys: frozenset[str]
    expected_by_shard: Mapping[str, Mapping[str, tuple[tuple[int, ...], str]]]
    primary_data_bytes: int
    replaced_mxfp4_bytes: int
    fp8_weight_bytes: int
    fp8_scale_bytes: int
    projected_output_data_bytes: int

    @property
    def premium_bytes(self) -> int:
        """Return the projected tensor-byte increase over the primary checkpoint."""
        return self.fp8_weight_bytes + self.fp8_scale_bytes - self.replaced_mxfp4_bytes

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serializable header-only summary."""
        return {
            "primary_model": str(self.primary.root),
            "dense_donor": str(self.dense.root),
            "replacement_format": "channel-wise-fp8-e4m3",
            "selected_modules": list(self.selected_modules),
            "selected_module_count": len(self.selected_modules),
            "mxfp4_module_count": len(self.mxfp4_modules),
            "ignored_module_count": len(self.ignored_modules),
            "shards": len(self.primary.shards),
            "primary_data_bytes": self.primary_data_bytes,
            "replaced_mxfp4_bytes": self.replaced_mxfp4_bytes,
            "fp8_weight_bytes": self.fp8_weight_bytes,
            "fp8_scale_bytes": self.fp8_scale_bytes,
            "premium_bytes": self.premium_bytes,
            "projected_output_data_bytes": self.projected_output_data_bytes,
        }


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON file is missing: {path}")
    value: Any = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return cast(dict[str, Any], value)


def _checkpoint(root: str | Path) -> _Checkpoint:
    checkpoint_root = Path(root)
    shards, weight_map = discover_shards(checkpoint_root)
    info_by_key: dict[str, TensorInfo] = {}
    shard_by_key: dict[str, ShardFile] = {}
    for shard in shards:
        for key, info in shard_tensor_info(shard).items():
            if key in info_by_key:
                raise ValueError(f"Duplicate tensor {key!r} in {checkpoint_root}")
            if weight_map is not None and weight_map.get(key) != shard.path.name:
                raise ValueError(
                    f"Index mismatch for {key!r}: expected {weight_map.get(key)!r}, "
                    f"found {shard.path.name!r}"
                )
            info_by_key[key] = info
            shard_by_key[key] = shard
    if weight_map is not None and set(weight_map) != set(info_by_key):
        missing = sorted(set(weight_map).difference(info_by_key))
        unindexed = sorted(set(info_by_key).difference(weight_map))
        raise ValueError(
            f"Index/header disagreement in {checkpoint_root}: "
            f"missing={missing[:5]}, unindexed={unindexed[:5]}"
        )
    return _Checkpoint(
        root=checkpoint_root,
        shards=tuple(shards),
        info_by_key=info_by_key,
        shard_by_key=shard_by_key,
        config=_read_json_object(checkpoint_root / "config.json"),
    )


def _base_config(config: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(config)
    result.pop("quantization_config", None)
    return result


def _config_modules(config: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    raw_quantization = config.get("quantization_config")
    if not isinstance(raw_quantization, dict):
        raise TypeError("Primary checkpoint has no quantization_config object")
    raw_groups = raw_quantization.get("config_groups")
    if not isinstance(raw_groups, dict):
        raise TypeError("Primary quantization_config has no config_groups object")

    targets: list[str] = []
    for raw_group in raw_groups.values():
        if not isinstance(raw_group, dict):
            raise TypeError("Quantization config group must be an object")
        raw_targets = raw_group.get("targets")
        if not isinstance(raw_targets, list) or not all(
            isinstance(item, str) for item in raw_targets
        ):
            raise TypeError("Quantization targets must be a list of strings")
        targets.extend(cast(list[str], raw_targets))
    raw_ignore = raw_quantization.get("ignore")
    if not isinstance(raw_ignore, list) or not all(isinstance(item, str) for item in raw_ignore):
        raise TypeError("Quantization ignore must be a list of strings")
    ignored = cast(list[str], raw_ignore)
    if any(value.startswith("re:") for value in (*targets, *ignored)):
        raise ValueError("Mixed-precision composition requires concrete module names")
    if len(targets) != len(set(targets)):
        raise ValueError("Primary quantization targets contain duplicates")
    overlap = sorted(set(targets).intersection(ignored))
    if overlap:
        raise ValueError(f"Primary targets and ignore list overlap: {overlap[:5]}")
    return tuple(sorted(targets)), tuple(sorted(set(ignored)))


def inspect_mxfp4_checkpoint(model_dir: str | Path) -> MixedPrecisionSource:
    """Inspect an MXFP4 checkpoint without loading tensor data."""
    checkpoint = _checkpoint(model_dir)
    targets, ignored = _config_modules(checkpoint.config)
    return MixedPrecisionSource(
        root=checkpoint.root,
        config=checkpoint.config,
        tensor_names=tuple(sorted(checkpoint.info_by_key)),
        target_modules=targets,
        ignored_modules=ignored,
    )


def _tensor_bytes(info: TensorInfo) -> int:
    return info.data_offsets[1] - info.data_offsets[0]


def _module_specs(
    module: str,
    primary: _Checkpoint,
    dense: _Checkpoint,
) -> tuple[TensorInfo, TensorInfo, TensorInfo]:
    packed_key = f"{module}.weight_packed"
    scale_key = f"{module}.weight_scale"
    dense_key = f"{module}.weight"
    try:
        packed = primary.info_by_key[packed_key]
        scale = primary.info_by_key[scale_key]
        dense_weight = dense.info_by_key[dense_key]
    except KeyError as exc:
        raise ValueError(f"Missing representation for selected module {module!r}") from exc
    if dense_weight.dtype not in _FLOAT_DTYPES or len(dense_weight.shape) != 2:
        raise ValueError(
            f"Dense donor tensor {dense_key!r} is {dense_weight.shape}/{dense_weight.dtype}, "
            "expected a floating matrix"
        )
    rows, columns = dense_weight.shape
    if columns % 32 != 0:
        raise ValueError(f"Dense donor width is not divisible by 32 for {module!r}")
    if packed.shape != (rows, columns // 2) or scale.shape != (rows, columns // 32):
        raise ValueError(
            f"Packed and dense shapes disagree for {module!r}: dense={dense_weight.shape}, "
            f"packed={packed.shape}, scale={scale.shape}"
        )
    if packed.dtype != "U8" or scale.dtype != "U8":
        raise ValueError(f"Selected module {module!r} is not an MXFP4 U8 representation")
    if primary.shard_by_key[packed_key].path.name != primary.shard_by_key[scale_key].path.name:
        raise ValueError(f"Packed tensors for {module!r} are split across shards")
    return packed, scale, dense_weight


def _validate_runtime_groups(
    config: Mapping[str, Any],
    dense_tensor_names: Sequence[str],
    target_modules: frozenset[str],
    selected_modules: frozenset[str],
) -> None:
    try:
        graph = resolve_runtime_graph(config, dense_tensor_names)
    except ValueError as exc:
        if "No runtime-operation adapter matches" in str(exc):
            raise ValueError(
                "Mixed-precision composition requires a runtime adapter so fused groups "
                "cannot be split"
            ) from exc
        raise
    for operation in graph.operations:
        for group in operation.linear_groups:
            members = {
                member.checkpoint_name.removesuffix(".weight")
                for member in group.members
                if member.checkpoint_name.removesuffix(".weight") in target_modules
            }
            selected = members.intersection(selected_modules)
            if selected and selected != members:
                raise ValueError(
                    f"Selection splits fused runtime group {group.runtime_name!r}: "
                    f"selected={sorted(selected)}, required={sorted(members)}"
                )


def build_fp8_composition_plan(
    primary_model: str | Path,
    dense_donor: str | Path,
    selected_modules: Sequence[str],
) -> Fp8CompositionPlan:
    """Validate and size one selective FP8 candidate without loading tensors."""
    if not selected_modules:
        raise ValueError("At least one module must be selected for FP8")
    if len(selected_modules) != len(set(selected_modules)):
        raise ValueError("Selected FP8 modules contain duplicates")
    primary = _checkpoint(primary_model)
    dense = _checkpoint(dense_donor)
    if _base_config(primary.config) != _base_config(dense.config):
        raise ValueError("Primary and dense-donor base model configs differ")

    target_modules, ignored_modules = _config_modules(primary.config)
    target_set = frozenset(target_modules)
    selected_set = frozenset(selected_modules)
    unknown = sorted(selected_set.difference(target_set))
    if unknown:
        raise ValueError(f"Selected modules are not MXFP4 targets: {unknown[:5]}")
    _validate_runtime_groups(
        primary.config,
        tuple(dense.info_by_key),
        target_set,
        selected_set,
    )

    removed_keys: set[str] = set()
    additions: dict[str, list[str]] = {shard.path.name: [] for shard in primary.shards}
    replaced_mxfp4_bytes = 0
    fp8_weight_bytes = 0
    fp8_scale_bytes = 0
    for module in sorted(selected_set):
        packed, scale, dense_weight = _module_specs(module, primary, dense)
        packed_key = f"{module}.weight_packed"
        scale_key = f"{module}.weight_scale"
        removed_keys.update((packed_key, scale_key))
        shard_name = primary.shard_by_key[packed_key].path.name
        additions[shard_name].append(module)
        rows, columns = dense_weight.shape
        replaced_mxfp4_bytes += _tensor_bytes(packed) + _tensor_bytes(scale)
        fp8_weight_bytes += rows * columns
        fp8_scale_bytes += rows * 4

    expected_by_shard: dict[str, dict[str, tuple[tuple[int, ...], str]]] = {}
    for shard in primary.shards:
        expected: dict[str, tuple[tuple[int, ...], str]] = {}
        for key, info in primary.info_by_key.items():
            if primary.shard_by_key[key].path.name == shard.path.name and key not in removed_keys:
                expected[key] = (info.shape, info.dtype)
        for module in additions[shard.path.name]:
            dense_info = dense.info_by_key[f"{module}.weight"]
            rows, _columns = dense_info.shape
            expected[f"{module}.weight"] = (dense_info.shape, "F8_E4M3")
            expected[f"{module}.weight_scale"] = ((rows, 1), "F32")
        expected_by_shard[shard.path.name] = expected

    primary_data_bytes = sum(_tensor_bytes(info) for info in primary.info_by_key.values())
    projected = primary_data_bytes - replaced_mxfp4_bytes + fp8_weight_bytes + fp8_scale_bytes
    remaining = tuple(sorted(target_set.difference(selected_set)))
    real_modules = tuple(sorted(target_set.union(ignored_modules)))
    return Fp8CompositionPlan(
        primary=primary,
        dense=dense,
        selected_modules=tuple(sorted(selected_set)),
        mxfp4_modules=remaining,
        ignored_modules=ignored_modules,
        real_modules=real_modules,
        additions_by_shard={key: tuple(value) for key, value in additions.items()},
        removed_keys=frozenset(removed_keys),
        expected_by_shard=expected_by_shard,
        primary_data_bytes=primary_data_bytes,
        replaced_mxfp4_bytes=replaced_mxfp4_bytes,
        fp8_weight_bytes=fp8_weight_bytes,
        fp8_scale_bytes=fp8_scale_bytes,
        projected_output_data_bytes=projected,
    )


def quantize_fp8_channelwise(
    weight: torch.Tensor,
    *,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FP8 E4M3 weights and one FP32 scale per output channel."""
    if weight.ndim != 2 or not weight.is_floating_point():
        raise ValueError("FP8 composition requires a floating two-dimensional weight")
    quant_device = torch.device(device) if device is not None else weight.device
    values = weight.to(device=quant_device, dtype=torch.float32)
    if not torch.isfinite(values).all():
        raise ValueError("Dense donor weight contains NaN or infinity")
    scale = values.abs().amax(dim=1, keepdim=True) / _FP8_MAX
    scale = torch.where(scale == 0, torch.finfo(torch.float32).tiny, scale)
    quantized = torch.clamp(values / scale, -_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    return quantized.cpu().contiguous(), scale.cpu().contiguous()


def _atomic_save_shard(tensors: Mapping[str, torch.Tensor], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.incomplete")
    temporary.unlink(missing_ok=True)
    try:
        save_file(dict(tensors), str(temporary))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.incomplete")
    temporary.unlink(missing_ok=True)
    try:
        temporary.write_text(value)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _verify_shard(
    path: Path,
    expected: Mapping[str, tuple[tuple[int, ...], str]],
) -> dict[str, TensorInfo]:
    actual = shard_tensor_info(ShardFile(path=path, weight_map={}))
    if set(actual) != set(expected):
        missing = sorted(set(expected).difference(actual))
        extra = sorted(set(actual).difference(expected))
        raise ValueError(
            f"Composed shard {path.name} has wrong keys: missing={missing[:5]}, extra={extra[:5]}"
        )
    for key, (shape, dtype) in expected.items():
        if actual[key].shape != shape or actual[key].dtype != dtype:
            raise ValueError(
                f"Composed tensor {key!r} is {actual[key].shape}/{actual[key].dtype}, "
                f"expected {shape}/{dtype}"
            )
    return actual


def _fp8_group(target_modules: Sequence[str]) -> dict[str, Any]:
    return {
        "targets": sorted(set(target_modules)),
        "weights": {
            "num_bits": 8,
            "type": "float",
            "symmetric": True,
            "group_size": None,
            "strategy": "channel",
            "block_structure": None,
            "dynamic": False,
            "actorder": None,
            "scale_dtype": None,
            "zp_dtype": None,
            "observer": "memoryless_minmax",
            "observer_kwargs": {},
        },
        "input_activations": None,
        "output_activations": None,
        "format": "float-quantized",
    }


def _mixed_quantization_config(
    plan: Fp8CompositionPlan,
    *,
    compression_ratio: float | None,
) -> dict[str, Any]:
    config = build_quantization_config(
        list(plan.mxfp4_modules),
        list(plan.ignored_modules),
        compression_ratio=compression_ratio,
    )
    raw_groups = config["config_groups"]
    if not isinstance(raw_groups, dict):
        raise TypeError("Generated quantization config groups must be an object")
    groups = cast(dict[str, Any], raw_groups)
    raw_mxfp4 = groups.get("group_0")
    if not isinstance(raw_mxfp4, dict):
        raise TypeError("Generated MXFP4 config group must be an object")
    cast(dict[str, Any], raw_mxfp4)["format"] = "mxfp4-pack-quantized"
    groups["group_1"] = _fp8_group(plan.selected_modules)
    config["format"] = "mixed-precision"
    return config


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _manifest_path(root: Path) -> Path:
    for name in ("mxwave-manifest.json", "mxstream-manifest.json"):
        path = root / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"Primary checkpoint has no MxWave manifest: {root}")


def _filter_metric(value: Any, selected: frozenset[str]) -> Any:
    if not isinstance(value, dict):
        return value
    raw_per_tensor = value.get("per_tensor")
    if not isinstance(raw_per_tensor, dict):
        return value
    per_tensor = {
        name: score
        for name, score in raw_per_tensor.items()
        if isinstance(name, str) and name.removesuffix(".weight") not in selected
    }
    scores = [float(score) for score in per_tensor.values() if isinstance(score, int | float)]
    result = dict(value)
    result["per_tensor"] = per_tensor
    result["count"] = len(scores)
    result["coverage"] = 1.0 if scores else 0.0
    result["mean"] = sum(scores) / len(scores) if scores else None
    result["minimum"] = min(scores) if scores else None
    return result


def _build_manifest(
    plan: Fp8CompositionPlan,
    *,
    output_bytes: int,
    copied_assets: list[str],
) -> dict[str, Any]:
    manifest_path = _manifest_path(plan.primary.root)
    manifest = _read_json_object(manifest_path)
    selected = frozenset(plan.selected_modules)
    for field in _METRIC_FIELDS:
        if field in manifest:
            manifest[field] = _filter_metric(manifest[field], selected)
    activation = manifest.get("activation_calibration")
    if isinstance(activation, dict):
        updated_activation = dict(activation)
        updated_activation["weighted_tensors"] = len(plan.mxfp4_modules)
        manifest["activation_calibration"] = updated_activation
    source_data_bytes = manifest.get("source_data_bytes")
    ratio = (
        round(float(source_data_bytes) / output_bytes, 4)
        if isinstance(source_data_bytes, int | float) and output_bytes > 0
        else None
    )
    manifest.update(
        {
            "policy": "measured-precision-budget-candidate",
            "policy_description": (
                "H64 MXFP4 with explicitly selected channel-wise FP8 runtime groups"
            ),
            "target_tensors": len(plan.mxfp4_modules) + len(plan.selected_modules),
            "mxfp4_target_tensors": len(plan.mxfp4_modules),
            "fp8_target_tensors": len(plan.selected_modules),
            "actual_output_bytes": output_bytes,
            "global_compression_ratio": ratio,
            "projected_output_data_bytes": plan.projected_output_data_bytes,
            "mxfp4_target_modules": list(plan.mxfp4_modules),
            "fp8_target_modules": list(plan.selected_modules),
            "target_modules": sorted((*plan.mxfp4_modules, *plan.selected_modules)),
            "ignored_modules": list(plan.ignored_modules),
            "copied_assets": copied_assets,
            "composition": {
                "kind": "selective-channel-fp8-from-mxfp4",
                "primary_manifest_sha256": _file_sha256(manifest_path),
                "selected_modules": list(plan.selected_modules),
                "replaced_mxfp4_bytes": plan.replaced_mxfp4_bytes,
                "fp8_weight_bytes": plan.fp8_weight_bytes,
                "fp8_scale_bytes": plan.fp8_scale_bytes,
                "premium_bytes": plan.premium_bytes,
            },
        }
    )
    manifest.pop("runtime_validation", None)
    return manifest


def _temporary_model_card(manifest: Mapping[str, Any]) -> str:
    premium = manifest.get("composition", {})
    premium_bytes = premium.get("premium_bytes") if isinstance(premium, dict) else None
    return "\n".join(
        [
            "# MxWave measured-precision candidate",
            "",
            "This is a temporary research artifact. Do not publish it before whole-model ",
            "quality and runtime gates pass.",
            "",
            f"- MXFP4 modules: {manifest.get('mxfp4_target_tensors', 'unknown')}",
            f"- FP8 modules: {manifest.get('fp8_target_tensors', 'unknown')}",
            f"- Added tensor bytes: {premium_bytes}",
            "",
        ]
    )


def compose_fp8_checkpoint(
    plan: Fp8CompositionPlan,
    output_dir: str | Path,
    *,
    quant_device: torch.device | str = "cpu",
    verbose: bool = True,
) -> dict[str, Any]:
    """Execute a validated mixed-precision plan and return its manifest."""
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    output_shards: list[Path] = []
    output_weight_map: dict[str, str] = {}
    total_tensor_bytes = 0
    for index, shard in enumerate(plan.primary.shards, start=1):
        if verbose:
            print(
                f"[mxwave] [{index}/{len(plan.primary.shards)}] compose FP8 {shard.path.name}",
                flush=True,
            )
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(str(shard.path), framework="pt", device="cpu") as source:
            for key in source.keys():  # noqa: SIM118
                if key not in plan.removed_keys:
                    tensors[key] = cast(torch.Tensor, source.get_tensor(key))
            for module in plan.additions_by_shard[shard.path.name]:
                dense_key = f"{module}.weight"
                donor_shard = plan.dense.shard_by_key[dense_key]
                dense_weight = read_tensor(donor_shard, dense_key, device=quant_device)
                fp8_weight, fp8_scale = quantize_fp8_channelwise(
                    dense_weight,
                    device=quant_device,
                )
                tensors[dense_key] = fp8_weight
                tensors[f"{module}.weight_scale"] = fp8_scale
                del dense_weight, fp8_weight, fp8_scale
            output_path = output / shard.path.name
            _atomic_save_shard(tensors, output_path)
        del tensors
        actual = _verify_shard(output_path, plan.expected_by_shard[shard.path.name])
        for key, info in actual.items():
            if key in output_weight_map:
                raise ValueError(f"Duplicate composed tensor key: {key}")
            output_weight_map[key] = output_path.name
            total_tensor_bytes += _tensor_bytes(info)
        output_shards.append(output_path)

    if total_tensor_bytes != plan.projected_output_data_bytes:
        raise ValueError(
            f"Composed tensor bytes are {total_tensor_bytes}, "
            f"expected {plan.projected_output_data_bytes}"
        )
    copied_assets = copy_model_assets(plan.primary.root, output)
    output_bytes = sum(path.stat().st_size for path in output_shards)
    manifest = _build_manifest(plan, output_bytes=output_bytes, copied_assets=copied_assets)
    source_data_bytes = manifest.get("source_data_bytes")
    compression_ratio = (
        round(float(source_data_bytes) / output_bytes, 4)
        if isinstance(source_data_bytes, int | float) and output_bytes > 0
        else None
    )
    config = dict(plan.primary.config)
    config["quantization_config"] = _mixed_quantization_config(
        plan,
        compression_ratio=compression_ratio,
    )
    index_document = {
        "metadata": {"total_size": total_tensor_bytes},
        "weight_map": output_weight_map,
    }
    _atomic_write_text(output / "config.json", json.dumps(config, indent=2) + "\n")
    _atomic_write_text(
        output / "model.safetensors.index.json",
        json.dumps(index_document, indent=2) + "\n",
    )
    _atomic_write_text(
        output / "mxwave-manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    _atomic_write_text(output / "README.md", _temporary_model_card(manifest))

    gaps = verify_emitted_config(output, list(plan.real_modules))
    if gaps:
        raise ValueError(f"Composed config coverage failed: {gaps[:5]}")
    output_keys = set(output_weight_map)
    for module in plan.selected_modules:
        if (
            f"{module}.weight" not in output_keys
            or f"{module}.weight_scale" not in output_keys
            or f"{module}.weight_packed" in output_keys
        ):
            raise ValueError(f"FP8 replacement verification failed for {module!r}")
    packed_modules = {
        key.removesuffix(".weight_packed") for key in output_keys if key.endswith(".weight_packed")
    }
    if packed_modules != set(plan.mxfp4_modules):
        raise ValueError("Composed packed tensors and MXFP4 config targets disagree")
    return manifest
