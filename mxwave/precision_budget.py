"""Plan bounded, fusion-safe mixed-precision intervention buckets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .mixed_precision import (
    Fp8CompositionPlan,
    build_fp8_composition_plan,
    inspect_mxfp4_checkpoint,
)
from .runtime_adapters import resolve_runtime_graph
from .runtime_ir import RuntimeLinearGroup, RuntimeOperation
from .shard import discover_shards, shard_tensor_info

PrecisionFamily = Literal[
    "mlp-input",
    "mlp-output",
    "sequence-input",
    "sequence-output",
]

__all__ = [
    "PrecisionBucket",
    "PrecisionBudgetPlan",
    "build_precision_budget_plan",
    "load_bucket_modules",
]

_FAMILY_ORDER: tuple[PrecisionFamily, ...] = (
    "sequence-output",
    "mlp-output",
    "mlp-input",
    "sequence-input",
)
_BAND_NAMES = ("early", "middle", "late")


@dataclass(frozen=True)
class PrecisionBucket:
    """One directly deployable mixed-precision intervention."""

    name: str
    family: PrecisionFamily
    band: int
    layer_start: int
    layer_stop: int
    selected_modules: tuple[str, ...]
    selected_runtime_groups: tuple[str, ...]
    premium_bytes: int
    projected_output_data_bytes: int

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable bucket record."""
        return {
            "name": self.name,
            "family": self.family,
            "band": self.band,
            "layer_start": self.layer_start,
            "layer_stop": self.layer_stop,
            "selected_modules": list(self.selected_modules),
            "selected_runtime_groups": list(self.selected_runtime_groups),
            "premium_bytes": self.premium_bytes,
            "projected_output_data_bytes": self.projected_output_data_bytes,
        }


@dataclass(frozen=True)
class PrecisionBudgetPlan:
    """Pre-registered set of bounded precision interventions."""

    primary_model: str
    dense_donor: str
    layer_count: int
    bands: int
    layers_per_bucket: int
    max_candidates: int
    max_premium_bytes: int
    buckets: tuple[PrecisionBucket, ...]

    def as_dict(self) -> dict[str, Any]:
        """Return the complete reproducible plan."""
        body: dict[str, Any] = {
            "format": "mxwave-precision-budget-plan-v1",
            "primary_model": self.primary_model,
            "dense_donor": self.dense_donor,
            "layer_count": self.layer_count,
            "bands": self.bands,
            "layers_per_bucket": self.layers_per_bucket,
            "max_candidates": self.max_candidates,
            "max_premium_bytes": self.max_premium_bytes,
            "families": list(_FAMILY_ORDER),
            "buckets": [bucket.as_dict() for bucket in self.buckets],
        }
        canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        body["plan_sha256"] = hashlib.sha256(canonical).hexdigest()
        return body


def _dense_tensor_names(model_dir: str | Path) -> tuple[str, ...]:
    shards, _weight_map = discover_shards(model_dir)
    names: list[str] = []
    for shard in shards:
        names.extend(shard_tensor_info(shard))
    return tuple(sorted(names))


def _family(operation: RuntimeOperation, group: RuntimeLinearGroup) -> PrecisionFamily:
    if operation.kind == "gated-mlp":
        return (
            "mlp-output" if any(member.role == "down" for member in group.members) else "mlp-input"
        )
    return (
        "sequence-output"
        if len(group.members) == 1 and group.members[0].role == "output"
        else "sequence-input"
    )


def _representative_window(
    band: int,
    *,
    layer_count: int,
    bands: int,
    layers_per_bucket: int,
) -> tuple[int, int]:
    band_start = band * layer_count // bands
    band_stop = (band + 1) * layer_count // bands
    width = min(layers_per_bucket, band_stop - band_start)
    midpoint = (band_start + band_stop) // 2
    start = max(band_start, midpoint - width // 2)
    stop = start + width
    if stop > band_stop:
        stop = band_stop
        start = stop - width
    return start, stop


def _bucket_from_composition(
    name: str,
    family: PrecisionFamily,
    band: int,
    layer_start: int,
    layer_stop: int,
    modules: set[str],
    runtime_groups: set[str],
    composition: Fp8CompositionPlan,
) -> PrecisionBucket:
    return PrecisionBucket(
        name=name,
        family=family,
        band=band,
        layer_start=layer_start,
        layer_stop=layer_stop,
        selected_modules=tuple(sorted(modules)),
        selected_runtime_groups=tuple(sorted(runtime_groups)),
        premium_bytes=composition.premium_bytes,
        projected_output_data_bytes=composition.projected_output_data_bytes,
    )


def build_precision_budget_plan(
    primary_model: str | Path,
    dense_donor: str | Path,
    *,
    bands: int = 3,
    layers_per_bucket: int = 4,
    max_candidates: int = 12,
    max_premium_bytes: int = 1024**3,
) -> PrecisionBudgetPlan:
    """Build representative operator-family buckets and validate every candidate."""
    if bands <= 0:
        raise ValueError("bands must be positive")
    if layers_per_bucket <= 0:
        raise ValueError("layers_per_bucket must be positive")
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    if max_premium_bytes <= 0:
        raise ValueError("max_premium_bytes must be positive")
    if bands != len(_BAND_NAMES):
        raise ValueError("The bounded v1 protocol requires exactly three bands")

    source = inspect_mxfp4_checkpoint(primary_model)
    graph = resolve_runtime_graph(source.config, _dense_tensor_names(dense_donor))
    layer_count = max(operation.layer_index for operation in graph.operations) + 1
    target_modules = frozenset(source.target_modules)
    buckets: list[PrecisionBucket] = []
    for family in _FAMILY_ORDER:
        for band in range(bands):
            layer_start, layer_stop = _representative_window(
                band,
                layer_count=layer_count,
                bands=bands,
                layers_per_bucket=layers_per_bucket,
            )
            modules: set[str] = set()
            runtime_groups: set[str] = set()
            for operation in graph.operations:
                if not layer_start <= operation.layer_index < layer_stop:
                    continue
                for group in operation.linear_groups:
                    if _family(operation, group) != family:
                        continue
                    selected = {
                        member.checkpoint_name.removesuffix(".weight")
                        for member in group.members
                        if member.checkpoint_name.removesuffix(".weight") in target_modules
                    }
                    if selected:
                        modules.update(selected)
                        runtime_groups.add(group.runtime_name)
            if not modules:
                raise ValueError(
                    f"No MXFP4 targets found for family={family} band={band} "
                    f"layers=[{layer_start}, {layer_stop})"
                )
            name = f"{family}-{_BAND_NAMES[band]}-l{layer_start:02d}-{layer_stop - 1:02d}"
            composition = build_fp8_composition_plan(
                primary_model,
                dense_donor,
                sorted(modules),
            )
            if composition.premium_bytes > max_premium_bytes:
                raise ValueError(
                    f"Bucket {name!r} premium {composition.premium_bytes} exceeds "
                    f"the fixed cap {max_premium_bytes}"
                )
            buckets.append(
                _bucket_from_composition(
                    name,
                    family,
                    band,
                    layer_start,
                    layer_stop,
                    modules,
                    runtime_groups,
                    composition,
                )
            )
    if len(buckets) > max_candidates:
        raise ValueError(f"Protocol produced {len(buckets)} candidates, above cap {max_candidates}")
    return PrecisionBudgetPlan(
        primary_model=str(Path(primary_model)),
        dense_donor=str(Path(dense_donor)),
        layer_count=layer_count,
        bands=bands,
        layers_per_bucket=layers_per_bucket,
        max_candidates=max_candidates,
        max_premium_bytes=max_premium_bytes,
        buckets=tuple(buckets),
    )


def load_bucket_modules(path: str | Path, bucket_name: str) -> tuple[str, ...]:
    """Load one named selection from a serialized precision-budget plan."""
    value: Any = json.loads(Path(path).read_text())
    if not isinstance(value, dict) or value.get("format") != "mxwave-precision-budget-plan-v1":
        raise ValueError("Unsupported precision-budget plan")
    raw_buckets = value.get("buckets")
    if not isinstance(raw_buckets, list):
        raise TypeError("Precision-budget plan has no bucket list")
    for raw_bucket in raw_buckets:
        if not isinstance(raw_bucket, dict) or raw_bucket.get("name") != bucket_name:
            continue
        raw_modules = raw_bucket.get("selected_modules")
        if (
            not isinstance(raw_modules, list)
            or not raw_modules
            or not all(isinstance(module, str) for module in raw_modules)
        ):
            raise ValueError(f"Bucket {bucket_name!r} has an invalid module list")
        return tuple(raw_modules)
    raise ValueError(f"Precision-budget plan has no bucket {bucket_name!r}")
