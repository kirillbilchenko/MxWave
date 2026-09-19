"""Model-independent description of quantization-relevant runtime operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

OperationKind = Literal["gated-mlp", "self-attention", "gated-delta"]
ResponsePointKind = Literal["activation", "distribution", "state"]

__all__ = [
    "OperationKind",
    "ResponsePoint",
    "ResponsePointKind",
    "RuntimeGraph",
    "RuntimeLinearGroup",
    "RuntimeOperation",
    "RuntimeOutputPath",
    "RuntimeWeight",
]


@dataclass(frozen=True)
class RuntimeWeight:
    """One checkpoint weight and its semantic role in a runtime linear."""

    checkpoint_name: str
    role: str

    def __post_init__(self) -> None:
        """Validate a checkpoint-weight reference."""
        if not self.checkpoint_name.endswith(".weight"):
            raise ValueError(
                f"Runtime weight must name a checkpoint .weight tensor: {self.checkpoint_name!r}"
            )
        if not self.role:
            raise ValueError("Runtime weight role must be non-empty")

    def as_dict(self) -> dict[str, str]:
        """Return a JSON-serializable member record."""
        return {"checkpoint_name": self.checkpoint_name, "role": self.role}


@dataclass(frozen=True)
class RuntimeLinearGroup:
    """Source weights loaded into one logical runtime linear operation.

    Member order records the runtime packing order. A group with more than one
    member may require a shared tensor-level scale in formats such as NVFP4.
    The IR records the fusion fact without imposing a particular number format.
    """

    runtime_name: str
    members: tuple[RuntimeWeight, ...]

    def __post_init__(self) -> None:
        """Validate group identity, member order, and uniqueness."""
        if not self.runtime_name or self.runtime_name.endswith(".weight"):
            raise ValueError("Runtime linear name must be a non-empty module name")
        if not self.members:
            raise ValueError(f"Runtime linear {self.runtime_name!r} has no checkpoint members")
        member_names = [member.checkpoint_name for member in self.members]
        if len(set(member_names)) != len(member_names):
            raise ValueError(f"Runtime linear {self.runtime_name!r} repeats a checkpoint weight")
        roles = [member.role for member in self.members]
        if len(set(roles)) != len(roles):
            raise ValueError(f"Runtime linear {self.runtime_name!r} repeats a member role")

    @property
    def is_fused(self) -> bool:
        """Return whether multiple source weights form this runtime linear."""
        return len(self.members) > 1

    @property
    def requires_shared_tensor_scale(self) -> bool:
        """Return whether tensor-wide formats must share a scale across members."""
        return self.is_fused

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable fusion-group record."""
        return {
            "runtime_name": self.runtime_name,
            "members": [member.as_dict() for member in self.members],
            "requires_shared_tensor_scale": self.requires_shared_tensor_scale,
        }


@dataclass(frozen=True)
class ResponsePoint:
    """Named tensor boundary that an operation replay may expose for scoring."""

    name: str
    kind: ResponsePointKind = "activation"

    def __post_init__(self) -> None:
        """Validate a response-point descriptor."""
        if not self.name:
            raise ValueError("Response point name must be non-empty")
        if self.kind not in ("activation", "distribution", "state"):
            raise ValueError(f"Unsupported response point kind: {self.kind}")

    def as_dict(self) -> dict[str, str]:
        """Return a JSON-serializable response-point record."""
        return {"name": self.name, "kind": self.kind}


@dataclass(frozen=True)
class RuntimeOperation:
    """One nonlinear or stateful operation used as a quantization boundary."""

    name: str
    layer_index: int
    kind: OperationKind
    linear_groups: tuple[RuntimeLinearGroup, ...]
    response_points: tuple[ResponsePoint, ...]

    def __post_init__(self) -> None:
        """Validate group ownership and observable outputs."""
        if not self.name:
            raise ValueError("Runtime operation name must be non-empty")
        if self.layer_index < 0:
            raise ValueError("Runtime operation layer index must be non-negative")
        if self.kind not in ("gated-mlp", "self-attention", "gated-delta"):
            raise ValueError(f"Unsupported runtime operation kind: {self.kind}")
        if not self.linear_groups:
            raise ValueError(f"Runtime operation {self.name!r} has no linear groups")
        if not self.response_points:
            raise ValueError(f"Runtime operation {self.name!r} has no response points")

        group_names = [group.runtime_name for group in self.linear_groups]
        if len(set(group_names)) != len(group_names):
            raise ValueError(f"Runtime operation {self.name!r} repeats a linear group")
        checkpoint_names = [
            member.checkpoint_name
            for group in self.linear_groups
            for member in group.members
        ]
        if len(set(checkpoint_names)) != len(checkpoint_names):
            raise ValueError(f"Runtime operation {self.name!r} reuses a checkpoint weight")
        point_names = [point.name for point in self.response_points]
        if len(set(point_names)) != len(point_names):
            raise ValueError(f"Runtime operation {self.name!r} repeats a response point")

    @property
    def checkpoint_names(self) -> tuple[str, ...]:
        """Return checkpoint weights in runtime group and packing order."""
        return tuple(
            member.checkpoint_name
            for group in self.linear_groups
            for member in group.members
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable operation record."""
        return {
            "name": self.name,
            "layer_index": self.layer_index,
            "kind": self.kind,
            "linear_groups": [group.as_dict() for group in self.linear_groups],
            "response_points": [point.as_dict() for point in self.response_points],
        }


@dataclass(frozen=True)
class RuntimeOutputPath:
    """Modules that map final decoder hidden states to vocabulary logits."""

    normalization_module: str
    projection_module: str

    def __post_init__(self) -> None:
        """Validate final normalization and projection module names."""
        if not self.normalization_module or self.normalization_module.endswith(".weight"):
            raise ValueError("Runtime output normalization must name a module")
        if not self.projection_module or self.projection_module.endswith(".weight"):
            raise ValueError("Runtime output projection must name a module")

    def as_dict(self) -> dict[str, str]:
        """Return a JSON-serializable output-path record."""
        return {
            "normalization_module": self.normalization_module,
            "projection_module": self.projection_module,
        }


@dataclass(frozen=True)
class RuntimeGraph:
    """Validated runtime-operation graph emitted by an architecture adapter."""

    architecture: str
    adapter_version: str
    operations: tuple[RuntimeOperation, ...]
    output_path: RuntimeOutputPath | None = None

    def __post_init__(self) -> None:
        """Validate graph-wide operation and checkpoint ownership."""
        if not self.architecture:
            raise ValueError("Runtime graph architecture must be non-empty")
        if not self.adapter_version:
            raise ValueError("Runtime graph adapter version must be non-empty")
        if not self.operations:
            raise ValueError("Runtime graph must contain at least one operation")
        operation_names = [operation.name for operation in self.operations]
        if len(set(operation_names)) != len(operation_names):
            raise ValueError("Runtime graph repeats an operation name")

        owners: dict[str, str] = {}
        for operation in self.operations:
            for checkpoint_name in operation.checkpoint_names:
                previous = owners.setdefault(checkpoint_name, operation.name)
                if previous != operation.name:
                    raise ValueError(
                        f"Checkpoint weight {checkpoint_name!r} belongs to both "
                        f"{previous!r} and {operation.name!r}"
                    )

    def operation_for_weight(self, checkpoint_name: str) -> RuntimeOperation:
        """Return the unique operation owning ``checkpoint_name``."""
        for operation in self.operations:
            if checkpoint_name in operation.checkpoint_names:
                return operation
        raise KeyError(f"No runtime operation owns checkpoint weight {checkpoint_name!r}")

    def linear_group_for_weight(self, checkpoint_name: str) -> RuntimeLinearGroup:
        """Return the unique runtime linear containing ``checkpoint_name``."""
        operation = self.operation_for_weight(checkpoint_name)
        for group in operation.linear_groups:
            if checkpoint_name in (member.checkpoint_name for member in group.members):
                return group
        raise AssertionError("Runtime graph ownership index is inconsistent")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable graph record."""
        return {
            "architecture": self.architecture,
            "adapter_version": self.adapter_version,
            "output_path": self.output_path.as_dict() if self.output_path is not None else None,
            "operations": [operation.as_dict() for operation in self.operations],
        }
