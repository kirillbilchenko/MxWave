"""Logical matrix views for fused mixture-of-experts checkpoint tensors.

Some MoE checkpoints store every routed expert in two three-dimensional banks
instead of ordinary ``.weight`` matrices.  This module describes those banks
as deterministic two-dimensional logical matrices without teaching the core
MXFP4 quantizer about rank-3 tensors.

The IR is deliberately limited to layout and emitted-tensor contracts.  It
does not imply that calibration, conversion, or checkpoint emission supports a
given architecture; those stages must opt in separately after validation.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal

import torch

from .core import BLOCK_SIZE

type ExpertBankKind = str
type ExpertProjection = str

__all__ = [
    "EmittedTensorSpec",
    "ExpertBank",
    "ExpertBankKind",
    "ExpertProjection",
    "ExpertQuantizationLayout",
    "LogicalExpertMatrix",
]


@dataclass(frozen=True)
class EmittedTensorSpec:
    """One expected tensor in an unfolded MXFP4 checkpoint."""

    name: str
    shape: tuple[int, int]
    dtype: Literal["U8"] = "U8"

    def __post_init__(self) -> None:
        """Validate the emitted key and matrix shape."""
        if not self.name:
            raise ValueError("Emitted tensor name must be non-empty")
        if len(self.shape) != 2 or any(dimension <= 0 for dimension in self.shape):
            raise ValueError(f"Emitted tensor {self.name!r} has invalid shape {self.shape}")


@dataclass(frozen=True)
class ExpertBank:
    """Header-only description of one fused routed-expert source bank."""

    source_name: str
    layer_index: int
    kind: ExpertBankKind
    shape: tuple[int, int, int]
    dtype: str

    def __post_init__(self) -> None:
        """Validate bank identity and header metadata."""
        if not self.source_name:
            raise ValueError("Expert bank source name must be non-empty")
        if self.layer_index < 0:
            raise ValueError("Expert bank layer index must be non-negative")
        if not self.kind:
            raise ValueError("Expert bank kind must be non-empty")
        if len(self.shape) != 3 or any(dimension <= 0 for dimension in self.shape):
            raise ValueError(f"Expert bank {self.source_name!r} has invalid shape {self.shape}")


@dataclass(frozen=True)
class LogicalExpertMatrix:
    """A two-dimensional view into one expert of a fused source bank."""

    source_name: str
    source_shape: tuple[int, int, int]
    layer_index: int
    expert_index: int
    projection: ExpertProjection
    output_module: str
    row_start: int
    row_stop: int

    def __post_init__(self) -> None:
        """Validate source slicing and MXFP4 input-width invariants."""
        if len(self.source_shape) != 3 or any(dimension <= 0 for dimension in self.source_shape):
            raise ValueError(f"Logical matrix source has invalid shape {self.source_shape}")
        if self.layer_index < 0:
            raise ValueError("Logical matrix layer index must be non-negative")
        if not 0 <= self.expert_index < self.source_shape[0]:
            raise ValueError(
                f"Expert index {self.expert_index} is outside source shape {self.source_shape}"
            )
        if not self.projection:
            raise ValueError("Expert projection must be non-empty")
        if not self.output_module or self.output_module.endswith(".weight"):
            raise ValueError("Logical expert output must be a non-empty module name")
        if not 0 <= self.row_start < self.row_stop <= self.source_shape[1]:
            raise ValueError(
                f"Invalid row range [{self.row_start}, {self.row_stop}) for "
                f"source shape {self.source_shape}"
            )
        if self.shape[1] % BLOCK_SIZE != 0:
            raise ValueError(
                f"Logical matrix {self.output_module!r} input width {self.shape[1]} "
                f"is not divisible by MXFP4 block size {BLOCK_SIZE}"
            )

    @property
    def shape(self) -> tuple[int, int]:
        """Return the logical two-dimensional weight shape."""
        return (self.row_stop - self.row_start, self.source_shape[2])

    def view(self, source: torch.Tensor) -> torch.Tensor:
        """Return the logical 2D tensor as a zero-copy view of ``source``.

        The caller remains responsible for reading the fused source bank in a
        bounded manner.  No contiguous copy is made here, and the returned
        matrix can be passed directly to the existing 2D quantization core.
        """
        if tuple(source.shape) != self.source_shape:
            raise ValueError(
                f"Source tensor {self.source_name!r} has shape {tuple(source.shape)}, "
                f"expected {self.source_shape}"
            )
        matrix = source[self.expert_index, self.row_start : self.row_stop, :]
        if tuple(matrix.shape) != self.shape:
            raise AssertionError("Logical expert view shape disagrees with validated layout")
        return matrix

    def output_specs(self) -> tuple[EmittedTensorSpec, EmittedTensorSpec]:
        """Return packed-weight and E8M0-scale specs for this matrix."""
        rows, columns = self.shape
        return (
            EmittedTensorSpec(
                name=f"{self.output_module}.weight_packed",
                shape=(rows, columns // 2),
            ),
            EmittedTensorSpec(
                name=f"{self.output_module}.weight_scale",
                shape=(rows, columns // BLOCK_SIZE),
            ),
        )


@dataclass(frozen=True)
class ExpertQuantizationLayout:
    """Architecture-adapter output consumed by the generic expert engine.

    Architecture-specific tensor names, completeness rules, and runtime module
    selectors belong in the adapter. The engine consumes only this explicit
    bank/slice contract and never infers a model family from tensor suffixes.
    """

    architecture: str
    policy_name: str
    banks: tuple[ExpertBank, ...]
    matrices: tuple[LogicalExpertMatrix, ...]
    target_patterns: tuple[str, ...]
    ignored_patterns: tuple[str, ...] = ()
    metadata: tuple[tuple[str, int | str], ...] = ()

    def __post_init__(self) -> None:
        """Validate the adapter's complete, collision-free logical contract."""
        if not self.architecture or not self.policy_name:
            raise ValueError("Expert layout identity fields must be non-empty")
        if not self.banks or not self.matrices or not self.target_patterns:
            raise ValueError("Expert layout banks, matrices, and target patterns are required")
        for label, patterns in (
            ("target", self.target_patterns),
            ("ignored", self.ignored_patterns),
        ):
            if len(set(patterns)) != len(patterns):
                raise ValueError(f"Expert layout repeats a {label} selector")
            for pattern in patterns:
                if not pattern:
                    raise ValueError(f"Expert layout has an empty {label} selector")
                if pattern.startswith("re:"):
                    expression = pattern[3:]
                    if not expression:
                        raise ValueError(f"Expert layout has an empty {label} regex")
                    try:
                        re.compile(expression)
                    except re.error as exc:
                        raise ValueError(
                            f"Expert layout has invalid {label} regex {pattern!r}: {exc}"
                        ) from exc
        source_names = [bank.source_name for bank in self.banks]
        if len(set(source_names)) != len(source_names):
            raise ValueError("Expert layout repeats a source tensor name")
        banks_by_name = {bank.source_name: bank for bank in self.banks}
        output_modules: set[str] = set()
        covered_sources: set[str] = set()
        for matrix in self.matrices:
            bank = banks_by_name.get(matrix.source_name)
            if bank is None:
                raise ValueError(
                    f"Logical matrix references unknown source bank {matrix.source_name!r}"
                )
            if matrix.source_shape != bank.shape:
                raise ValueError(
                    f"Logical matrix source shape {matrix.source_shape} disagrees with "
                    f"bank {bank.source_name!r} shape {bank.shape}"
                )
            if matrix.output_module in output_modules:
                raise ValueError(
                    f"Expert layout repeats output module {matrix.output_module!r}"
                )
            output_modules.add(matrix.output_module)
            covered_sources.add(matrix.source_name)
        if covered_sources != set(source_names):
            missing = sorted(set(source_names).difference(covered_sources))
            raise ValueError(f"Expert layout has unused source banks: {missing[:5]}")
        if len(dict(self.metadata)) != len(self.metadata):
            raise ValueError("Expert layout metadata repeats a key")

    @property
    def source_bank_count(self) -> int:
        """Return the exact number of selected fused source tensors."""
        return len(self.banks)

    @property
    def logical_matrix_count(self) -> int:
        """Return the exact number of unfolded gate/up/down matrices."""
        return len(self.matrices)

    @property
    def output_tensor_count(self) -> int:
        """Return the exact packed-weight plus scale tensor count."""
        return self.logical_matrix_count * 2

    @property
    def source_names(self) -> frozenset[str]:
        """Return the fused source-bank names selected by this policy."""
        return frozenset(bank.source_name for bank in self.banks)

    def iter_logical_matrices(self) -> Iterator[LogicalExpertMatrix]:
        """Yield adapter-declared logical matrices in deterministic order."""
        yield from self.matrices

    def iter_output_specs(self) -> Iterator[EmittedTensorSpec]:
        """Yield every expected output tensor spec in deterministic order."""
        for matrix in self.iter_logical_matrices():
            yield from matrix.output_specs()

    def summary(self) -> dict[str, Any]:
        """Return a compact JSON-serializable header-plan summary."""
        return {
            "architecture": self.architecture,
            "policy": self.policy_name,
            "source_banks": self.source_bank_count,
            "logical_matrices": self.logical_matrix_count,
            "output_tensors": self.output_tensor_count,
            **dict(self.metadata),
        }
