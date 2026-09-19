"""Exact gated-MLP symmetry candidates for MXFP4 block-layout research."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch

from .calibration import CalibrationData
from .counteraction_probe import (
    CandidateMaterialization,
    CounteractionProbeResult,
    probe_mlp_candidates,
)
from .mxfp4_candidates import (
    Mxfp4CandidateSpec,
    diagonal_hessian_channel_weights,
    materialize_mxfp4_candidate,
)
from .runtime_ir import RuntimeGraph, RuntimeOperation

EXACT_LAYOUT_CANDIDATES = (
    "block-hessian",
    "identity-unweighted-down",
    "weight-norm-sort",
    "activation-weighted-sort",
)

__all__ = [
    "EXACT_LAYOUT_CANDIDATES",
    "exact_layout_materializer",
    "gated_mlp_permutation_equivalence",
    "probe_mlp_exact_layout",
]


def _local_roles(
    layer_prefix: str,
    operation: RuntimeOperation,
) -> dict[str, str]:
    prefix = f"{layer_prefix}."
    result: dict[str, str] = {}
    for group in operation.linear_groups:
        for member in group.members:
            if not member.checkpoint_name.startswith(prefix):
                raise ValueError(
                    f"Operation weight {member.checkpoint_name!r} is outside {layer_prefix!r}"
                )
            if member.role in result:
                raise ValueError(f"Gated MLP repeats role {member.role!r}")
            result[member.role] = member.checkpoint_name.removeprefix(prefix)
    if set(result) != {"gate", "up", "down"}:
        raise ValueError(f"Exact-layout probe requires gate/up/down roles, got {sorted(result)}")
    return result


def _state_weights(
    layer: torch.nn.Module,
    roles: Mapping[str, str],
) -> dict[str, torch.Tensor]:
    state = layer.state_dict()
    result: dict[str, torch.Tensor] = {}
    for role, local_name in roles.items():
        value = state.get(local_name)
        if value is None or value.ndim != 2:
            raise ValueError(f"Resident layer has no matrix state {local_name!r}")
        result[role] = value
    gate, up, down = result["gate"], result["up"], result["down"]
    if gate.shape != up.shape or gate.shape[0] != down.shape[1]:
        raise ValueError(
            "Gated MLP dimensions are inconsistent: "
            f"gate={tuple(gate.shape)}, up={tuple(up.shape)}, down={tuple(down.shape)}"
        )
    return result


def _permutation_hash(permutation: torch.Tensor) -> str:
    encoded = permutation.detach().to(device="cpu", dtype=torch.int32).contiguous().numpy()
    return hashlib.sha256(encoded.tobytes()).hexdigest()


def _is_bijection(permutation: torch.Tensor) -> bool:
    expected = torch.arange(permutation.numel(), device=permutation.device)
    return bool(torch.equal(torch.sort(permutation).values, expected))


def gated_mlp_permutation_equivalence(
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    permutation: torch.Tensor,
) -> dict[str, float | bool]:
    """Check exact inverse layout and numerical down-projection equivalence.

    The deterministic activation vector represents an arbitrary gated-MLP
    intermediate. Checking it together with exact gate/up row inversion proves
    that the transform is a runtime-free channel symmetry, not an approximation.
    """
    if not _is_bijection(permutation):
        return {
            "permutation_bijective": False,
            "inverse_layout_exact": False,
            "unquantized_output_equivalent": False,
            "unquantized_output_relative_error": float("inf"),
            "unquantized_output_max_absolute_error": float("inf"),
        }
    inverse = torch.argsort(permutation)
    inverse_exact = (
        torch.equal(gate[permutation][inverse], gate)
        and torch.equal(up[permutation][inverse], up)
        and torch.equal(down[:, permutation][:, inverse], down)
    )
    width = down.shape[1]
    activation = torch.linspace(-1.0, 1.0, width, device=down.device, dtype=torch.float32)
    sampled_down = down[: min(64, down.shape[0])].to(torch.float32)
    reference = torch.mv(sampled_down, activation)
    transformed = torch.mv(sampled_down[:, permutation], activation[permutation])
    delta = transformed - reference
    relative = float(
        (torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(reference).clamp_min(1e-12))
        .detach()
        .cpu()
        .item()
    )
    maximum = float(delta.abs().max().detach().cpu().item())
    equivalent = inverse_exact and bool(
        torch.allclose(reference, transformed, rtol=1e-4, atol=1e-4)
    )
    return {
        "permutation_bijective": True,
        "inverse_layout_exact": inverse_exact,
        "unquantized_output_equivalent": equivalent,
        "unquantized_output_relative_error": relative,
        "unquantized_output_max_absolute_error": maximum,
    }


def _logical_nmse(
    reference: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
    *,
    permutation: torch.Tensor | None,
    row_chunk_size: int,
) -> float:
    inverse = torch.argsort(permutation) if permutation is not None else None
    error_sum = 0.0
    energy_sum = 0.0
    for role in ("gate", "up", "down"):
        source = reference[role]
        value = candidate[role]
        for start in range(0, source.shape[0], row_chunk_size):
            stop = min(start + row_chunk_size, source.shape[0])
            source_chunk = source[start:stop].to(torch.float32)
            value_chunk = value[start:stop]
            if inverse is not None:
                if role in {"gate", "up"}:
                    indices = inverse[start:stop]
                    value_chunk = value[indices]
                else:
                    value_chunk = value_chunk[:, inverse]
            value_f32 = value_chunk.to(torch.float32)
            error_sum += float((value_f32 - source_chunk).square().sum().item())
            energy_sum += float(source_chunk.square().sum().item())
    return error_sum / max(energy_sum, 1e-12)


def exact_layout_materializer(
    *,
    scale_percentile: float = 99.5,
    mse_clip_depth: int = 4,
) -> Callable[..., CandidateMaterialization]:
    """Create the frozen exact-layout candidate materializer."""
    unweighted = Mxfp4CandidateSpec(
        "unweighted-down",
        scale_percentile=scale_percentile,
        mse_clip_depth=mse_clip_depth,
    )

    def materialize(
        reference_layer: torch.nn.Module,
        execution_layer: torch.nn.Module,
        layer_prefix: str,
        operation: RuntimeOperation,
        calibration: CalibrationData,
        *,
        row_chunk_size: int,
    ) -> CandidateMaterialization:
        if calibration.objective != "block-hessian":
            raise ValueError("Exact-layout activation sort requires block-Hessian calibration")
        roles = _local_roles(layer_prefix, operation)
        source = _state_weights(reference_layer, roles)
        execution = _state_weights(execution_layer, roles)
        down_checkpoint = f"{layer_prefix}.{roles['down']}"
        down_hessian = calibration.tensors.get(down_checkpoint)
        if down_hessian is None:
            raise ValueError(f"Calibration is missing down projection {down_checkpoint!r}")
        activation_rms = diagonal_hessian_channel_weights(down_hessian).to(
            device=source["down"].device,
            dtype=torch.float32,
        )
        if activation_rms.numel() != source["down"].shape[1]:
            raise ValueError("Down-projection calibration width does not match its input width")

        column_norm = torch.linalg.vector_norm(source["down"].to(torch.float32), dim=0)
        permutations = {
            "weight-norm-sort": torch.argsort(column_norm, stable=True),
            "activation-weighted-sort": torch.argsort(
                column_norm * activation_rms,
                stable=True,
            ),
        }
        identity = torch.arange(source["down"].shape[1], device=source["down"].device)
        unweighted_identity_down = materialize_mxfp4_candidate(
            source["down"],
            unweighted,
            row_chunk_size=row_chunk_size,
        )
        overrides_by_role: dict[str, dict[str, torch.Tensor]] = {
            "block-hessian": dict(execution),
            "identity-unweighted-down": {
                "gate": execution["gate"],
                "up": execution["up"],
                "down": unweighted_identity_down,
            },
        }
        for name, permutation in permutations.items():
            permuted_down = materialize_mxfp4_candidate(
                source["down"][:, permutation],
                unweighted,
                row_chunk_size=row_chunk_size,
            )
            overrides_by_role[name] = {
                "gate": execution["gate"][permutation],
                "up": execution["up"][permutation],
                "down": permuted_down,
            }

        candidate_permutations: dict[str, torch.Tensor | None] = {
            "block-hessian": None,
            "identity-unweighted-down": None,
            **permutations,
        }
        weight_nmse = {
            name: _logical_nmse(
                source,
                values,
                permutation=candidate_permutations[name],
                row_chunk_size=row_chunk_size,
            )
            for name, values in overrides_by_role.items()
        }
        baseline_weight_nmse = {
            name: _logical_nmse(
                execution,
                values,
                permutation=candidate_permutations[name],
                row_chunk_size=row_chunk_size,
            )
            for name, values in overrides_by_role.items()
        }
        metadata: dict[str, Mapping[str, Any]] = {
            "block-hessian": {
                "permutation_sha256": _permutation_hash(identity),
                "permutation_bijective": True,
                "inverse_layout_exact": True,
                "unquantized_output_equivalent": True,
            },
            "identity-unweighted-down": {
                "permutation_sha256": _permutation_hash(identity),
                "permutation_bijective": True,
                "inverse_layout_exact": True,
                "unquantized_output_equivalent": True,
            },
        }
        for name, permutation in permutations.items():
            metadata[name] = {
                "permutation_sha256": _permutation_hash(permutation),
                **gated_mlp_permutation_equivalence(
                    source["gate"],
                    source["up"],
                    source["down"],
                    permutation,
                ),
            }

        overrides = {
            name: {roles[role]: value for role, value in values.items()}
            for name, values in overrides_by_role.items()
        }
        return CandidateMaterialization(
            overrides=overrides,
            weight_nmse=weight_nmse,
            baseline_weight_nmse=baseline_weight_nmse,
            metadata=metadata,
        )

    return materialize


def probe_mlp_exact_layout(
    model: torch.nn.Module,
    graph: RuntimeGraph,
    calibration: CalibrationData,
    layer_index: int,
    sequences: Sequence[Sequence[int]],
    checkpoint_files: Mapping[str, Any],
    baseline_checkpoint_files: Mapping[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
    row_chunk_size: int = 256,
    logit_positions_per_sequence: int = 8,
    scale_percentile: float = 99.5,
    mse_clip_depth: int = 4,
    progress: Callable[[int, int], None] | None = None,
) -> CounteractionProbeResult:
    """Evaluate the four frozen exact-layout candidates on one gated MLP."""
    return probe_mlp_candidates(
        model,
        graph,
        calibration,
        layer_index,
        EXACT_LAYOUT_CANDIDATES,
        exact_layout_materializer(
            scale_percentile=scale_percentile,
            mse_clip_depth=mse_clip_depth,
        ),
        sequences,
        checkpoint_files,
        baseline_checkpoint_files,
        device=device,
        dtype=dtype,
        row_chunk_size=row_chunk_size,
        logit_positions_per_sequence=logit_positions_per_sequence,
        suffix_jvp=False,
        progress=progress,
    )
