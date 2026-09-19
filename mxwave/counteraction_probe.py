"""Bounded counteraction probes for legal MXFP4 candidates."""

from __future__ import annotations

import copy
import inspect
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch

from .calibration import CalibrationData
from .calibration_stream import (
    _advise_prefix_unused,
    _attention_mask_for_layer,
    _load_module_from_checkpoint,
    _module_at,
    _new_rotary,
    infer_sequential_decoder_layout,
)
from .counteraction import CounteractionMetrics, measure_counteraction
from .module_replay import ModuleReplaySample, make_module_replay
from .mxfp4_candidates import Mxfp4CandidateSpec, materialize_mxfp4_candidate
from .mxfp4_checkpoint import load_mxfp4_module
from .runtime_ir import RuntimeGraph, RuntimeOperation
from .suffix_jvp import module_tensor_jvp

__all__ = [
    "BaselineCounteractionResult",
    "CandidateCounteractionResult",
    "CounteractionProbeResult",
    "probe_mlp_counteraction",
]


@dataclass(frozen=True)
class CandidateCounteractionResult:
    """Local counteraction controls and final teacher KL for one candidate."""

    candidate: str
    weight_nmse: float
    baseline_weight_nmse: float
    mean_operator_nmse: float
    sample_operator_nmse: tuple[float, ...]
    mean_inherited_hidden_nmse: float
    mean_update_error_nmse: float
    mean_interaction_nmse: float
    mean_resulting_hidden_nmse: float
    mean_error_growth_nmse: float
    mean_counteraction_fraction: float
    max_recurrence_relative_residual: float
    sample_counteraction: tuple[CounteractionMetrics, ...]
    mean_teacher_kl: float
    max_teacher_kl: float
    sample_teacher_kl: tuple[float, ...]
    mean_suffix_jvp_teacher_kl: float | None = None
    max_suffix_jvp_teacher_kl: float | None = None
    sample_suffix_jvp_teacher_kl: tuple[float, ...] | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable candidate record."""
        return {
            "candidate": self.candidate,
            "weight_nmse": self.weight_nmse,
            "baseline_weight_nmse": self.baseline_weight_nmse,
            "mean_operator_nmse": self.mean_operator_nmse,
            "sample_operator_nmse": list(self.sample_operator_nmse),
            "mean_inherited_hidden_nmse": self.mean_inherited_hidden_nmse,
            "mean_update_error_nmse": self.mean_update_error_nmse,
            "mean_interaction_nmse": self.mean_interaction_nmse,
            "mean_resulting_hidden_nmse": self.mean_resulting_hidden_nmse,
            "mean_error_growth_nmse": self.mean_error_growth_nmse,
            "mean_counteraction_fraction": self.mean_counteraction_fraction,
            "max_recurrence_relative_residual": self.max_recurrence_relative_residual,
            "sample_counteraction": [item.as_dict() for item in self.sample_counteraction],
            "mean_teacher_kl": self.mean_teacher_kl,
            "max_teacher_kl": self.max_teacher_kl,
            "sample_teacher_kl": list(self.sample_teacher_kl),
            "mean_suffix_jvp_teacher_kl": self.mean_suffix_jvp_teacher_kl,
            "max_suffix_jvp_teacher_kl": self.max_suffix_jvp_teacher_kl,
            "sample_suffix_jvp_teacher_kl": (
                list(self.sample_suffix_jvp_teacher_kl)
                if self.sample_suffix_jvp_teacher_kl is not None
                else None
            ),
        }


@dataclass(frozen=True)
class BaselineCounteractionResult:
    """Final teacher divergence for the unchanged quantized checkpoint."""

    mean_teacher_kl: float
    max_teacher_kl: float
    sample_teacher_kl: tuple[float, ...]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable baseline record."""
        return {
            "mean_teacher_kl": self.mean_teacher_kl,
            "max_teacher_kl": self.max_teacher_kl,
            "sample_teacher_kl": list(self.sample_teacher_kl),
        }


@dataclass(frozen=True)
class CounteractionProbeResult:
    """Counteraction and end-to-end results for one decoder MLP."""

    layer_index: int
    operation: str
    logit_positions_per_sequence: int
    candidates: tuple[CandidateCounteractionResult, ...]
    baseline: BaselineCounteractionResult
    suffix_sensitivity: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable result with predeclared rankings."""
        ranking_keys: tuple[tuple[str, Callable[[CandidateCounteractionResult], float]], ...] = (
            ("weight_ranking", lambda item: item.weight_nmse),
            ("operator_ranking", lambda item: item.mean_operator_nmse),
            ("update_error_ranking", lambda item: item.mean_update_error_nmse),
            (
                "counteraction_ranking",
                lambda item: item.mean_resulting_hidden_nmse,
            ),
            ("teacher_kl_ranking", lambda item: item.mean_teacher_kl),
        )
        result: dict[str, Any] = {
            "layer_index": self.layer_index,
            "operation": self.operation,
            "logit_positions_per_sequence": self.logit_positions_per_sequence,
            "execution_mode": "quantized-baseline-prefix-and-suffix",
            "baseline": self.baseline.as_dict(),
            "suffix_sensitivity": self.suffix_sensitivity,
            "candidates": [candidate.as_dict() for candidate in self.candidates],
        }
        for name, key in ranking_keys:
            candidates_by_name = sorted(self.candidates, key=lambda item: item.candidate)
            result[name] = [
                candidate.candidate
                for candidate in sorted(candidates_by_name, key=key)
            ]
        if all(
            candidate.mean_suffix_jvp_teacher_kl is not None
            for candidate in self.candidates
        ):
            result["suffix_jvp_ranking"] = [
                candidate.candidate
                for candidate in sorted(
                    self.candidates,
                    key=lambda item: (
                        cast(float, item.mean_suffix_jvp_teacher_kl),
                        item.candidate,
                    ),
                )
            ]
        return result


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty sequence")
    return math.fsum(values) / len(values)


def _layer_output(raw: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(raw, torch.Tensor):
        return {"output": raw}
    if isinstance(raw, (tuple, list)) and raw and isinstance(raw[0], torch.Tensor):
        return {"output": raw[0]}
    try:
        first = raw[0]
    except (KeyError, TypeError) as exc:
        raise TypeError("Replayed decoder layer returned no hidden-state tensor") from exc
    if not isinstance(first, torch.Tensor):
        raise TypeError("Replayed decoder layer returned no hidden-state tensor")
    return {"output": first}


def _layer_tensor(raw: Any) -> torch.Tensor:
    """Adapt a decoder-layer return value to its hidden-state tensor."""
    return _layer_output(raw)["output"]


def _tensor_output(raw: Any) -> torch.Tensor:
    """Validate a tensor-returning normalization or projection module."""
    if not isinstance(raw, torch.Tensor):
        raise TypeError("Suffix JVP module did not return a tensor")
    return raw


def _run_layer_with_kwargs(
    layer: torch.nn.Module,
    hidden_states: torch.Tensor,
    *,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    signature = inspect.signature(layer.forward)
    accepts_kwargs = any(
        item.kind == inspect.Parameter.VAR_KEYWORD for item in signature.parameters.values()
    )
    candidates: dict[str, Any] = {
        "position_embeddings": position_embeddings,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "past_key_values": None,
        "past_key_value": None,
        "use_cache": False,
    }
    kwargs = {
        name: value
        for name, value in candidates.items()
        if accepts_kwargs or name in signature.parameters
    }
    raw = layer(hidden_states, **kwargs)
    return _layer_output(raw)["output"], kwargs


def _normalized_mse(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    reference_f32 = reference.detach().to(torch.float32)
    candidate_f32 = candidate.detach().to(torch.float32)
    error = (candidate_f32 - reference_f32).square().mean()
    energy = reference_f32.square().mean().clamp_min(1e-12)
    return float((error / energy).item())


def _mlp_operation(graph: RuntimeGraph, layer_index: int) -> RuntimeOperation:
    matches = [
        operation
        for operation in graph.operations
        if operation.layer_index == layer_index and operation.kind == "gated-mlp"
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Runtime graph must contain one gated MLP for layer {layer_index}; "
            f"found {len(matches)}"
        )
    return matches[0]


def _validate_candidates(
    operation: RuntimeOperation,
    calibration: CalibrationData,
    candidate_specs: Sequence[Mxfp4CandidateSpec],
) -> tuple[Mxfp4CandidateSpec, ...]:
    specs = tuple(candidate_specs)
    if not specs:
        raise ValueError("Counteraction probe requires at least one candidate")
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError("Counteraction candidate names must be unique")
    if any(spec.weighting in {"diagonal-hessian", "block-hessian"} for spec in specs):
        if calibration.objective != "block-hessian":
            raise ValueError("Hessian candidates require block-hessian calibration")
        missing = sorted(
            name for name in operation.checkpoint_names if name not in calibration.tensors
        )
        if missing:
            raise ValueError(
                "Block-Hessian calibration is missing selected operation weights: "
                f"{missing[:5]}"
            )
    return specs


def _normalized_weight_mse(
    reference_by_name: Mapping[str, torch.Tensor],
    candidate_by_name: Mapping[str, torch.Tensor],
    *,
    row_chunk_size: int,
) -> float:
    error_sum = 0.0
    energy_sum = 0.0
    for name, reference in reference_by_name.items():
        candidate = candidate_by_name[name]
        for start in range(0, reference.shape[0], row_chunk_size):
            stop = min(start + row_chunk_size, reference.shape[0])
            reference_chunk = reference[start:stop].to(torch.float32)
            candidate_chunk = candidate[start:stop].to(torch.float32)
            error_sum += float((candidate_chunk - reference_chunk).square().sum().item())
            energy_sum += float(reference_chunk.square().sum().item())
    return error_sum / max(energy_sum, 1e-12)


def _materialize_operation_candidates(
    layer: torch.nn.Module,
    layer_prefix: str,
    operation: RuntimeOperation,
    calibration: CalibrationData,
    candidate_specs: Sequence[Mxfp4CandidateSpec],
    *,
    row_chunk_size: int,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, float]]:
    state = layer.state_dict()
    prefix = f"{layer_prefix}."
    references: dict[str, torch.Tensor] = {}
    local_by_checkpoint: dict[str, str] = {}
    for checkpoint_name in operation.checkpoint_names:
        if not checkpoint_name.startswith(prefix):
            raise ValueError(f"Operation weight {checkpoint_name!r} is outside {layer_prefix!r}")
        local_name = checkpoint_name.removeprefix(prefix)
        reference = state.get(local_name)
        if reference is None:
            raise ValueError(f"Resident layer {layer_prefix!r} has no state {local_name!r}")
        references[local_name] = reference
        local_by_checkpoint[checkpoint_name] = local_name

    overrides_by_candidate: dict[str, dict[str, torch.Tensor]] = {}
    weight_nmse: dict[str, float] = {}
    for spec in candidate_specs:
        overrides = {
            local_name: materialize_mxfp4_candidate(
                references[local_name],
                spec,
                channel_weights=(
                    calibration.tensors.get(checkpoint_name)
                    if calibration.objective in ("mean-abs", "rms")
                    else None
                ),
                block_hessian=calibration.tensors.get(checkpoint_name),
                row_chunk_size=row_chunk_size,
            )
            for checkpoint_name, local_name in local_by_checkpoint.items()
        }
        overrides_by_candidate[spec.name] = overrides
        weight_nmse[spec.name] = _normalized_weight_mse(
            references,
            overrides,
            row_chunk_size=row_chunk_size,
        )
    return overrides_by_candidate, weight_nmse


def _baseline_weight_nmse(
    layer: torch.nn.Module,
    layer_prefix: str,
    operation: RuntimeOperation,
    overrides_by_candidate: Mapping[str, Mapping[str, torch.Tensor]],
    *,
    row_chunk_size: int,
) -> dict[str, float]:
    """Measure candidate weights against the resident packed baseline."""
    state = layer.state_dict()
    prefix = f"{layer_prefix}."
    local_names: list[str] = []
    for checkpoint_name in operation.checkpoint_names:
        if not checkpoint_name.startswith(prefix):
            raise ValueError(f"Operation weight {checkpoint_name!r} is outside {layer_prefix!r}")
        local_name = checkpoint_name.removeprefix(prefix)
        if local_name not in state:
            raise ValueError(f"Resident baseline layer has no state {local_name!r}")
        local_names.append(local_name)

    result: dict[str, float] = {}
    for candidate, overrides in overrides_by_candidate.items():
        error_sum = 0.0
        energy_sum = 0.0
        for local_name in local_names:
            reference = state[local_name]
            value = overrides.get(local_name)
            if value is None:
                raise ValueError(
                    f"Candidate {candidate!r} has no override for baseline state {local_name!r}"
                )
            for start in range(0, reference.shape[0], row_chunk_size):
                stop = min(start + row_chunk_size, reference.shape[0])
                reference_chunk = reference[start:stop].to(torch.float32)
                value_chunk = value[start:stop].to(torch.float32)
                error_sum += float((value_chunk - reference_chunk).square().sum().item())
                energy_sum += float(reference_chunk.square().sum().item())
        result[candidate] = error_sum / max(energy_sum, 1e-12)
    return result


def _embed_sequences(
    model: torch.nn.Module,
    embedding_prefix: str,
    sequences: Sequence[Sequence[int]],
    checkpoint_files: Mapping[str, Path],
    *,
    device: torch.device,
    dtype: torch.dtype,
    baseline: bool,
    batch_size: int,
) -> torch.Tensor:
    template = _module_at(model, embedding_prefix)
    if baseline:
        embedding = load_mxfp4_module(
            copy.deepcopy(template),
            embedding_prefix,
            checkpoint_files,
            device=device,
            dtype=dtype,
        )
    else:
        embedding = _load_module_from_checkpoint(
            copy.deepcopy(template),
            embedding_prefix,
            checkpoint_files,
            device=device,
            dtype=dtype,
        )
    embedding.eval()
    first_parameter = next(embedding.parameters(), None)
    if first_parameter is None or first_parameter.ndim != 2:
        raise ValueError("Sequential counteraction embedding has no matrix parameter")
    sequence_length = len(sequences[0])
    hidden_cpu = torch.empty(
        (len(sequences), sequence_length, first_parameter.shape[-1]),
        dtype=dtype,
        device="cpu",
    )
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            rows = sequences[start : start + batch_size]
            input_ids = torch.tensor(rows, dtype=torch.long, device=device)
            hidden_cpu[start : start + len(rows)].copy_(embedding(input_ids).to("cpu"))
    del embedding
    _advise_prefix_unused(checkpoint_files, embedding_prefix)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return hidden_cpu


def _normalize_positions(
    module: torch.nn.Module,
    hidden_cpu: torch.Tensor,
    *,
    positions: int,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    selected = hidden_cpu[:, -positions:, :]
    output = torch.empty_like(selected)
    with torch.inference_mode():
        for start in range(0, selected.shape[0], batch_size):
            stop = min(start + batch_size, selected.shape[0])
            normalized = module(selected[start:stop].to(device))
            if not isinstance(normalized, torch.Tensor):
                raise TypeError("Runtime output normalization did not return a tensor")
            output[start:stop].copy_(normalized.to("cpu"))
    return output


def _normalize_tangent_positions(
    module: torch.nn.Module,
    hidden_cpu: torch.Tensor,
    tangent_hidden: Mapping[str, torch.Tensor],
    *,
    positions: int,
    device: torch.device,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    """Propagate candidate tangents through the final normalization."""
    selected = hidden_cpu[:, -positions:, :]
    selected_tangents = {
        name: tangent[:, -positions:, :] for name, tangent in tangent_hidden.items()
    }
    outputs = {name: torch.empty_like(tangent) for name, tangent in selected_tangents.items()}
    with torch.no_grad():
        for start in range(0, selected.shape[0], batch_size):
            stop = min(start + batch_size, selected.shape[0])
            inputs = selected[start:stop].to(device)
            for name, tangent in selected_tangents.items():
                _primal, directional = module_tensor_jvp(
                    module,
                    inputs,
                    tangent[start:stop].to(device),
                    {},
                    _tensor_output,
                )
                outputs[name][start:stop].copy_(directional.to("cpu"))
    return outputs


def _teacher_kl(
    reference_projection: torch.nn.Module,
    reference_hidden: torch.Tensor,
    candidate_hidden: Mapping[str, torch.Tensor],
    *,
    candidate_projection: torch.nn.Module,
    device: torch.device,
) -> dict[str, tuple[float, ...]]:
    with torch.inference_mode():
        reference_logits = reference_projection(reference_hidden.to(device))
        if not isinstance(reference_logits, torch.Tensor):
            raise TypeError("Runtime output projection did not return a tensor")
        reference_log_probs = torch.log_softmax(reference_logits.to(torch.float32), dim=-1)
        reference_probs = reference_log_probs.exp()
        results: dict[str, tuple[float, ...]] = {}
        for name, hidden in candidate_hidden.items():
            candidate_logits = candidate_projection(hidden.to(device))
            if not isinstance(candidate_logits, torch.Tensor):
                raise TypeError("Runtime output projection did not return a tensor")
            candidate_log_probs = torch.log_softmax(candidate_logits.to(torch.float32), dim=-1)
            token_kl = torch.sum(
                reference_probs * (reference_log_probs - candidate_log_probs),
                dim=-1,
            ).clamp_min(0.0)
            sample_kl = token_kl.mean(dim=-1)
            results[name] = tuple(float(value) for value in sample_kl.cpu().tolist())
            del candidate_logits, candidate_log_probs, token_kl, sample_kl
    return results


def _suffix_jvp_teacher_kl(
    reference_projection: torch.nn.Module,
    reference_hidden: torch.Tensor,
    execution_projection: torch.nn.Module,
    execution_hidden: torch.Tensor,
    tangent_hidden: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
) -> dict[str, tuple[float, ...]]:
    """Score linearized final logits against the exact BF16 teacher."""
    with torch.no_grad():
        reference_logits = reference_projection(reference_hidden.to(device))
        baseline_input = execution_hidden.to(device)
        baseline_logits = execution_projection(baseline_input)
        if not isinstance(reference_logits, torch.Tensor) or not isinstance(
            baseline_logits, torch.Tensor
        ):
            raise TypeError("Runtime output projection did not return a tensor")
        reference_log_probs = torch.log_softmax(reference_logits.to(torch.float32), dim=-1)
        reference_probs = reference_log_probs.exp()
        results: dict[str, tuple[float, ...]] = {}
        for name, tangent in tangent_hidden.items():
            _primal, tangent_logits = module_tensor_jvp(
                execution_projection,
                baseline_input,
                tangent.to(device),
                {},
                _tensor_output,
            )
            predicted_logits = baseline_logits.to(torch.float32) + tangent_logits.to(
                torch.float32
            )
            predicted_log_probs = torch.log_softmax(predicted_logits, dim=-1)
            token_kl = torch.sum(
                reference_probs * (reference_log_probs - predicted_log_probs),
                dim=-1,
            ).clamp_min(0.0)
            sample_kl = token_kl.mean(dim=-1)
            results[name] = tuple(float(value) for value in sample_kl.cpu().tolist())
            del tangent_logits, predicted_logits, predicted_log_probs, token_kl, sample_kl
    return results


def _position_embeddings(
    rotary: torch.nn.Module,
    hidden: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw = rotary(hidden, position_ids)
    if (
        not isinstance(raw, tuple)
        or len(raw) != 2
        or not all(isinstance(item, torch.Tensor) for item in raw)
    ):
        raise TypeError("rotary_emb must return a (cos, sin) tensor tuple")
    return cast(tuple[torch.Tensor, torch.Tensor], raw)


def probe_mlp_counteraction(
    model: torch.nn.Module,
    graph: RuntimeGraph,
    calibration: CalibrationData,
    layer_index: int,
    candidate_specs: Sequence[Mxfp4CandidateSpec],
    sequences: Sequence[Sequence[int]],
    checkpoint_files: Mapping[str, Path],
    baseline_checkpoint_files: Mapping[str, Path],
    *,
    device: torch.device,
    dtype: torch.dtype,
    row_chunk_size: int = 256,
    logit_positions_per_sequence: int = 8,
    suffix_jvp: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> CounteractionProbeResult:
    """Measure candidate counteraction on the real packed prefix and suffix.

    BF16 supplies the teacher trajectory and candidate source weights. The
    packed checkpoint supplies the execution trajectory. Only one source layer,
    one reconstructed packed layer, and the selected MLP candidates are
    resident at once; hidden trajectories remain on CPU between layers.
    """
    if not sequences:
        raise ValueError("Counteraction probe requires at least one sequence")
    sequence_length = len(sequences[0])
    if sequence_length == 0 or any(len(sequence) != sequence_length for sequence in sequences):
        raise ValueError("Counteraction probe requires equal non-empty sequence lengths")
    if row_chunk_size <= 0:
        raise ValueError("Counteraction row_chunk_size must be positive")
    if not 0 < logit_positions_per_sequence <= sequence_length:
        raise ValueError("Counteraction logit positions must be in [1, sequence length]")
    if graph.output_path is None:
        raise ValueError("Runtime graph has no final output path for teacher-KL scoring")

    operation = _mlp_operation(graph, layer_index)
    specs = _validate_candidates(operation, calibration, candidate_specs)
    gate_name = next(
        member.checkpoint_name
        for group in operation.linear_groups
        for member in group.members
        if member.role == "gate"
    )
    gate_statistic = calibration.tensors.get(gate_name)
    if gate_statistic is None:
        raise ValueError(f"Calibration is missing selected gate weight {gate_name!r}")
    gate_width = (
        gate_statistic.shape[0] * gate_statistic.shape[1]
        if calibration.objective == "block-hessian"
        else gate_statistic.shape[0]
    )
    layout = infer_sequential_decoder_layout(
        model,
        {gate_name: gate_width},
        checkpoint_files,
    )
    if layer_index <= 0 or layer_index >= layout.layer_count:
        raise ValueError(
            f"Counteraction layer must be in [1, {layout.layer_count - 1}], got {layer_index}"
        )

    base = _module_at(model, layout.base_prefix)
    stack = cast(torch.nn.ModuleList, _module_at(model, layout.stack_prefix))
    reference_hidden = _embed_sequences(
        model,
        layout.embedding_prefix,
        sequences,
        checkpoint_files,
        device=device,
        dtype=dtype,
        baseline=False,
        batch_size=1,
    )
    execution_hidden = _embed_sequences(
        model,
        layout.embedding_prefix,
        sequences,
        baseline_checkpoint_files,
        device=device,
        dtype=dtype,
        baseline=True,
        batch_size=1,
    )
    candidate_hidden: dict[str, torch.Tensor] = {}
    tangent_hidden: dict[str, torch.Tensor] = {}
    operator_values: dict[str, list[float]] = {spec.name: [] for spec in specs}
    counteraction_values: dict[str, list[CounteractionMetrics]] = {
        spec.name: [] for spec in specs
    }
    weight_nmse: dict[str, float] = {}
    baseline_weight_nmse: dict[str, float] = {}

    rotary = _new_rotary(
        _module_at(base, "rotary_emb"),
        base.config,
        f"{layout.base_prefix}.rotary_emb",
        checkpoint_files,
        device=device,
        dtype=dtype,
    )
    position_ids = torch.arange(sequence_length, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        for current_layer in range(layout.layer_count):
            layer_prefix = f"{layout.stack_prefix}.{current_layer}"
            reference_layer = _load_module_from_checkpoint(
                copy.deepcopy(stack[current_layer]),
                layer_prefix,
                checkpoint_files,
                device=device,
                dtype=dtype,
            )
            reference_layer.eval()
            execution_layer = load_mxfp4_module(
                copy.deepcopy(stack[current_layer]),
                layer_prefix,
                baseline_checkpoint_files,
                device=device,
                dtype=dtype,
            )
            execution_layer.eval()

            overrides: dict[str, dict[str, torch.Tensor]] = {}
            replay: Any = None
            if current_layer == layer_index:
                overrides, weight_nmse = _materialize_operation_candidates(
                    reference_layer,
                    layer_prefix,
                    operation,
                    calibration,
                    specs,
                    row_chunk_size=row_chunk_size,
                )
                baseline_weight_nmse = _baseline_weight_nmse(
                    execution_layer,
                    layer_prefix,
                    operation,
                    overrides,
                    row_chunk_size=row_chunk_size,
                )
                replay = make_module_replay(execution_layer, _layer_output)

            next_reference = torch.empty_like(reference_hidden)
            next_execution = torch.empty_like(execution_hidden)
            next_candidates = (
                {spec.name: torch.empty_like(execution_hidden) for spec in specs}
                if current_layer >= layer_index
                else {}
            )
            next_tangents = (
                {spec.name: torch.empty_like(execution_hidden) for spec in specs}
                if suffix_jvp and current_layer >= layer_index
                else {}
            )
            for sample_index in range(len(sequences)):
                sample_slice = slice(sample_index, sample_index + 1)
                ones = torch.ones(
                    (1, sequence_length),
                    dtype=torch.long,
                    device=device,
                )
                reference_input = reference_hidden[sample_slice].to(device)
                reference_mask = _attention_mask_for_layer(
                    base,
                    reference_input,
                    ones,
                    position_ids,
                    current_layer,
                )
                reference_output, _reference_kwargs = _run_layer_with_kwargs(
                    reference_layer,
                    reference_input,
                    position_embeddings=_position_embeddings(
                        rotary,
                        reference_input,
                        position_ids,
                    ),
                    attention_mask=reference_mask,
                    position_ids=position_ids,
                )
                reference_output_cpu = reference_output.to("cpu")
                next_reference[sample_slice].copy_(reference_output_cpu)

                execution_input = execution_hidden[sample_slice].to(device)
                execution_mask = _attention_mask_for_layer(
                    base,
                    execution_input,
                    ones,
                    position_ids,
                    current_layer,
                )
                execution_output, execution_kwargs = _run_layer_with_kwargs(
                    execution_layer,
                    execution_input,
                    position_embeddings=_position_embeddings(
                        rotary,
                        execution_input,
                        position_ids,
                    ),
                    attention_mask=execution_mask,
                    position_ids=position_ids,
                )
                execution_output_cpu = execution_output.to("cpu")
                next_execution[sample_slice].copy_(execution_output_cpu)

                if current_layer == layer_index:
                    if replay is None:
                        raise AssertionError("Counteraction replay was not initialized")
                    replay_sample = ModuleReplaySample(
                        args=(execution_input,),
                        kwargs=execution_kwargs,
                    )
                    for spec in specs:
                        candidate_output = replay(overrides[spec.name], replay_sample)["output"]
                        candidate_output_cpu = candidate_output.to("cpu")
                        operator_values[spec.name].append(
                            _normalized_mse(execution_output, candidate_output)
                        )
                        counteraction_values[spec.name].append(
                            measure_counteraction(
                                reference_hidden[sample_slice],
                                execution_hidden[sample_slice],
                                reference_output_cpu,
                                candidate_output_cpu,
                            )
                        )
                        next_candidates[spec.name][sample_slice].copy_(candidate_output_cpu)
                        if suffix_jvp:
                            next_tangents[spec.name][sample_slice].copy_(
                                candidate_output_cpu - execution_output_cpu
                            )
                elif current_layer > layer_index:
                    for spec in specs:
                        candidate_input = candidate_hidden[spec.name][sample_slice].to(device)
                        candidate_mask = _attention_mask_for_layer(
                            base,
                            candidate_input,
                            ones,
                            position_ids,
                            current_layer,
                        )
                        candidate_output, _candidate_kwargs = _run_layer_with_kwargs(
                            execution_layer,
                            candidate_input,
                            position_embeddings=_position_embeddings(
                                rotary,
                                candidate_input,
                                position_ids,
                            ),
                            attention_mask=candidate_mask,
                            position_ids=position_ids,
                        )
                        next_candidates[spec.name][sample_slice].copy_(
                            candidate_output.to("cpu")
                        )
                        if suffix_jvp:
                            tangent_input = tangent_hidden[spec.name][sample_slice].to(device)
                            _primal, tangent_output = module_tensor_jvp(
                                execution_layer,
                                execution_input,
                                tangent_input,
                                execution_kwargs,
                                _layer_tensor,
                            )
                            next_tangents[spec.name][sample_slice].copy_(
                                tangent_output.to("cpu")
                            )

            reference_hidden = next_reference
            execution_hidden = next_execution
            if current_layer >= layer_index:
                candidate_hidden = next_candidates
                if suffix_jvp:
                    tangent_hidden = next_tangents
            del reference_layer, execution_layer, overrides, replay
            _advise_prefix_unused(checkpoint_files, layer_prefix)
            _advise_prefix_unused(baseline_checkpoint_files, layer_prefix)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if progress is not None:
                progress(current_layer + 1, layout.layer_count)

    output_path = graph.output_path
    reference_normalization = _load_module_from_checkpoint(
        copy.deepcopy(_module_at(model, output_path.normalization_module)),
        output_path.normalization_module,
        checkpoint_files,
        device=device,
        dtype=dtype,
    )
    reference_normalization.eval()
    execution_normalization = load_mxfp4_module(
        copy.deepcopy(_module_at(model, output_path.normalization_module)),
        output_path.normalization_module,
        baseline_checkpoint_files,
        device=device,
        dtype=dtype,
    )
    execution_normalization.eval()
    normalized_reference = _normalize_positions(
        reference_normalization,
        reference_hidden,
        positions=logit_positions_per_sequence,
        device=device,
        batch_size=1,
    )
    normalized_execution = _normalize_positions(
        execution_normalization,
        execution_hidden,
        positions=logit_positions_per_sequence,
        device=device,
        batch_size=1,
    )
    normalized_candidates = {
        name: _normalize_positions(
            execution_normalization,
            hidden,
            positions=logit_positions_per_sequence,
            device=device,
            batch_size=1,
        )
        for name, hidden in candidate_hidden.items()
    }
    normalized_tangents = (
        _normalize_tangent_positions(
            execution_normalization,
            execution_hidden,
            tangent_hidden,
            positions=logit_positions_per_sequence,
            device=device,
            batch_size=1,
        )
        if suffix_jvp
        else {}
    )
    del reference_normalization, execution_normalization
    del reference_hidden, execution_hidden, candidate_hidden, tangent_hidden, rotary
    _advise_prefix_unused(checkpoint_files, output_path.normalization_module)
    _advise_prefix_unused(baseline_checkpoint_files, output_path.normalization_module)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    reference_projection = _load_module_from_checkpoint(
        copy.deepcopy(_module_at(model, output_path.projection_module)),
        output_path.projection_module,
        checkpoint_files,
        device=device,
        dtype=dtype,
    )
    reference_projection.eval()
    execution_projection = load_mxfp4_module(
        copy.deepcopy(_module_at(model, output_path.projection_module)),
        output_path.projection_module,
        baseline_checkpoint_files,
        device=device,
        dtype=dtype,
    )
    execution_projection.eval()
    scoring_hidden = dict(normalized_candidates)
    scoring_hidden["__baseline__"] = normalized_execution
    kl_by_candidate = _teacher_kl(
        reference_projection,
        normalized_reference,
        scoring_hidden,
        candidate_projection=execution_projection,
        device=device,
    )
    suffix_jvp_kl_by_candidate = (
        _suffix_jvp_teacher_kl(
            reference_projection,
            normalized_reference,
            execution_projection,
            normalized_execution,
            normalized_tangents,
            device=device,
        )
        if suffix_jvp
        else {}
    )
    del reference_projection, execution_projection
    del normalized_reference, normalized_execution, normalized_candidates, normalized_tangents
    del scoring_hidden
    _advise_prefix_unused(checkpoint_files, output_path.projection_module)
    _advise_prefix_unused(baseline_checkpoint_files, output_path.projection_module)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    results: list[CandidateCounteractionResult] = []
    for spec in specs:
        operator_samples = tuple(operator_values[spec.name])
        metrics = tuple(counteraction_values[spec.name])
        kl_values = kl_by_candidate[spec.name]
        suffix_jvp_kl_values = suffix_jvp_kl_by_candidate.get(spec.name)
        if not (
            len(operator_samples) == len(metrics) == len(kl_values) == len(sequences)
        ):
            raise ValueError(
                f"Counteraction candidate {spec.name!r} produced incomplete sample coverage"
            )
        results.append(
            CandidateCounteractionResult(
                candidate=spec.name,
                weight_nmse=weight_nmse[spec.name],
                baseline_weight_nmse=baseline_weight_nmse[spec.name],
                mean_operator_nmse=_mean(operator_samples),
                sample_operator_nmse=operator_samples,
                mean_inherited_hidden_nmse=_mean(
                    [item.inherited_hidden_nmse for item in metrics]
                ),
                mean_update_error_nmse=_mean([item.update_error_nmse for item in metrics]),
                mean_interaction_nmse=_mean([item.interaction_nmse for item in metrics]),
                mean_resulting_hidden_nmse=_mean(
                    [item.resulting_hidden_nmse for item in metrics]
                ),
                mean_error_growth_nmse=_mean([item.error_growth_nmse for item in metrics]),
                mean_counteraction_fraction=_mean(
                    [item.counteraction_fraction for item in metrics]
                ),
                max_recurrence_relative_residual=max(
                    item.recurrence_relative_residual for item in metrics
                ),
                sample_counteraction=metrics,
                mean_teacher_kl=_mean(kl_values),
                max_teacher_kl=max(kl_values),
                sample_teacher_kl=kl_values,
                mean_suffix_jvp_teacher_kl=(
                    _mean(suffix_jvp_kl_values)
                    if suffix_jvp_kl_values is not None
                    else None
                ),
                max_suffix_jvp_teacher_kl=(
                    max(suffix_jvp_kl_values)
                    if suffix_jvp_kl_values is not None
                    else None
                ),
                sample_suffix_jvp_teacher_kl=suffix_jvp_kl_values,
            )
        )

    baseline_values = kl_by_candidate.get("__baseline__")
    if baseline_values is None or len(baseline_values) != len(sequences):
        raise ValueError("Quantized baseline produced incomplete teacher-KL coverage")
    return CounteractionProbeResult(
        layer_index=layer_index,
        operation=operation.name,
        logit_positions_per_sequence=logit_positions_per_sequence,
        candidates=tuple(results),
        baseline=BaselineCounteractionResult(
            mean_teacher_kl=_mean(baseline_values),
            max_teacher_kl=max(baseline_values),
            sample_teacher_kl=baseline_values,
        ),
        suffix_sensitivity="forward-ad" if suffix_jvp else None,
    )
