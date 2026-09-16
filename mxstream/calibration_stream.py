"""Bounded-residency, sequential decoder calibration.

The runner keeps hidden states on CPU and loads one original-precision decoder
layer at a time from safetensors.  It deliberately uses the Transformers layer
implementation instead of reimplementing model math.
"""

from __future__ import annotations

import copy
import importlib
import inspect
import os
import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from safetensors import safe_open

from .calibration import (
    ActivationCollector,
    CalibrationObjective,
    attach_activation_hooks,
)

_LAYER_TARGET = re.compile(r"^(?P<stack>.+\.layers)\.(?P<index>\d+)\..+\.weight$")
_SUPPORTED_MODEL_TYPES = frozenset({"qwen3_5_text"})


@dataclass(frozen=True)
class SequentialDecoderLayout:
    """Validated module and checkpoint layout for sequential calibration."""

    base_prefix: str
    stack_prefix: str
    embedding_prefix: str
    layer_count: int


@dataclass(frozen=True)
class StreamingCalibrationResult:
    """Statistics and execution facts produced by sequential calibration."""

    statistics: dict[CalibrationObjective, dict[str, torch.Tensor]]
    observation_counts: dict[str, int]
    layer_count: int


def _module_at(model: torch.nn.Module, name: str) -> torch.nn.Module:
    modules = dict(model.named_modules())
    module = modules.get(name)
    if module is None:
        raise ValueError(f"Model is missing sequential-calibration module {name!r}")
    return cast(torch.nn.Module, module)


def infer_sequential_decoder_layout(
    model: torch.nn.Module,
    target_widths: Mapping[str, int],
    checkpoint_files: Mapping[str, Path],
) -> SequentialDecoderLayout:
    """Infer and validate one standard ``embed_tokens -> layers`` decoder stack."""
    stacks: set[str] = set()
    target_indices: set[int] = set()
    for target_name in target_widths:
        matched = _LAYER_TARGET.fullmatch(target_name)
        if matched is None:
            raise ValueError(
                "Sequential weight streaming requires every calibration target inside "
                f"one decoder layer stack; unsupported target: {target_name!r}"
            )
        stacks.add(matched.group("stack"))
        target_indices.add(int(matched.group("index")))
    if len(stacks) != 1:
        raise ValueError(
            "Sequential weight streaming requires exactly one decoder layer stack; "
            f"found {sorted(stacks)}"
        )

    stack_prefix = stacks.pop()
    base_prefix = stack_prefix.removesuffix(".layers")
    embedding_prefix = f"{base_prefix}.embed_tokens"
    embedding_key = f"{embedding_prefix}.weight"
    if embedding_key not in checkpoint_files:
        raise ValueError(
            f"Sequential weight streaming requires checkpoint tensor {embedding_key!r}"
        )

    stack = _module_at(model, stack_prefix)
    if not isinstance(stack, torch.nn.ModuleList) or len(stack) == 0:
        raise ValueError(f"Sequential decoder stack {stack_prefix!r} is not a non-empty ModuleList")
    if target_indices.difference(range(len(stack))):
        raise ValueError("Calibration target references a decoder layer outside the model")
    for index in range(len(stack)):
        layer_prefix = f"{stack_prefix}.{index}."
        if not any(name.startswith(layer_prefix) for name in checkpoint_files):
            raise ValueError(f"Checkpoint contains no tensors for decoder layer {index}")

    base = _module_at(model, base_prefix)
    if not hasattr(base, "config"):
        raise ValueError(f"Decoder base module {base_prefix!r} has no config")
    model_type = getattr(base.config, "model_type", None)
    if model_type not in _SUPPORTED_MODEL_TYPES:
        raise ValueError(
            "Sequential weight streaming has no verified adapter for model_type "
            f"{model_type!r}; use --weight-loading resident explicitly or add a tested adapter"
        )
    if not hasattr(base, "rotary_emb"):
        raise ValueError(f"Decoder base module {base_prefix!r} has no rotary_emb")
    layer_types = getattr(base.config, "layer_types", None)
    if layer_types is not None and (
        not isinstance(layer_types, list) or len(layer_types) != len(stack)
    ):
        raise ValueError("Decoder config has an invalid layer_types list")

    return SequentialDecoderLayout(
        base_prefix=base_prefix,
        stack_prefix=stack_prefix,
        embedding_prefix=embedding_prefix,
        layer_count=len(stack),
    )


def _advise_prefix_unused(
    checkpoint_files: Mapping[str, Path],
    module_prefix: str,
) -> None:
    """Ask the OS to reclaim checkpoint page cache after a module is consumed."""
    advise = getattr(os, "posix_fadvise", None)
    advice = getattr(os, "POSIX_FADV_DONTNEED", None)
    if advise is None or advice is None:
        return
    prefix = f"{module_prefix}."
    for path in {path for name, path in checkpoint_files.items() if name.startswith(prefix)}:
        try:
            descriptor = os.open(path, os.O_RDONLY)
            try:
                advise(descriptor, 0, 0, advice)
            finally:
                os.close(descriptor)
        except OSError:
            # Cache reclamation is an optional memory-accounting hint.  A
            # filesystem that rejects it must not invalidate calibration.
            continue


def _load_module_from_checkpoint(
    module: torch.nn.Module,
    module_prefix: str,
    checkpoint_files: Mapping[str, Path],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    expected = tuple(module.state_dict().keys())
    if not expected:
        return module.to(device)

    keys_by_file: dict[Path, list[tuple[str, str]]] = defaultdict(list)
    for local_name in expected:
        checkpoint_name = f"{module_prefix}.{local_name}"
        path = checkpoint_files.get(checkpoint_name)
        if path is None:
            raise ValueError(f"Checkpoint is missing tensor {checkpoint_name!r}")
        keys_by_file[path].append((local_name, checkpoint_name))

    state: dict[str, torch.Tensor] = {}
    for path, names in keys_by_file.items():
        with safe_open(str(path), framework="pt", device=str(device)) as source:
            for local_name, checkpoint_name in names:
                value = cast(torch.Tensor, source.get_tensor(checkpoint_name))
                if value.is_floating_point() and value.dtype != dtype:
                    value = value.to(dtype=dtype)
                state[local_name] = value
    module.load_state_dict(state, strict=True, assign=True)
    for name, value in module.state_dict().items():
        if value.device.type == "meta":
            raise ValueError(f"Sequential module retained an unloaded meta tensor: {name}")
    return module


def _new_rotary(
    template: torch.nn.Module,
    config: Any,
    prefix: str,
    checkpoint_files: Mapping[str, Path],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    rotary_type = type(template)
    try:
        rotary = rotary_type(config, device=device)
    except TypeError:
        rotary = rotary_type(config)
    if not isinstance(rotary, torch.nn.Module):
        raise TypeError("rotary_emb constructor did not return a torch module")
    if rotary.state_dict():
        rotary = _load_module_from_checkpoint(
            rotary,
            prefix,
            checkpoint_files,
            device=device,
            dtype=dtype,
        )
    else:
        rotary.to(device)
    rotary.eval()
    return rotary


def _attention_mask_for_layer(
    base: torch.nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    layer_index: int,
) -> torch.Tensor | None:
    layer_types = getattr(base.config, "layer_types", None)
    layer_type = layer_types[layer_index] if isinstance(layer_types, list) else "full_attention"
    module = importlib.import_module(type(base).__module__)
    function_name = (
        "create_recurrent_attention_mask"
        if layer_type == "linear_attention"
        else "create_causal_mask"
    )
    mask_function = getattr(module, function_name, None)
    if mask_function is None:
        if layer_type == "linear_attention":
            raise ValueError(
                f"{type(base).__name__} does not expose {function_name} for streaming calibration"
            )
        return None
    result = mask_function(
        config=base.config,
        inputs_embeds=hidden_states,
        attention_mask=attention_mask,
        past_key_values=None,
        position_ids=position_ids,
    )
    if result is not None and not isinstance(result, torch.Tensor):
        raise TypeError("Sequential calibration currently requires tensor attention masks")
    return result


def _run_layer(
    layer: torch.nn.Module,
    hidden_states: torch.Tensor,
    *,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor,
) -> torch.Tensor:
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
    output = layer(hidden_states, **kwargs)
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    try:
        first = output[0]
    except (KeyError, TypeError) as exc:
        raise TypeError("Decoder layer returned no hidden-state tensor") from exc
    if not isinstance(first, torch.Tensor):
        raise TypeError("Decoder layer returned no hidden-state tensor")
    return first


def calibrate_decoder_sequentially(
    model: torch.nn.Module,
    target_widths: Mapping[str, int],
    objectives: Sequence[CalibrationObjective],
    sequences: Sequence[Sequence[int]],
    checkpoint_files: Mapping[str, Path],
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    hessian_damp: float,
    progress: Callable[[int, int], None] | None = None,
) -> StreamingCalibrationResult:
    """Capture statistics while keeping at most one decoder layer resident."""
    if not sequences:
        raise ValueError("Sequential calibration requires at least one sequence")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    sequence_length = len(sequences[0])
    if sequence_length == 0 or any(len(sequence) != sequence_length for sequence in sequences):
        raise ValueError("Sequential calibration requires equal non-empty sequence lengths")

    layout = infer_sequential_decoder_layout(model, target_widths, checkpoint_files)
    base = _module_at(model, layout.base_prefix)
    stack = cast(torch.nn.ModuleList, _module_at(model, layout.stack_prefix))
    embedding_template = _module_at(model, layout.embedding_prefix)
    embedding = _load_module_from_checkpoint(
        copy.deepcopy(embedding_template),
        layout.embedding_prefix,
        checkpoint_files,
        device=device,
        dtype=dtype,
    )
    embedding.eval()

    first_parameter = next(embedding.parameters(), None)
    if first_parameter is None or first_parameter.ndim != 2:
        raise ValueError("Sequential calibration embedding has no matrix parameter")
    hidden_width = first_parameter.shape[-1]
    hidden_cpu = torch.empty(
        (len(sequences), sequence_length, hidden_width),
        dtype=dtype,
        device="cpu",
    )
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            rows = sequences[start : start + batch_size]
            input_ids = torch.tensor(rows, dtype=torch.long, device=device)
            hidden_cpu[start : start + len(rows)].copy_(embedding(input_ids).to("cpu"))
    del embedding
    _advise_prefix_unused(checkpoint_files, layout.embedding_prefix)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    rotary_template = _module_at(base, "rotary_emb")
    rotary = _new_rotary(
        rotary_template,
        base.config,
        f"{layout.base_prefix}.rotary_emb",
        checkpoint_files,
        device=device,
        dtype=dtype,
    )
    position_ids = torch.arange(sequence_length, dtype=torch.long, device=device).unsqueeze(0)
    statistics: dict[CalibrationObjective, dict[str, torch.Tensor]] = {
        objective: {} for objective in objectives
    }
    counts: dict[str, int] = {}

    with torch.inference_mode():
        for layer_index in range(layout.layer_count):
            layer_prefix = f"{layout.stack_prefix}.{layer_index}"
            layer = _load_module_from_checkpoint(
                copy.deepcopy(stack[layer_index]),
                layer_prefix,
                checkpoint_files,
                device=device,
                dtype=dtype,
            )
            layer.eval()
            layer_widths = {
                name: width
                for name, width in target_widths.items()
                if name.startswith(f"{layer_prefix}.")
            }
            collector = (
                ActivationCollector(
                    layer_widths,
                    objectives,
                    hessian_damp=hessian_damp,
                )
                if layer_widths
                else None
            )
            handles = (
                attach_activation_hooks(
                    layer,
                    collector,
                    module_prefix=layer_prefix,
                )
                if collector is not None
                else []
            )
            next_hidden_cpu = torch.empty_like(hidden_cpu)
            try:
                for start in range(0, len(sequences), batch_size):
                    stop = min(start + batch_size, len(sequences))
                    hidden = hidden_cpu[start:stop].to(device)
                    ones = torch.ones(
                        (stop - start, sequence_length),
                        dtype=torch.long,
                        device=device,
                    )
                    layer_mask = _attention_mask_for_layer(
                        base,
                        hidden,
                        ones,
                        position_ids,
                        layer_index,
                    )
                    raw_position_embeddings = rotary(hidden, position_ids)
                    if (
                        not isinstance(raw_position_embeddings, tuple)
                        or len(raw_position_embeddings) != 2
                        or not all(
                            isinstance(item, torch.Tensor) for item in raw_position_embeddings
                        )
                    ):
                        raise TypeError("rotary_emb must return a (cos, sin) tensor tuple")
                    position_embeddings = cast(
                        tuple[torch.Tensor, torch.Tensor], raw_position_embeddings
                    )
                    output = _run_layer(
                        layer,
                        hidden,
                        position_embeddings=position_embeddings,
                        attention_mask=layer_mask,
                        position_ids=position_ids,
                    )
                    next_hidden_cpu[start:stop].copy_(output.to("cpu"))
            finally:
                for handle in handles:
                    handle.remove()

            if collector is not None:
                layer_statistics = collector.finalize()
                for objective, values in layer_statistics.items():
                    statistics[objective].update(values)
                counts.update(collector.observation_counts)
            hidden_cpu = next_hidden_cpu
            del layer, collector
            _advise_prefix_unused(checkpoint_files, layer_prefix)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if progress is not None:
                progress(layer_index + 1, layout.layer_count)

    if set(counts) != set(target_widths):
        missing = sorted(set(target_widths).difference(counts))
        raise ValueError(f"Sequential calibration missed {len(missing)} target(s): {missing[:5]}")
    return StreamingCalibrationResult(
        statistics=statistics,
        observation_counts=counts,
        layer_count=layout.layer_count,
    )
