"""Forward-mode suffix sensitivity for bounded candidate probes."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, cast

import torch

__all__ = ["module_tensor_jvp"]


def module_tensor_jvp(
    module: torch.nn.Module,
    inputs: torch.Tensor,
    tangent: torch.Tensor,
    kwargs: Mapping[str, Any],
    output_adapter: Callable[[Any], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a module output and its directional derivative.

    The module state and keyword arguments are treated as fixed. Only
    ``inputs`` carries a tangent. This makes the function suitable for
    streaming one decoder layer at a time along a frozen execution trajectory.
    """
    if inputs.shape != tangent.shape:
        raise ValueError(
            f"JVP input and tangent shapes differ: {tuple(inputs.shape)} vs "
            f"{tuple(tangent.shape)}"
        )
    if inputs.device != tangent.device:
        raise ValueError(
            f"JVP input and tangent devices differ: {inputs.device} vs {tangent.device}"
        )
    if inputs.dtype != tangent.dtype:
        raise ValueError(
            f"JVP input and tangent dtypes differ: {inputs.dtype} vs {tangent.dtype}"
        )
    if not torch.is_floating_point(inputs):
        raise TypeError("JVP inputs must be floating point")

    call_kwargs = dict(kwargs)

    def forward(value: torch.Tensor) -> torch.Tensor:
        output = output_adapter(module(value, **call_kwargs))
        if not isinstance(output, torch.Tensor):
            raise TypeError("JVP output adapter must return a tensor")
        return output

    raw_result = torch.func.jvp(forward, (inputs,), (tangent,), has_aux=False)
    if len(raw_result) != 2:
        raise TypeError("JVP unexpectedly returned auxiliary output")
    primal, directional = cast(tuple[torch.Tensor, torch.Tensor], raw_result)
    if primal.shape != directional.shape:
        raise ValueError(
            f"JVP output and tangent shapes differ: {tuple(primal.shape)} vs "
            f"{tuple(directional.shape)}"
        )
    if not bool(torch.isfinite(directional).all().item()):
        raise ValueError("JVP produced NaN or infinity")
    return primal, directional
