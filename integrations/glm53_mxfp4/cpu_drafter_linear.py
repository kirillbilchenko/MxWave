"""Keep BF16 linear forwards while using BLAS-friendly FP32 CPU derivatives."""

from __future__ import annotations

import types
from typing import Any

import torch
import torch.nn.functional as F


class _CpuLinear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, value: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None,
    ) -> torch.Tensor:
        ctx.save_for_backward(value, weight)
        ctx.has_bias = bias is not None
        ctx.bias_dtype = None if bias is None else bias.dtype
        return F.linear(value, weight, bias)

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[Any, Any, Any]:
        value, weight = ctx.saved_tensors
        precision = torch.float64 if value.dtype == torch.float64 else torch.float32
        flat = gradient.reshape(-1, weight.shape[0]).to(precision)
        grad_value = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_value = (flat @ weight.to(precision)).reshape_as(value).to(value.dtype)
        if ctx.needs_input_grad[1]:
            inputs = value.reshape(-1, weight.shape[1]).to(precision)
            grad_weight = (flat.T @ inputs).to(weight.dtype)
        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_bias = flat.sum(0).to(ctx.bias_dtype)
        return grad_value, grad_weight, grad_bias


def linear(value: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
           ) -> torch.Tensor:
    """Use unchanged CPU forwards and explicitly FP32-accumulated first derivatives."""
    if value.device.type != "cpu" or weight.device.type != "cpu":
        raise ValueError("The training derivative override is CPU-only")
    return _CpuLinear.apply(value, weight, bias)


def install(model: torch.nn.Module) -> int:
    """Override Linear derivatives without changing any weights or forward arithmetic."""
    def forward(module: torch.nn.Linear, value: torch.Tensor) -> torch.Tensor:
        return linear(value, module.weight, module.bias)

    count = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            module.forward = types.MethodType(forward, module)
            count += 1
    return count
