"""Tests for forward-mode suffix sensitivity."""

from __future__ import annotations

import pytest
import torch

from mxwave.suffix_jvp import module_tensor_jvp


class _ResidualNonlinearity(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(4, 4, bias=False, dtype=torch.float64)

    def forward(self, inputs: torch.Tensor, *, scale: float) -> torch.Tensor:
        return inputs + torch.sin(self.projection(inputs) * scale)


def test_module_tensor_jvp_matches_central_difference_under_no_grad() -> None:
    torch.manual_seed(13)
    module = _ResidualNonlinearity().eval()
    inputs = torch.randn(2, 3, 4, dtype=torch.float64)
    tangent = torch.randn_like(inputs)
    epsilon = 1e-5

    with torch.no_grad():
        primal, directional = module_tensor_jvp(
            module,
            inputs,
            tangent,
            {"scale": 0.7},
            lambda output: output,
        )
        upper = module(inputs + epsilon * tangent, scale=0.7)
        lower = module(inputs - epsilon * tangent, scale=0.7)

    assert torch.equal(primal, module(inputs, scale=0.7))
    assert torch.allclose(directional, (upper - lower) / (2.0 * epsilon), atol=1e-9)


def test_module_tensor_jvp_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shapes differ"):
        module_tensor_jvp(
            _ResidualNonlinearity(),
            torch.zeros(2, 4, dtype=torch.float64),
            torch.zeros(1, 4, dtype=torch.float64),
            {"scale": 1.0},
            lambda output: output,
        )
