"""The CPU derivative override preserves forward values and analytical gradients."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from integrations.glm53_mxfp4.cpu_drafter_linear import install, linear


def test_bf16_forward_exact_and_gradients_match_high_precision_formula() -> None:
    generator = torch.Generator().manual_seed(42)
    value = torch.randn(2, 7, 128, generator=generator).bfloat16().requires_grad_()
    weight = torch.randn(32, 128, generator=generator).bfloat16().requires_grad_()
    bias = torch.randn(32, generator=generator).bfloat16().requires_grad_()
    actual = linear(value, weight, bias)
    assert torch.equal(actual, F.linear(value, weight, bias))
    upstream = torch.randn(actual.shape, generator=generator).bfloat16()
    actual.backward(upstream)
    g = upstream.double().reshape(-1, 32)
    references = (
        (g @ weight.detach().double()).reshape_as(value),
        g.T @ value.detach().double().reshape(-1, 128),
        g.sum(0),
    )
    for parameter, reference in zip((value, weight, bias), references, strict=True):
        assert parameter.grad is not None
        assert torch.allclose(parameter.grad.float(), reference.bfloat16().float(),
                              rtol=0.01, atol=0.01)


def test_double_gradcheck_and_frozen_weight() -> None:
    torch.manual_seed(7)
    value = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(4, 5, dtype=torch.float64)
    assert torch.autograd.gradcheck(linear, (value, weight))
    linear(value, weight).sum().backward()
    assert value.grad is not None and weight.grad is None


def test_install_preserves_parameter_identity_and_functional_override() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(5, 3, bias=False)).bfloat16()
    model.requires_grad_(False)
    value = torch.randn(2, 5).bfloat16()
    original = model(value)
    weight = model[0].weight
    assert install(model) == 1 and model[0].weight is weight
    assert torch.equal(model(value), original)
    master = weight.detach().float().clone().requires_grad_()
    output = torch.func.functional_call(model, {"0.weight": master.bfloat16()}, (value,))
    assert torch.equal(output, original)
    output.float().square().sum().backward()
    assert master.grad is not None and torch.isfinite(master.grad).all()
    assert weight.grad is None


def test_reject_non_cpu() -> None:
    with pytest.raises(ValueError, match="CPU-only"):
        linear(torch.empty(2, 4, device="meta"), torch.empty(3, 4, device="meta"))
