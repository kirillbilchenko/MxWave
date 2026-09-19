"""Tests for exact gated-MLP layout candidates."""

from __future__ import annotations

import torch

from mxwave.calibration import CalibrationData
from mxwave.exact_layout import (
    EXACT_LAYOUT_CANDIDATES,
    exact_layout_materializer,
    gated_mlp_permutation_equivalence,
)
from mxwave.runtime_ir import (
    ResponsePoint,
    RuntimeLinearGroup,
    RuntimeOperation,
    RuntimeWeight,
)


class _TinyMlp(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = torch.nn.Linear(32, 32, bias=False)
        self.up_proj = torch.nn.Linear(32, 32, bias=False)
        self.down_proj = torch.nn.Linear(32, 32, bias=False)


def _operation() -> RuntimeOperation:
    prefix = "layers.0.mlp"
    return RuntimeOperation(
        name=prefix,
        layer_index=0,
        kind="gated-mlp",
        linear_groups=(
            RuntimeLinearGroup(
                f"{prefix}.gate_up_proj",
                (
                    RuntimeWeight(f"{prefix}.gate_proj.weight", "gate"),
                    RuntimeWeight(f"{prefix}.up_proj.weight", "up"),
                ),
            ),
            RuntimeLinearGroup(
                f"{prefix}.down_proj",
                (RuntimeWeight(f"{prefix}.down_proj.weight", "down"),),
            ),
        ),
        response_points=(ResponsePoint("output"),),
    )


def test_permutation_is_function_preserving() -> None:
    torch.manual_seed(13)
    layer = _TinyMlp().double()
    permutation = torch.randperm(32)
    inputs = torch.randn(4, 32, dtype=torch.float64)
    original = layer.down_proj(
        torch.nn.functional.silu(layer.gate_proj(inputs)) * layer.up_proj(inputs)
    )
    transformed = torch.nn.functional.linear(
        torch.nn.functional.silu(
            torch.nn.functional.linear(inputs, layer.gate_proj.weight[permutation])
        )
        * torch.nn.functional.linear(inputs, layer.up_proj.weight[permutation]),
        layer.down_proj.weight[:, permutation],
    )

    assert torch.allclose(original, transformed, rtol=1e-12, atol=1e-12)
    invariants = gated_mlp_permutation_equivalence(
        layer.gate_proj.weight,
        layer.up_proj.weight,
        layer.down_proj.weight,
        permutation,
    )
    assert invariants["permutation_bijective"] is True
    assert invariants["inverse_layout_exact"] is True
    assert invariants["unquantized_output_equivalent"] is True


def test_materializer_reproduces_baseline_and_emits_stable_invariants() -> None:
    torch.manual_seed(17)
    reference = _TinyMlp().eval()
    execution = _TinyMlp().eval()
    execution.load_state_dict(reference.state_dict())
    prefix = "layers.0.mlp"
    identity = torch.eye(32).unsqueeze(0)
    calibration = CalibrationData(
        objective="block-hessian",
        tensors={
            f"{prefix}.{name}.weight": identity.clone()
            for name in ("gate_proj", "up_proj", "down_proj")
        },
        metadata={},
        file_sha256="test",
    )
    materialize = exact_layout_materializer(mse_clip_depth=2)
    first = materialize(
        reference,
        execution,
        prefix,
        _operation(),
        calibration,
        row_chunk_size=7,
    )
    second = materialize(
        reference,
        execution,
        prefix,
        _operation(),
        calibration,
        row_chunk_size=7,
    )

    assert set(first.overrides) == set(EXACT_LAYOUT_CANDIDATES)
    assert first.baseline_weight_nmse["block-hessian"] == 0.0
    for candidate in EXACT_LAYOUT_CANDIDATES:
        assert first.metadata[candidate]["permutation_bijective"] is True
        assert first.metadata[candidate]["inverse_layout_exact"] is True
        assert first.metadata[candidate]["unquantized_output_equivalent"] is True
        assert (
            first.metadata[candidate]["permutation_sha256"]
            == second.metadata[candidate]["permutation_sha256"]
        )
