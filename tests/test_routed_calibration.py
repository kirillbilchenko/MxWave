"""Routing-aware calibration must match the selected expert inputs exactly."""

from __future__ import annotations

import pytest
import torch

from mxwave.routed_calibration import (
    RoutedLayerMoments,
    dequantize_fp8_activation_blocks,
    soften_routed_rms,
)


def test_routed_rms_matches_weighted_explicit_inputs_and_retains_unseen_experts() -> None:
    moments = RoutedLayerMoments.create(3, 2, 2, torch.device("cpu"))
    inputs = torch.tensor([[1.0, 4.0], [3.0, 2.0]])
    ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
    weights = torch.tensor([[0.2, 0.7], [0.4, 0.9]])
    down = torch.tensor([[2.0, 3.0], [4.0, 1.0], [3.0, 2.0], [1.0, 5.0]])
    moments.update_inputs(inputs, ids, weights)
    moments.update_down(down, ids, weights)
    result, record = moments.finalize(4)
    expected = torch.sqrt(
        (inputs[0].square() * 0.2**2 + inputs[1].square() * 0.9**2) / (0.2**2 + 0.9**2)
    )
    assert torch.allclose(
        result["model.layers.4.mlp.experts.0.gate_proj.weight"], expected / expected.amax()
    )
    expected_down = torch.sqrt(
        (down[1].square() * 0.7**2 + down[2].square() * 0.4**2) / (0.7**2 + 0.4**2)
    )
    assert torch.allclose(
        result["model.layers.4.mlp.experts.1.down_proj.weight"],
        expected_down / expected_down.amax(),
    )
    assert record["routed_observations"] == [2, 2, 0]
    assert record["calibrated_targets"] == 6
    assert len(record["retained_unweighted_targets"]) == 3


def test_capture_rejects_missing_down_observations_and_out_of_range_routes() -> None:
    moments = RoutedLayerMoments.create(2, 2, 2, torch.device("cpu"))
    ids = torch.tensor([[0]], dtype=torch.int32)
    moments.update_inputs(torch.ones(1, 2), ids, torch.ones(1, 1))
    with pytest.raises(ValueError, match="same routed observations"):
        moments.finalize(0)
    with pytest.raises(ValueError, match="Routing IDs"):
        moments.update_down(torch.ones(1, 2), torch.tensor([[2]]), torch.ones(1, 1))


def test_fp8_activation_decode_uses_float_scales_and_rejects_e8m0() -> None:
    values = torch.ones(2, 256).to(torch.float8_e4m3fn)
    scales = torch.tensor([[0.5, 2.0], [4.0, 0.25]])
    decoded = dequantize_fp8_activation_blocks(values, scales)
    assert torch.equal(decoded[:, :128], scales[:, :1].expand(-1, 128))
    assert torch.equal(decoded[:, 128:], scales[:, 1:].expand(-1, 128))
    with pytest.raises(ValueError, match="float32"):
        dequantize_fp8_activation_blocks(values, scales.to(torch.uint8))


def test_soft_rms_shrinkage_is_scale_invariant_and_retains_positive_channel_weights() -> None:
    values = torch.tensor([0.0, 0.01, 1.0, 4.0], dtype=torch.float64)
    low = soften_routed_rms(values, observations=1)
    high = soften_routed_rms(values, observations=10000)
    assert low.dtype == torch.float32 and low.device == values.device
    assert torch.all(low > 0) and torch.all(high > 0)
    assert torch.equal(high, soften_routed_rms(values * 1e300, observations=10000))
    assert float(low.min()) > float(high.min())
    assert float(high.max()) == 1.0
    assert torch.equal(soften_routed_rms(values, observations=0), torch.ones(4))
    assert torch.equal(soften_routed_rms(values, observations=100, strength=0), torch.ones(4))
    # Strength .25 and clipped variance [.25,4] bound the normalized squared objective.
    assert float(high.square().min()) >= (0.75 + 0.25 * 0.25) / (0.75 + 0.25 * 4)


@pytest.mark.parametrize(
    "values",
    [
        torch.zeros(4),
        torch.tensor([-1.0, 2.0]),
        torch.tensor([float("nan")]),
        torch.tensor([float("inf")]),
        torch.ones(2, 2),
        torch.tensor([1, 2]),
    ],
)
def test_soft_rms_rejects_invalid_channel_statistics(values: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="RMS must"):
        soften_routed_rms(values, observations=1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"strength": -0.1},
        {"strength": 1.1},
        {"pseudo_count": -1.0},
        {"max_variance_ratio": 0.5},
        {"strength": float("nan")},
        {"observations": -1},
    ],
)
def test_soft_rms_rejects_invalid_recipe(kwargs: dict[str, float]) -> None:
    arguments = {"observations": 1, **kwargs}
    with pytest.raises(ValueError, match="Invalid activation"):
        soften_routed_rms(torch.ones(4), **arguments)  # type: ignore[arg-type]
