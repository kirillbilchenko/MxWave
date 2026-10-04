"""Unit tests for rotation-based quantization primitives."""

from __future__ import annotations

import torch

from mxwave.rotate import (
    apply_weight_rotation,
    block_hadamard_transform,
    hadamard_matrix,
    random_orthogonal,
)


def test_hadamard_is_orthogonal():
    H = hadamard_matrix(32)
    # H @ H^T ≈ I
    eye = H @ H.T
    assert torch.allclose(eye, torch.eye(32), atol=1e-5)


def test_hadamard_rejects_non_power_of_two():
    try:
        hadamard_matrix(24)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_random_orthogonal_is_orthogonal():
    Q = random_orthogonal(16, seed=0)
    assert torch.allclose(Q @ Q.T, torch.eye(16), atol=1e-4)


def test_apply_weight_rotation_preserves_product():
    # (W @ R^T) @ (R @ x) == W @ x  → rotation is algebraically free
    W = torch.randn(8, 16)
    x = torch.randn(16)
    R = hadamard_matrix(16)
    rotated_w = apply_weight_rotation(W, R)  # W @ R^T
    rotated_x = R @ x
    torch.testing.assert_close(rotated_w @ rotated_x, W @ x, atol=1e-5, rtol=1e-5)


def test_fold_rotation_equals_apply():
    W = torch.randn(8, 16)
    R = hadamard_matrix(16)
    from mxwave.rotate import fold_rotation

    assert torch.allclose(fold_rotation(W, R), apply_weight_rotation(W, R))


def test_output_and_input_rotations_preserve_a_two_linear_chain() -> None:
    producer = torch.randn(16, 7)
    consumer = torch.randn(5, 16)
    rotation = random_orthogonal(16, seed=19)
    assert not torch.allclose(rotation, rotation.T)
    rotated_producer = apply_weight_rotation(producer, rotation, dim=0)
    rotated_consumer = apply_weight_rotation(consumer, rotation, dim=1)
    torch.testing.assert_close(rotated_producer, rotation @ producer)
    torch.testing.assert_close(rotated_consumer @ rotated_producer, consumer @ producer)


def test_block_hadamard_supports_qwen_width_without_dense_rotation() -> None:
    values = torch.randn(3, 5120)
    rotated = block_hadamard_transform(values)
    torch.testing.assert_close(rotated.square().sum(), values.square().sum())
    torch.testing.assert_close(block_hadamard_transform(rotated), values)
    torch.testing.assert_close(block_hadamard_transform(values.T, dim=0), rotated.T)


def test_random_orthogonal_has_unbiased_signs_and_unseeded_randomness() -> None:
    first_entries = [random_orthogonal(2, seed=seed)[0, 0].item() for seed in range(128)]
    assert 32 < sum(value > 0 for value in first_entries) < 96
    assert not torch.equal(random_orthogonal(8), random_orthogonal(8))
    torch.testing.assert_close(random_orthogonal(8, seed=7), random_orthogonal(8, seed=7))
