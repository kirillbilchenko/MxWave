"""Unit tests for rotation-based quantization primitives."""

from __future__ import annotations

import torch

from mxstream.rotate import apply_weight_rotation, hadamard_matrix, random_orthogonal


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
    from mxstream.rotate import fold_rotation

    assert torch.allclose(fold_rotation(W, R), apply_weight_rotation(W, R))
