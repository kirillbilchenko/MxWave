"""Rotation-based quantization (QuaRot / SpinQuant / DuQuant family).

Activation outliers are structural and survive across layers. Applying a fused
orthogonal (Hadamard) rotation to both activations and weights makes outliers
uniform, dramatically reducing 4-bit quantization error.

At inference the rotation is *free*: fold it into the preceding LayerNorm scale
and the next layer's weights, emitting a standard ``compressed-tensors`` model
with ``transform_config`` (QuaRot's "rotation is free" trick).
"""

from __future__ import annotations

import math
from typing import cast

import torch

__all__ = [
    "apply_weight_rotation",
    "fold_rotation",
    "hadamard_matrix",
    "random_orthogonal",
]


def hadamard_matrix(n: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """Return an n x n (normalized) Hadamard matrix, n a power of two.

    Uses the Sylvester construction: H(1)=[[1]], H(2n) = [[H, H],[H, -H]].
    Normalized by 1/sqrt(n) so it is orthogonal (H @ H^T = I).
    """
    dev = torch.device(device)
    if n == 1:
        return torch.ones((1, 1), dtype=torch.float32, device=dev)
    if n & (n - 1) != 0:
        raise ValueError(f"Hadamard requires power-of-two n, got {n}")
    h = torch.tensor([[1.0]], device=dev)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(n)


def random_orthogonal(
    n: int,
    *,
    device: torch.device | str = "cpu",
    seed: int | None = None,
) -> torch.Tensor:
    """Return a random orthogonal n x n matrix (via QR of a Gaussian)."""
    dev = torch.device(device)
    g = torch.Generator(device=dev)
    if seed is not None:
        g.manual_seed(seed)
    a = torch.randn(n, n, device=dev, generator=g)
    q, _ = torch.linalg.qr(a)
    return cast(torch.Tensor, q)


def apply_weight_rotation(
    weight: torch.Tensor,
    rotation: torch.Tensor,
    *,
    dim: int = -1,
) -> torch.Tensor:
    """Rotate a weight matrix along the given (input) dimension.

    For a [out, in] weight W, rotating input activations x by R means the
    equivalent weight is W @ R^T (so that (W R^T)(R x) = W x).
    """
    rot_t = rotation.T
    if dim == -1:
        return weight @ rot_t
    if dim == 0:
        return rot_t @ weight
    raise ValueError(f"Unsupported dim {dim}")


def fold_rotation(
    weight: torch.Tensor,
    rotation: torch.Tensor,
    *,
    dim: int = -1,
) -> torch.Tensor:
    """Fold a rotation into a weight matrix (same as apply_weight_rotation).

    Kept as a distinct name to mirror the QuaRot terminology: the rotation
    applied at one layer is folded into the next layer's weights.
    """
    return apply_weight_rotation(weight, rotation, dim=dim)
