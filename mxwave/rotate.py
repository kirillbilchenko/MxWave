"""Experimental orthogonal rotations and bounded, block-wise Hadamard transforms.

Function-preserving checkpoint integration requires architecture-aware folding
and, for online activation transforms, runtime support. These primitives are
not wired into checkpoint emission.
"""

from __future__ import annotations

import math
from typing import cast

import torch

__all__ = [
    "apply_weight_rotation",
    "block_hadamard_transform",
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
    if n <= 0:
        raise ValueError("Hadamard size must be positive")
    if n == 1:
        return torch.ones((1, 1), dtype=torch.float32, device=dev)
    if n & (n - 1) != 0:
        raise ValueError(f"Hadamard requires power-of-two n, got {n}")
    h = torch.tensor([[1.0]], dtype=torch.float32, device=dev)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(n)


def block_hadamard_transform(
    tensor: torch.Tensor,
    block_size: int = 32,
    *,
    dim: int = -1,
) -> torch.Tensor:
    """Rotate consecutive blocks along ``dim`` and return float32 on the input device.

    Only the block size must be a power of two. A width such as 5120 is valid
    with 32-wide blocks; no dense 5120-by-5120 rotation is materialized.
    """
    if tensor.ndim == 0 or not -tensor.ndim <= dim < tensor.ndim:
        raise ValueError(f"Invalid rotation dimension {dim} for shape {tuple(tensor.shape)}")
    rotation = hadamard_matrix(block_size, device=tensor.device)
    values = tensor.float().movedim(dim, -1)
    width = values.shape[-1]
    if width % block_size:
        raise ValueError(f"Width {width} must be divisible by block_size {block_size}")
    rotated = values.reshape(-1, block_size) @ rotation.T
    return rotated.reshape(values.shape).movedim(-1, dim)


def random_orthogonal(
    n: int,
    *,
    device: torch.device | str = "cpu",
    seed: int | None = None,
) -> torch.Tensor:
    """Sample a Haar-uniform orthogonal matrix using sign-corrected Gaussian QR."""
    if n <= 0:
        raise ValueError("Orthogonal matrix size must be positive")
    dev = torch.device(device)
    g = None
    if seed is not None:
        g = torch.Generator(device=dev).manual_seed(seed)
    a = torch.randn(n, n, dtype=torch.float32, device=dev, generator=g)
    q, r = torch.linalg.qr(a)
    signs = r.diagonal().sign()
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return cast(torch.Tensor, q * signs.unsqueeze(0))


def apply_weight_rotation(
    weight: torch.Tensor,
    rotation: torch.Tensor,
    *,
    dim: int = -1,
) -> torch.Tensor:
    """Rotate the input or output coordinates of a two-dimensional weight matrix.

    For a [out, in] weight W, rotating input activations x by R means the
    equivalent weight is W @ R^T (so that (W R^T)(R x) = W x).
    Rotating output coordinates (``dim=0``) gives R @ W.
    """
    if weight.ndim != 2 or rotation.ndim != 2 or rotation.shape[0] != rotation.shape[1]:
        raise ValueError("Expected a weight matrix and a square rotation matrix")
    rot_t = rotation.T
    if dim in (-1, 1):
        return weight @ rot_t
    if dim == 0:
        return rotation @ weight
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
