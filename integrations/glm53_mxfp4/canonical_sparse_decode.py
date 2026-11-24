"""Canonicalize the observed eight-row flattened native decode selection."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import torch


def canonical_decode(
    native: Callable[..., Any], scores: torch.Tensor, next_n: int, lengths: torch.Tensor,
    output: torch.Tensor, rows: int, stride0: int, stride1: int, k: int,
) -> Any:
    """Sort only the qualified eight-row selection; leave other decode shapes native."""
    selected = rows == 8 and next_n == 1 and k == 512
    if selected and (
        scores.ndim != 2 or scores.shape[0] != 8 or scores.dtype != torch.float32
        or scores.stride() != (stride0, stride1) or output.shape != (8, 512)
        or output.dtype != torch.int32 or lengths.numel() != 8
        or lengths.dtype != torch.int32 or not lengths.is_contiguous()
        or len({scores.device, lengths.device, output.device}) != 1
    ):
        raise ValueError("Unqualified flattened eight-row decode layout")
    result = native(scores, next_n, lengths, output, rows, stride0, stride1, k)
    if selected:
        output.copy_(torch.sort(output, dim=-1).values)
    return result


def install() -> dict[str, Any]:
    """Install before graph capture without changing native selection, scores or weights."""
    namespace = torch.ops._C
    native = namespace.top_k_per_row_decode
    if not getattr(native, "_mxwave_canonical_decode", False):
        @functools.wraps(native)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            return canonical_decode(native, *args, **kwargs)

        wrapped._mxwave_canonical_decode = True  # type: ignore[attr-defined]
        namespace.top_k_per_row_decode = wrapped
    return {"status": "installed", "rows": 8, "next_n": 1, "k": 512,
            "policy": "ascending selected pool indices; preserve all -1 padding",
            "other_shapes": "native fallback", "scores_or_membership_changed": False,
            "scope": "C1 fixed-seven verification; no C8/C16 qualification", "new_kernel": False}
