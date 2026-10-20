"""Canonicalize native selected pool indices without changing scores or membership."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import torch


def canonical_topk(
    native: Callable[..., Any], scores: torch.Tensor, starts: torch.Tensor,
    ends: torch.Tensor, output: torch.Tensor, rows: int, stride0: int, stride1: int, k: int,
) -> Any:
    """Keep native selection and return semantics, then sort the selected integer slots."""
    if (not 1 <= rows <= 4096 or k != 512 or output.shape != (rows, k)
            or output.dtype != torch.int32 or scores.dtype != torch.float32
            or scores.ndim != 2 or scores.shape[0] != rows
            or scores.stride() != (stride0, stride1)
            or starts.shape != (rows,) or ends.shape != (rows,)
            or starts.dtype != torch.int32 or ends.dtype != torch.int32
            or len({scores.device, starts.device, ends.device, output.device}) != 1):
        raise ValueError("Unqualified native sparse-prefill top-k layout")
    result = native(scores, starts, ends, output, rows, stride0, stride1, k)
    # Invalid -1 slots sort first. The native pool expansion masks every negative
    # index, and the existing stable compaction drops them regardless of position.
    # Sorting integer values preserves multiplicity, including all padding slots.
    output.copy_(torch.sort(output, dim=-1).values)
    return result


def install() -> dict[str, Any]:
    """Wrap only native prefill selection; retain the existing decode operator."""
    namespace = torch.ops._C
    native = namespace.top_k_per_row_prefill
    if not getattr(native, "_mxwave_canonical_prefill", False):
        @functools.wraps(native)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            return canonical_topk(native, *args, **kwargs)

        wrapped._mxwave_canonical_prefill = True  # type: ignore[attr-defined]
        namespace.top_k_per_row_prefill = wrapped
    return {"status": "installed", "policy": "ascending selected pool indices, -1 preserved",
            "max_rows": 4096, "k": 512, "new_kernel": False,
            "scores_or_selected_membership_changed": False,
            "scope": "native top_k_per_row_prefill; bounded diagnostic only"}
