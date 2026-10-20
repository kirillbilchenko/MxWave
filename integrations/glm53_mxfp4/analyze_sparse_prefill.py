"""Separate consumed-score changes, selected membership and ordering in sparse prefill."""

from __future__ import annotations

from typing import Any

import numpy as np


def masked_score_difference(
    first: np.ndarray, second: np.ndarray, starts: np.ndarray, ends: np.ndarray,
) -> dict[str, Any]:
    """Compare only score columns consumed by each row's native causal bounds."""
    if (first.shape != second.shape or first.dtype != np.float32 or second.dtype != np.float32
            or first.ndim != 2 or starts.shape != (len(first),) or ends.shape != starts.shape
            or not np.issubdtype(starts.dtype, np.integer)
            or not np.issubdtype(ends.dtype, np.integer)):
        raise ValueError("Require matching FP32 score matrices and integer row bounds")
    if np.any(starts < 0) or np.any(ends < starts) or np.any(ends > first.shape[1]):
        raise ValueError("Score consumption bounds exceed the retained matrix")
    count = rows = nonfinite = 0
    first_index = None
    maximum = 0.0
    for row, (start, end) in enumerate(zip(starts, ends, strict=True)):
        a, b = first[row, start:end], second[row, start:end]
        changed = a.view(np.uint32) != b.view(np.uint32)
        n = int(np.count_nonzero(changed))
        count += n
        rows += int(n > 0)
        if n and first_index is None:
            first_index = [row, int(start + np.flatnonzero(changed)[0])]
        finite = np.isfinite(a) & np.isfinite(b)
        nonfinite += int(np.count_nonzero(~finite))
        if finite.any():
            maximum = max(maximum, float(np.max(np.abs(a[finite].astype(np.float64) - b[finite]))))
    return {"consumed_elements": int((ends - starts).sum()), "different_elements": count,
            "different_rows": rows, "first_index": first_index,
            "nonfinite_pairs": nonfinite, "maximum_absolute_error_finite": maximum,
            "unused_columns_excluded": True, "exact": count == 0}


def selection_difference(
    first: np.ndarray, second: np.ndarray, upper: np.ndarray,
) -> dict[str, Any]:
    """Distinguish order from selected multiplicity and report invalid or duplicate indices."""
    if (first.shape != second.shape or first.ndim != 2 or upper.shape != (len(first),)
            or not np.issubdtype(first.dtype, np.integer)
            or not np.issubdtype(second.dtype, np.integer)
            or not np.issubdtype(upper.dtype, np.integer) or np.any(upper < 0)):
        raise ValueError("Require matching integer selections and exclusive per-row causal limits")
    ordered = np.any(first != second, axis=1)
    normalized = [np.sort(np.where(value >= 0, value, -1), axis=1) for value in (first, second)]
    membership = np.any(normalized[0] != normalized[1], axis=1)
    inventories = []
    for value, sorted_value in zip((first, second), normalized, strict=True):
        duplicates = (sorted_value[:, 1:] >= 0) & (sorted_value[:, 1:] == sorted_value[:, :-1])
        inventories.append({"selected": int(np.count_nonzero(value >= 0)),
                            "non_sentinel_negative": int(np.count_nonzero(value < -1)),
                            "causal_or_extent_violations": int(np.count_nonzero(value >= upper[:, None])),
                            "duplicate_entries": int(np.count_nonzero(duplicates)),
                            "rows_with_duplicates": int(np.count_nonzero(np.any(duplicates, axis=1)))})
    return {"different_order_rows": int(np.count_nonzero(ordered)),
            "different_multiset_rows": int(np.count_nonzero(membership)),
            "order_only_rows": int(np.count_nonzero(ordered & ~membership)),
            "first_different_order_row": int(np.flatnonzero(ordered)[0]) if ordered.any() else None,
            "first_different_multiset_row": int(np.flatnonzero(membership)[0]) if membership.any() else None,
            "inventories": inventories,
            "multiset_comparison_preserves_duplicate_multiplicity": True}
