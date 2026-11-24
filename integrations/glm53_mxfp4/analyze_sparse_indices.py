"""Check sparse MLA slot membership independently of the runtime's compaction order."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def check_slot_multisets(
    request_ids: NDArray[np.int32],
    block_table: NDArray[np.int32],
    token_indices: NDArray[np.int32],
    slots: NDArray[np.int32],
    valid_counts: NDArray[np.int32],
    block_size: int,
) -> dict[str, Any]:
    """Recompute TP-only physical slots, preserving duplicates and checking the entire tail."""
    arrays = (request_ids, block_table, token_indices, slots, valid_counts)
    if block_size <= 0 or any(not np.issubdtype(a.dtype, np.signedinteger) for a in arrays):
        raise ValueError("Require signed integer indices and a positive block size")
    if (
        request_ids.ndim != 1
        or block_table.ndim != 2
        or token_indices.ndim != 2
        or slots.shape != token_indices.shape
        or request_ids.shape != valid_counts.shape
        or request_ids.size != token_indices.shape[0]
    ):
        raise ValueError("Sparse conversion shapes disagree")
    if np.any(request_ids < 0) or np.any(request_ids >= block_table.shape[0]):
        raise ValueError("Request index exceeds the retained block table")
    width = token_indices.shape[1]
    if np.any(valid_counts < 0) or np.any(valid_counts > width):
        raise ValueError("Valid count exceeds the returned slot row")
    rows = []
    for row, request in enumerate(request_ids):
        tokens = token_indices[row].astype(np.int64)
        blocks = tokens // block_size
        valid = (tokens >= 0) & (blocks < block_table.shape[1])
        expected = (
            block_table[int(request), blocks[valid]].astype(np.int64) * block_size
            + tokens[valid] % block_size
        )
        count = int(valid_counts[row])
        current = slots[row, :count].astype(np.int64)
        membership = count == expected.size and np.array_equal(
            np.sort(current), np.sort(expected)
        )
        rows.append(
            {
                "row": row,
                "expected_count": int(expected.size),
                "recorded_count": count,
                "multiset_matches": bool(membership),
                "order_matches_input_columns": bool(np.array_equal(current, expected)),
                "tail_is_minus_one": bool(np.all(slots[row, count:] == -1)),
            }
        )
    return {
        "rows": rows,
        "all_multisets_match": all(r["multiset_matches"] for r in rows),
        "all_tails_are_minus_one": all(r["tail_is_minus_one"] for r in rows),
        "all_orders_match_input_columns": all(r["order_matches_input_columns"] for r in rows),
        "DCP_world_size": 1,
        "limits": "Membership and order checks do not establish attention or inference equivalence.",
    }
