"""Keep the measured output-only sixteen-row exception separate from the global fused limit."""

from __future__ import annotations


def use_fused(tokens: int, native_rows: int, width: int, logical_rows: int, *,
              limit: int, output_sixteen: bool) -> bool:
    """Select audited <=8 rows or exactly16 for this checkpoint's square output projection."""
    if limit not in (1, 8) or tokens <= 0:
        raise ValueError("Require positive rows and the audited one/eight-row fused limit")
    return tokens <= limit or (
        output_sixteen and tokens == 16
        and (native_rows, width, logical_rows) == (4096, 4096, 4096)
    )
