"""Bound the exploratory FP32 input route to its measured native geometry and row counts."""

from __future__ import annotations

from typing import Any

INPUT_GEOMETRY = (12608, 4096, 12576)
ROWS = (1, 4, 8, 16)
SPLIT = 2
RUNTIME_SHA256 = "f9ed82f5445c61990aa981058b0dcc03d31b4fce363a156c0be4db7bcd3c3753"
LIBRARY_SHA256 = "20458fa202200f895d5a95edbe1cd85f0d71927485ba102d616718c4957f8d10"
BUILD_SHA256 = "823c001ccaed7a349ff56f747934603bd8809fb2665cb8185623eb653296384b"
PRECISION_ARCHIVE_SHA256 = "e40d606164bbf8ead44a3f932f6f615d1d7b4be70d60a6a17facf0ce571ae94b"
SPEED_ARCHIVE_SHA256 = "55dde1aedfa17bdf97cb607a51d318edb7c3ae0202ab3081335657cb8aa360cc"


def selected(tokens: int, m: int, k: int, rows: int) -> bool:
    """Route only the source-bound input geometry and explicitly measured physical sizes."""
    return (m, k, rows) == INPUT_GEOMETRY and tokens in ROWS


def scratch_bytes(tokens: int) -> int:
    """Include every FP32 partial for native row padding and the fixed split factor."""
    if tokens not in ROWS:
        raise ValueError("FP32 input workspace requested for untested rows")
    padded = 8 if tokens <= 8 else 16
    return INPUT_GEOMETRY[0] * padded * SPLIT * 4


def policy() -> dict[str, Any]:
    """Retain prior failed gates and separate component evidence from model qualification."""
    return {"revision": "fp32inputv1", "input_geometry": list(INPUT_GEOMETRY),
        "physical_rows": list(ROWS), "split": SPLIT, "partial_dtype": "float32",
        "runtime_sha256": RUNTIME_SHA256, "library_sha256": LIBRARY_SHA256,
        "native_build_sha256": BUILD_SHA256,
        "precision_archive_sha256": PRECISION_ARCHIVE_SHA256,
        "speed_archive_sha256": SPEED_ARCHIVE_SHA256,
        "workspace_reuses_existing_two_slot_pool": True,
        "maximum_workspace_bytes_per_slot": scratch_bytes(16),
        "additional_resident_scratch_bytes": 0, "resident_BF16_cache_bytes": 0,
        "output_route_changed": False, "untested_rows_keep_qualified_route": True,
        "parent_exact_word_screen_passed": False,
        "parent_input_timing_conditions_passed": 46, "parent_input_timing_conditions_total": 48,
        "model_quality_or_serving_speed_qualified": False, "automatic_promotion_allowed": False}
