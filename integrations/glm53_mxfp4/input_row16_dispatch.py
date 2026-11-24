"""Opt in the qualified exact16 input route only inside a pinned disposable model."""

from __future__ import annotations

import functools
import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

RUNTIME_SHA256 = "f9ed82f5445c61990aa981058b0dcc03d31b4fce363a156c0be4db7bcd3c3753"
INPUT_GEOMETRY = (12608, 4096, 12576)
COMPONENT_EVIDENCE_SHA256 = "76b00c670bf2747448afe35ca9e7f011747b5c3d17ac0b6fca9caa930936b34c"


def select_route(
    native: Callable[..., bool],
    tokens: int,
    native_rows: int,
    width: int,
    logical_rows: int,
    *,
    limit: int,
    output_sixteen: bool,
) -> bool:
    """Keep the original decision and add only the measured exact16 input geometry."""
    original = native(
        tokens, native_rows, width, logical_rows, limit=limit, output_sixteen=output_sixteen
    )
    return original or (
        limit == 8 and tokens == 16 and (native_rows, width, logical_rows) == INPUT_GEOMETRY
    )


def install() -> dict[str, Any]:
    """Install a route wrapper in an isolated frozen child without editing its runtime."""
    import kda_runtime as runtime

    if getattr(runtime.use_fused, "_mxwave_input_sixteen", False):
        return policy()
    if hashlib.sha256(Path(runtime.__file__).read_bytes()).hexdigest() != RUNTIME_SHA256:
        raise ValueError("Require the qualified frozen output16 parent runtime")
    if runtime.FUSED_MAX_TOKENS != 8 or not runtime.OUTPUT_FUSED_16:
        raise ValueError("Require the eight-row limit and qualified exact16 output policy")
    original_route, original_prepare = runtime.use_fused, runtime.prepare

    @functools.wraps(original_route)
    def route(
        tokens: int,
        native_rows: int,
        width: int,
        logical_rows: int,
        *,
        limit: int,
        output_sixteen: bool,
    ) -> bool:
        return select_route(
            original_route,
            tokens,
            native_rows,
            width,
            logical_rows,
            limit=limit,
            output_sixteen=output_sixteen,
        )

    @functools.wraps(original_prepare)
    def prepare(layer: Any, target: Any, max_words: int) -> None:
        original_prepare(layer, target, max_words)
        if target.kind == "input":
            if (target.native_rows, target.width, target.rows) != INPUT_GEOMETRY:
                raise ValueError("Input projection differs from the measured geometry")
            layer.mxwave_kda_qualification["fused_exact_rows"] = [16]

    route._mxwave_input_sixteen = True  # type: ignore[attr-defined]
    runtime.use_fused = route
    runtime.prepare = prepare
    return policy()


def policy() -> dict[str, Any]:
    """Describe this explicit experiment without claiming model qualification."""
    return {
        "global_fused_limit": 8,
        "input_exact_rows": [16],
        "input_native_geometry": list(INPUT_GEOMETRY),
        "runtime_source_sha256": RUNTIME_SHA256,
        "component_evidence_sha256": COMPONENT_EVIDENCE_SHA256,
        "intermediate_rows_qualified": False,
        "resident_BF16_cache_bytes": 0,
        "native_kernel_changed": False,
        "quality_qualified": False,
    }
