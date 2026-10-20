"""Retire transient initialization stream leases around the pinned native graph capture."""

from __future__ import annotations

import functools
import hashlib
import inspect
import os
from pathlib import Path
from typing import Any

NATIVE_SHA = "c183937e6eb5b9c28c79d98fb4c64f562e7649d5f6d65743e6640b2f378ecf9f"
_INSTALLED = False
_RECORDS: list[dict[str, Any]] = []


def patch(manager_type: Any, runtime: Any) -> None:
    """Keep native capture and model math intact; retire only completed non-graph leases."""
    original = manager_type.capture

    @functools.wraps(original)
    def capture(manager: Any, *args: Any, **kwargs: Any) -> Any:
        before = runtime.retire_idle_eager_slots()
        result = original(manager, *args, **kwargs)
        after = runtime.retire_idle_eager_slots()
        _RECORDS.append({"before": before, "after": after,
            "captured_token_counts": sorted({d.num_tokens for d in manager.graphs}),
            "startup_only": True, "pool_allocation_changed": False})
        return result

    manager_type.capture = capture


def install() -> None:
    """Opt into the source-bound lifecycle only for the compressed worker process."""
    global _INSTALLED
    if _INSTALLED or os.environ.get("MXWAVE_GLM53_KDA_MODE") != "packed":
        return
    import kda_runtime
    from vllm.v1.worker.gpu import cudagraph_utils

    if hashlib.sha256(Path(inspect.getfile(cudagraph_utils)).read_bytes()).hexdigest() != NATIVE_SHA:
        raise ValueError("KDA capture lifecycle requires the audited native graph manager")
    patch(cudagraph_utils.ModelCudaGraphManager, kda_runtime)
    _INSTALLED = True


def report() -> dict[str, Any]:
    """Expose completed startup transitions without clearing or changing any live graph storage."""
    return {"installed": _INSTALLED, "native_capture_sha256": NATIVE_SHA,
            "transitions": list(_RECORDS), "pool_slots": 2, "maximum_pool_bytes": 256 * 1024**2,
            "normal_inference_synchronization_added": False}
