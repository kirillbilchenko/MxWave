"""Observe matched qualification cohorts while retaining the tested scratch lifecycle."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from typing import Any

from c8_trace_worker import C8TraceWorker
from high_concurrency_pool_worker import HighConcurrencyPoolWorker

MHC_LAYER_SHA = "0aa41ebd29fea15b91969125712f900edda1f9af0451db65d7e85d473943fc6e"


class MatchedC8Worker(C8TraceWorker, HighConcurrencyPoolWorker):
    """Combine host-only attribution with bounded captured scratch ownership."""

    def mxwave_native_mhc_policy(self) -> dict[str, Any]:
        """Require the unwrapped native mHC layer dispatch in this isolated comparison."""
        from vllm.model_executor.layers.mhc import MHCFusedPostPreOp

        path = Path(inspect.getfile(MHCFusedPostPreOp.forward_cuda))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != MHC_LAYER_SHA or hasattr(
            MHCFusedPostPreOp, "_mxwave_consistent_mhc_snapshot"
        ):
            raise ValueError("Matched qualification requires unchanged native mHC")
        return {"native_mhc_unchanged": True, "native_layer_sha256": digest}

    def mxwave_gpu_telemetry(self) -> dict[str, Any]:
        """Read clocks, thermals and process/container memory outside timed API groups."""
        from gpu_telemetry import snapshot

        result = snapshot()
        result["process_memory_kib"] = {
            key: int(value.split()[0])
            for line in Path("/proc/self/status").read_text().splitlines()
            if ":" in line
            for key, value in [line.split(":", 1)]
            if key in {"VmRSS", "VmHWM", "RssAnon", "RssFile", "RssShmem"}
        }
        for name in ("memory.current", "memory.peak"):
            path = Path("/sys/fs/cgroup") / name
            result["container_" + name.replace(".", "_") + "_bytes"] = (
                int(path.read_text()) if path.exists() else None
            )
        result["memory_accounting"] = (
            "Process RSS and container bytes; Spark shares physical CPU/GPU memory. "
            "Do not add these to Torch CUDA allocations."
        )
        return result
