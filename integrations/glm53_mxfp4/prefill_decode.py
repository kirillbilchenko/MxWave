"""Caller-owned CUDA destinations for the isolated integer-only decoder experiment."""

from __future__ import annotations

import ctypes
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from zipserv_contract import require_qualified_exponent_base

MODES = {"stock": 0, "direct_pair": 1, "row_coalesced": 2}


class NativeDecoder:
    """Bind a separate experimental library without replacing the serving runtime."""

    def __init__(self, build: Path) -> None:
        manifest = json.loads((build / "native-build.json").read_bytes())
        path = build / manifest["library"]
        if (manifest["status"] != "built_pending_gpu_qualification"
                or hashlib.sha256(path.read_bytes()).hexdigest() != manifest["library_sha256"]):
            raise ValueError("Decoder build or library binding changed")
        self.manifest = manifest
        self.library = ctypes.CDLL(str(path))
        self.library.mxwave_prefill_decode.argtypes = [ctypes.c_int] + [ctypes.c_void_p] * 8 + [
            ctypes.c_int] * 3 + [ctypes.c_void_p] + [ctypes.c_int] * 2
        self.library.mxwave_prefill_decode.restype = ctypes.c_int
        self.library.mxwave_prefill_resources.argtypes = [ctypes.c_int] * 3 + [
            ctypes.POINTER(ctypes.c_int)]
        self.library.mxwave_prefill_resources.restype = ctypes.c_int

    def decode(self, mode: str, packed: Any, output: torch.Tensor) -> None:
        """Reconstruct exact BF16 words on the caller's current stream without allocation."""
        if (mode not in MODES or output.device.type != "cuda" or output.dtype != torch.bfloat16
                or not output.is_contiguous() or output.shape != (packed.m, packed.k)
                or not 0 <= packed.start_exp <= 255
                or any(t.device != output.device for t in packed.buffers)):
            raise ValueError("Require matching CUDA BF16 buffers and an eight-bit exponent base")
        if mode == "stock":
            require_qualified_exponent_base(packed.start_exp, where="unchanged decoder control")
        code = self.library.mxwave_prefill_decode(
            MODES[mode], torch.cuda.current_stream(output.device).cuda_stream,
            *(t.data_ptr() for t in packed.buffers), packed.max_common, packed.max_rare,
            packed.start_exp, output.data_ptr(), packed.m, packed.k)
        if code:
            raise RuntimeError(f"Experimental decoder CUDA error {code}")

    def resources(self, mode: str, packed: Any) -> dict[str, int]:
        """Report actual compiler registers, dynamic shared storage and occupancy bound."""
        result = (ctypes.c_int * 5)()
        code = self.library.mxwave_prefill_resources(
            MODES[mode], packed.max_common, packed.max_rare, result)
        if code:
            raise RuntimeError(f"Decoder resource query CUDA error {code}")
        return dict(zip(("dynamic_shared_bytes", "registers_per_thread", "static_shared_bytes",
                         "max_blocks_per_sm", "threads_per_block"), result, strict=True))
