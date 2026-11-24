"""Bounded component interface for the separately compiled grouped MoE kernels.

This is not a vLLM engine plugin. Callers own every input and scratch tensor
until CUDA completion, including during graph replay and on non-default streams.
"""

from __future__ import annotations

import ctypes
import runpy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class PackedScales:
    """Exact records for a contiguous [experts, groups, output] E8M0 tensor."""

    shape: tuple[int, ...]
    records: torch.Tensor
    exceptions: torch.Tensor
    descriptor: torch.Tensor
    encoding: str = "records"
    affine_palette: bool = False
    stage_metadata_verified: bool = False
    aligned_nibble_base: int | None = None

    @property
    def retained_bytes(self) -> int:
        """Include all record, exception and descriptor tensor payloads."""
        return sum(t.numel() * t.element_size()
                   for t in (self.records, self.exceptions, self.descriptor))


def sparse_cpu_pack(raw: Any) -> Any:
    """Pack flat U8 scale bytes into exact sparse-palette buffers on the CPU."""
    from scale_tiles import pack_sparse

    return pack_sparse(raw)


def sparse_cpu_verify(packed: Any, raw: Any) -> None:
    """Prove on the CPU that the sparse buffers restore the source bytes."""
    import numpy as np
    from scale_tiles import unpack_sparse

    if not np.array_equal(unpack_sparse(packed), raw):
        raise ValueError("Sparse scale CPU round-trip changed bytes")


def sparse_stage(raw_min: int, raw_max: int, descriptor: int) -> tuple[int, int | None]:
    """Staging policy and nibble base from the source value range alone.

    sparse_stage_descriptor reads only the minimum and maximum of the source bytes, so a
    two-value stand-in with the same range yields exactly the same result.
    """
    import numpy as np
    from moe_scale_policy import sparse_stage_descriptor

    if not 0 <= raw_min <= raw_max <= 255:
        raise ValueError("Invalid source scale range")
    return sparse_stage_descriptor(np.array([raw_min, raw_max] * 8, dtype=np.uint8), descriptor)


def sparse_upload(
    selectors: Any, exceptions: Any, descriptor: int, shape: tuple[int, ...],
    device: torch.device, policy: int, base: int | None,
) -> PackedScales:
    """Move sparse buffers to the device and build a descriptor over the new pointers."""
    import numpy as np

    records = torch.from_numpy(selectors.view(np.int32)).to(device)
    exceptions_gpu = torch.from_numpy(exceptions).to(device)
    table = torch.tensor([records.data_ptr(), exceptions_gpu.data_ptr(), policy],
                         dtype=torch.int64, device=device)
    return PackedScales(shape, records, exceptions_gpu, table, "sparse_palette",
                        bool(descriptor & (1 << 24)), True, base)


def sparse_upload_ranged(
    selectors: Any, exceptions: Any, descriptor: int, shape: tuple[int, ...],
    device: torch.device, raw_min: int, raw_max: int,
) -> PackedScales:
    """sparse_upload with the staging policy derived from the source value range."""
    policy, base = sparse_stage(raw_min, raw_max, descriptor)
    return sparse_upload(selectors, exceptions, descriptor, shape, device, policy, base)


def check_scales(scales: torch.Tensor) -> None:
    """Reject anything but contiguous 3D CUDA E8M0 scales in whole 16-byte vectors."""
    if (scales.device.type != "cuda" or scales.ndim != 3 or not scales.is_contiguous()
            or scales.dtype != torch.float8_e8m0fnu or scales.numel() % 16):
        raise ValueError("Expected contiguous 3D CUDA E8M0 scales in whole 16-byte vectors")


def pack_scales(
    scales: torch.Tensor, codec_path: Path, *, encoding: str = "records"
) -> PackedScales:
    """Pack on CPU and retain only the exact fixed records on the original GPU."""
    check_scales(scales)
    import numpy as np

    if encoding == "sparse_palette":
        from moe_scale_policy import sparse_stage_descriptor

        raw = scales.view(torch.uint8).cpu().numpy().reshape(-1)
        packed = sparse_cpu_pack(raw)
        sparse_cpu_verify(packed, raw)
        policy, base = sparse_stage_descriptor(raw, packed.descriptor)
        return sparse_upload(packed.selectors, packed.exceptions, packed.descriptor,
                             tuple(scales.shape), scales.device, policy, base)
    if encoding in ("nibbles", "aligned_nibbles"):
        raw = scales.view(torch.uint8).cpu().numpy().reshape(-1)
        if raw.size == 0:
            raise ValueError("Cannot choose a base for empty scales")
        base, upper = int(raw.min()), int(raw.max())
        if encoding == "aligned_nibbles":
            base &= ~15
        if upper - base > 15:
            raise ValueError("Exact nibble scales exceed the selected 16-value exponent range")
        order = np.array([0, 1, 4, 5, 2, 3, 6, 7])
        shifts = np.arange(8, dtype=np.uint32) * 4
        packed_records = np.empty((raw.size // 16, 2), dtype=np.uint32)
        for start in range(0, raw.size, 1024**2):
            chunk = raw[start : start + 1024**2].reshape(-1, 2, 8)
            delta = chunk.astype(np.uint32) - base
            packed = np.bitwise_or.reduce(delta[:, :, order] << shifts, axis=-1)
            decoded = ((packed[:, :, None] >> shifts) & 15).astype(np.uint8)
            decoded = decoded[:, :, order] + base
            if not np.array_equal(decoded, chunk):
                raise ValueError("Exact nibble CPU round-trip changed scale bytes")
            packed_records[start // 16 : (start + chunk.size) // 16] = packed
        records = torch.from_numpy(packed_records.view(np.int32)).to(scales.device)
        exceptions = torch.empty((0, 16), dtype=torch.uint8, device=scales.device)
        descriptor = torch.tensor([records.data_ptr(), base], dtype=torch.int64, device=scales.device)
        return PackedScales(tuple(scales.shape), records, exceptions, descriptor, encoding)
    if encoding != "records":
        raise ValueError("Unsupported exact scale encoding")
    codec = runpy.run_path(str(codec_path))
    raw = scales.view(torch.uint8).cpu().numpy().reshape(-1)
    # Bound the codec's U32 expansion scratch, including on complete 288-expert tensors.
    packed_records = np.empty((raw.size // 16, 2), dtype=np.uint32)
    exception_chunks = []
    exceptions_seen = 0
    for start in range(0, raw.size, 1024**2):
        chunk = raw[start : start + 1024**2].tobytes()
        packed = codec["pack_records"](chunk)
        if codec["unpack_records"](packed) != chunk:
            raise ValueError("Scale record CPU round-trip changed source bits")
        escapes = packed.records[:, 1] == 256
        if exceptions_seen + len(packed.exceptions) > 2**32 - 1:
            raise ValueError("Global scale exception table exceeds its U32 index")
        packed.records[escapes, 0] += exceptions_seen
        packed_records[start // 16 : (start + len(chunk)) // 16] = packed.records
        exception_chunks.append(packed.exceptions)
        exceptions_seen += len(packed.exceptions)
    packed_exceptions = np.concatenate(exception_chunks, axis=0)
    records = torch.from_numpy(packed_records.view(np.int32)).to(scales.device)
    exceptions = torch.from_numpy(packed_exceptions).to(scales.device)
    descriptor = torch.tensor([records.data_ptr(), exceptions.data_ptr()],
                              dtype=torch.int64, device=scales.device)
    return PackedScales(tuple(scales.shape), records, exceptions, descriptor)


@dataclass
class Buffers:
    """Output, FP32 partial reduction and zero-initialized managed lock storage."""

    output: torch.Tensor
    temporary: torch.Tensor
    locks: torch.Tensor


def allocate_buffers(
    m: int, n: int, topk: int, block: int, sorted_capacity: int, device: torch.device
) -> Buffers:
    """Match stock's grouped GEMM temporary and lock bounds, allocating once."""
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    count = min(n * sorted_capacity, sms * 4 * block * 256)
    if block == 8:
        count *= 2
    return Buffers(
        torch.empty((m * topk, n), dtype=torch.bfloat16, device=device),
        torch.empty(count, dtype=torch.float32, device=device),
        torch.zeros(sms * 4, dtype=torch.int32, device=device),
    )


class NativeMoe:
    """Launch grouped BF16/MXFP4 GEMMs on the caller's current CUDA stream."""

    def __init__(self, path: Path) -> None:
        """Load one raw or compressed-scale variant from its isolated library."""
        self.library = ctypes.CDLL(str(path.resolve()))
        self.library.mxwave_moe_compressed.restype = ctypes.c_int
        self.compressed = bool(self.library.mxwave_moe_compressed())
        self.affine_only = False
        if hasattr(self.library, "mxwave_moe_affine_only"):
            self.library.mxwave_moe_affine_only.restype = ctypes.c_int
            self.affine_only = bool(self.library.mxwave_moe_affine_only())
        self.shared_stage = False
        if hasattr(self.library, "mxwave_moe_shared_stage"):
            self.library.mxwave_moe_shared_stage.restype = ctypes.c_int
            self.shared_stage = bool(self.library.mxwave_moe_shared_stage())
        self.nibble_stage = False
        if hasattr(self.library, "mxwave_moe_nibble_stage"):
            self.library.mxwave_moe_nibble_stage.restype = ctypes.c_int
            self.nibble_stage = bool(self.library.mxwave_moe_nibble_stage())
        self.static_nibble_stage = False
        if hasattr(self.library, "mxwave_moe_static_nibble_stage"):
            self.library.mxwave_moe_static_nibble_stage.restype = ctypes.c_int
            self.static_nibble_stage = bool(self.library.mxwave_moe_static_nibble_stage())
        self.u8_fallback: NativeMoe | None = None
        self.encoding = "records"
        if hasattr(self.library, "mxwave_moe_encoding"):
            self.library.mxwave_moe_encoding.restype = ctypes.c_int
            encodings = {0: "records", 1: "nibbles", 2: "aligned_nibbles", 3: "sparse_palette"}
            self.encoding = encodings[self.library.mxwave_moe_encoding()]
        self.library.mxwave_moe_last_error.restype = ctypes.c_char_p
        self.library.mxwave_moe_plan.argtypes = [ctypes.c_int] * 7 + [ctypes.POINTER(ctypes.c_int)]
        self.library.mxwave_moe_plan.restype = ctypes.c_int
        self.library.mxwave_moe_run.argtypes = (
            [ctypes.c_void_p] * 10 + [ctypes.c_int] * 11 + [ctypes.c_size_t]
        )
        self.library.mxwave_moe_run.restype = ctypes.c_int
        self.library.mxwave_moe_decode.argtypes = (
            [ctypes.c_void_p] * 2 + [ctypes.c_int, ctypes.c_size_t]
        )
        self.library.mxwave_moe_decode.restype = ctypes.c_int
        self.has_bf16_decoder = hasattr(self.library, "mxwave_moe_decode_bf16")
        if self.has_bf16_decoder:
            self.library.mxwave_moe_decode_bf16.argtypes = self.library.mxwave_moe_decode.argtypes
            self.library.mxwave_moe_decode_bf16.restype = ctypes.c_int

    def plan(
        self, m: int, n: int, k: int, block: int, topk: int, experts: int, device: int
    ) -> tuple[int, int, int]:
        """Choose stock's tile once; use this same plan for both compared variants."""
        values = (ctypes.c_int * 3)()
        if self.library.mxwave_moe_plan(m, n, k, block, topk, experts, device, values):
            raise RuntimeError("No supported grouped GEMM plan")
        return values[0], values[1], values[2]

    def set_u8_fallback(self, fallback: NativeMoe) -> None:
        """Bind an exact U8-stage consumer outside capture and all timed launches."""
        if (not self.static_nibble_stage or fallback is self or fallback.nibble_stage
                or not fallback.shared_stage or fallback.encoding != self.encoding):
            raise ValueError("Static nibble staging requires a separate exact U8-stage fallback")
        self.u8_fallback = fallback

    def scale_consumer(self, scales: PackedScales) -> NativeMoe:
        """Select a verified tensor layout once, rejecting a missing exact fallback."""
        if self.nibble_stage and not scales.stage_metadata_verified:
            raise ValueError("Nibble staging requires verified complete source ranges")
        if self.static_nibble_stage and scales.aligned_nibble_base is None:
            if self.u8_fallback is None:
                raise ValueError("Unsupported range requires an explicitly bound exact U8 fallback")
            return self.u8_fallback
        return self

    def decode(self, scales: PackedScales) -> torch.Tensor:
        """Check the GPU record decoder independently of routing or GEMM arithmetic."""
        if scales.encoding != self.encoding:
            raise ValueError("Native library and scale encoding differ")
        if self.nibble_stage and not scales.stage_metadata_verified:
            raise ValueError("Nibble staging requires verified complete source ranges")
        output = torch.empty(scales.shape, dtype=torch.uint8, device=scales.descriptor.device)
        if self.library.mxwave_moe_decode(
            scales.descriptor.data_ptr(), output.data_ptr(), output.numel(),
            torch.cuda.current_stream(output.device).cuda_stream,
        ):
            raise RuntimeError("Grouped scale decoding launch failed")
        return output

    def decode_bf16(self, scales: PackedScales) -> torch.Tensor:
        """Expose the exact stock BF16 fragment bits for independent decoder checks."""
        if not self.has_bf16_decoder:
            raise ValueError("This native library has no BF16 fragment decoder")
        if scales.encoding != self.encoding:
            raise ValueError("Native library and scale encoding differ")
        if self.nibble_stage and not scales.stage_metadata_verified:
            raise ValueError("Nibble staging requires verified complete source ranges")
        shape = (*scales.shape[:-1], scales.shape[-1] * 2)
        output = torch.empty(shape, dtype=torch.uint8, device=scales.descriptor.device)
        if self.library.mxwave_moe_decode_bf16(
            scales.descriptor.data_ptr(), output.data_ptr(), output.numel() // 2,
            torch.cuda.current_stream(output.device).cuda_stream,
        ):
            raise RuntimeError("Grouped BF16 fragment decoding launch failed")
        return output

    def bind(
        self, x: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor | PackedScales,
        buffers: Buffers, routing: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        routing_weights: torch.Tensor, topk: int, block: int, multiply: bool,
        plan: tuple[int, int, int],
    ) -> Any:
        """Validate all static buffers once and return an allocation-free launch closure."""
        shape = tuple(scales.shape)
        if len(shape) != 3:
            raise ValueError("Expected [experts, groups, output] scale storage")
        e, groups, n = shape
        k = groups * 32
        m = x.shape[0]
        if min(e, m, n, k, topk) < 1 or n % 64 or k % 64 or block not in (8, 16, 32, 48, 64):
            raise ValueError("Unsupported grouped GEMM shape or block size")
        if self.compressed != isinstance(scales, PackedScales):
            raise ValueError("Native variant and scale representation differ")
        if isinstance(scales, PackedScales) and scales.encoding != self.encoding:
            raise ValueError("Native library and packed scale encoding differ")
        if self.affine_only and (not isinstance(scales, PackedScales) or not scales.affine_palette):
            raise ValueError("Affine kernel requires a verified consecutive scale palette")
        if self.nibble_stage and (not isinstance(scales, PackedScales)
                                  or not scales.stage_metadata_verified):
            raise ValueError("Nibble staging requires verified complete source ranges")
        if isinstance(scales, PackedScales):
            consumer = self.scale_consumer(scales)
            if consumer is not self:
                return consumer.bind(x, weight, scales, buffers, routing, routing_weights,
                                     topk, block, multiply, plan)
        if isinstance(scales, torch.Tensor) and scales.dtype != torch.float8_e8m0fnu:
            raise ValueError("Raw scales must retain the runtime E8M0 dtype")
        storage = scales.descriptor if isinstance(scales, PackedScales) else scales
        if x.dtype != torch.bfloat16 or x.ndim != 2 or x.shape[1] != k:
            raise ValueError("Expected compatible BF16 input")
        if weight.dtype != torch.int32 or tuple(weight.shape) != (e, k // 16, n * 2):
            raise ValueError("Expected grouped stock-Marlin-repacked int32 weights")
        if tuple(buffers.output.shape) != (m * topk, n):
            raise ValueError("Native GEMM output shape is wrong")
        if (buffers.output.dtype != torch.bfloat16 or buffers.temporary.dtype != torch.float32
                or buffers.locks.dtype != torch.int32 or routing_weights.dtype != torch.float32
                or any(t.dtype != torch.int32 for t in routing)):
            raise ValueError("Native buffer dtype mismatch")
        sms = torch.cuda.get_device_properties(x.device).multi_processor_count
        needed = min(n * routing[0].numel(), sms * 4 * block * 256) * (2 if block == 8 else 1)
        if (buffers.temporary.numel() < needed or buffers.locks.numel() < sms * 4
                or routing_weights.numel() < m * topk or routing[2].numel() != 1):
            raise ValueError("Insufficient grouped GEMM scratch or routing metadata")
        tensors = (x, weight, storage, buffers.output, buffers.temporary, buffers.locks,
                   *routing, routing_weights)
        if x.device.type != "cuda" or any(t.device != x.device or not t.is_contiguous()
                                         for t in tensors):
            raise ValueError("All buffers must be contiguous on the same CUDA device")
        owners = (tensors, scales, buffers)

        def launch() -> torch.Tensor:
            _ = owners  # Keep every referenced allocation alive through the caller's CUDA work.
            status = self.library.mxwave_moe_run(
                *(t.data_ptr() for t in tensors), m, n, k, e, topk, block, int(multiply),
                x.device.index, *plan, torch.cuda.current_stream(x.device).cuda_stream,
            )
            if status:
                raise RuntimeError(self.library.mxwave_moe_last_error().decode())
            return buffers.output

        return launch
