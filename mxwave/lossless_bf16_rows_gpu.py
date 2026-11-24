"""Triton row lookup for the row-addressable BF16 embedding layout.

Only compressed buffers persist on the GPU. ``gather_rows`` reconstructs the
exact BF16 bit patterns of the requested rows into a bounded scratch buffer;
the full expanded table is never allocated. The kernel performs no host
synchronization beyond caller-requested validation, so repeated lookups can be
captured in a CUDA graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import torch
import triton  # type: ignore[import-not-found]
import triton.language as tl  # type: ignore[import-not-found]

from .lossless_bf16_rows import Bf16RowPacked

_ID_DTYPES: Final = (torch.int32, torch.int64)
_DEFAULT_STAGING_ROWS: Final = 2048


@triton.jit  # type: ignore[untyped-decorator]
def _row_gather_kernel(  # type: ignore[no-untyped-def]
    ids_ptr,
    base_ptr,
    row_offsets_ptr,
    block_offsets_ptr,
    sign_mantissa_ptr,
    planes_ptr,
    exceptions_ptr,
    out_ptr,
    WIDTH: tl.constexpr,
    BLOCK_WORDS: tl.constexpr,
    BLOCKS: tl.constexpr,
    PLANE_ROW: tl.constexpr,
) -> None:
    """Reconstruct one 512-word block of one requested row into BF16 bits."""
    slot = tl.program_id(0).to(tl.int64)
    blk = tl.program_id(1)
    row = tl.load(ids_ptr + slot).to(tl.int64)
    words = blk * BLOCK_WORDS + tl.arange(0, BLOCK_WORDS)
    byte = tl.arange(0, BLOCK_WORDS // 8)
    bit = tl.arange(0, 8)
    plane_base = planes_ptr + row * (3 * PLANE_ROW) + blk * (BLOCK_WORDS // 8)
    codes = tl.zeros((BLOCK_WORDS,), dtype=tl.int32)
    for plane in tl.static_range(3):
        raw = tl.load(plane_base + plane * PLANE_ROW + byte)
        expanded = (raw[:, None].to(tl.int32) >> bit[None, :]) & 1
        codes += tl.reshape(expanded, (BLOCK_WORDS,)) << plane
    rare = codes == 0
    base = tl.load(base_ptr + row).to(tl.int32)
    row_start = tl.load(row_offsets_ptr + row).to(tl.int64)
    block_start = tl.load(block_offsets_ptr + row * BLOCKS + blk).to(tl.int64)
    rank = tl.cumsum(rare.to(tl.int32), axis=0) - rare.to(tl.int32)
    exception = tl.load(
        exceptions_ptr + row_start + block_start + rank.to(tl.int64),
        mask=rare,
        other=0,
    ).to(tl.int32)
    exponent = tl.where(rare, exception, base + codes - 1)
    sm = tl.load(sign_mantissa_ptr + row * WIDTH + words).to(tl.int32)
    value = ((sm & 128) << 8) | (exponent << 7) | (sm & 127)
    bits16 = value.to(tl.uint16).to(tl.bfloat16, bitcast=True)
    tl.store(out_ptr + slot * WIDTH + words, bits16)


@dataclass(frozen=True)
class GpuBf16Rows:
    """Compressed GPU row buffers with no persistent expanded copy."""

    rows: int
    width: int
    block_size: int
    source_sha256: str
    base: torch.Tensor
    row_offsets: torch.Tensor
    block_offsets: torch.Tensor
    sign_mantissa: torch.Tensor
    planes: torch.Tensor
    exceptions: torch.Tensor

    @property
    def blocks_per_row(self) -> int:
        """Number of codec blocks in one row."""
        return self.width // self.block_size

    @property
    def resident_bytes(self) -> int:
        """Actual compressed tensor bytes on the GPU."""
        return sum(tensor.numel() * tensor.element_size() for tensor in self.tensors())

    def tensors(self) -> tuple[torch.Tensor, ...]:
        """The compressed GPU buffers in serialization order."""
        return (
            self.base,
            self.row_offsets,
            self.block_offsets,
            self.sign_mantissa,
            self.planes,
            self.exceptions,
        )


def upload_rows(
    packed: Bf16RowPacked,
    *,
    device: torch.device | str = "cuda",
    staging_rows: int = 2048,
) -> GpuBf16Rows:
    """Upload a validated row layout to CUDA without expanding it."""
    target = torch.device(device)
    if target.type != "cuda":
        raise ValueError("Triton row lookup requires a CUDA device")
    if staging_rows <= 0:
        raise ValueError("staging_rows must be positive")
    blocks = packed.blocks_per_row
    plane_row = packed.plane_bytes

    def allocate(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        return torch.empty(shape, dtype=dtype, device=target)

    buffers = {
        "base": allocate((packed.rows,), torch.uint8),
        "row_offsets": allocate((packed.rows + 1,), torch.uint32),
        "block_offsets": allocate((packed.rows, blocks), torch.uint32),
        "sign_mantissa": allocate((packed.rows * packed.width,), torch.uint8),
        "planes": allocate((packed.rows, 3 * plane_row), torch.uint8),
        "exceptions": allocate((len(packed.exceptions),), torch.uint8),
    }
    for name, buffer in buffers.items():
        source = np.ascontiguousarray(getattr(packed, name))
        flat = buffer.view(-1)
        per_row = source.size // packed.rows if packed.rows else 0
        chunk = max(1, staging_rows * per_row) or source.size or 1
        for start in range(0, source.size, chunk):
            stop = min(start + chunk, source.size)
            flat[start:stop].copy_(
                torch.from_numpy(source.reshape(-1)[start:stop]).to(target, non_blocking=False)
            )
    return GpuBf16Rows(
        rows=packed.rows,
        width=packed.width,
        block_size=packed.block_size,
        source_sha256=packed.source_sha256,
        base=buffers["base"],
        row_offsets=buffers["row_offsets"],
        block_offsets=buffers["block_offsets"],
        sign_mantissa=buffers["sign_mantissa"],
        planes=buffers["planes"],
        exceptions=buffers["exceptions"],
    )


def _validated_ids(packed: GpuBf16Rows, ids: torch.Tensor, *, validate_range: bool) -> torch.Tensor:
    """Check the supported id contract and return the flat view to decode."""
    if ids.device != packed.base.device:
        raise ValueError("row ids must live on the same CUDA device as the packed rows")
    if ids.dtype not in _ID_DTYPES:
        raise ValueError("row ids must be int32 or int64")
    if ids.ndim == 0:
        raise ValueError("scalar row ids are unsupported; use a one-dimensional id tensor")
    if not ids.is_contiguous():
        raise ValueError("row ids must be contiguous")
    flat = ids.view(-1)
    if validate_range and flat.numel():
        low = int(flat.min().item())
        high = int(flat.max().item())
        if low < 0 or high >= packed.rows:
            raise ValueError("row id outside the packed table")
    return flat


def gather_rows(
    packed: GpuBf16Rows,
    ids: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    validate_range: bool = True,
) -> torch.Tensor:
    """Decode the requested rows into a bounded ``[..., width]`` BF16 buffer.

    The returned buffer holds only the requested rows; it is scratch space the
    caller must release or reuse. Pass ``out`` to reuse one allocation across
    lookups. During CUDA graph capture, range validation is skipped (it would
    force a host synchronization); callers must then guarantee valid ids.
    """
    capturing = torch.cuda.is_current_stream_capturing()
    flat = _validated_ids(packed, ids, validate_range=validate_range and not capturing)
    output_shape = (*ids.shape, packed.width)
    if out is None:
        result = torch.empty(output_shape, dtype=torch.bfloat16, device=ids.device)
    else:
        if out.shape != output_shape or out.dtype != torch.bfloat16:
            raise ValueError("out must be a bf16 tensor shaped [*ids.shape, width]")
        if out.device != ids.device or not out.is_contiguous():
            raise ValueError("out must be contiguous and on the id device")
        result = out
    if flat.numel():
        _row_gather_kernel[(flat.numel(), packed.blocks_per_row)](
            flat,
            packed.base,
            packed.row_offsets,
            packed.block_offsets,
            packed.sign_mantissa,
            packed.planes,
            packed.exceptions,
            result.view(-1, packed.width),
            WIDTH=packed.width,
            BLOCK_WORDS=packed.block_size,
            BLOCKS=packed.blocks_per_row,
            PLANE_ROW=(packed.width + 7) // 8,
        )
    return result
