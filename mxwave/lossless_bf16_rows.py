"""Row-addressable BF16 embedding layout built on the BF16 exponent codec.

Every row is encoded independently with the exact exponent-window semantics of
:mod:`mxwave.lossless_bf16` (``pack_bf16_bytes`` with a 512-word block): eight
sign/mantissa bits are kept per value, three bit planes encode a common
seven-exponent window, and code zero selects an exact exception byte indexed
per block. This module stores the per-row streams in flat, row-aligned buffers
so a GPU kernel can reconstruct only the requested rows without decompressing
the whole table.

The runtime representation is experimental. It is not a checkpoint format and
not the ZipServ codec.
"""

from __future__ import annotations

import hashlib
import os
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np

_MAGIC = b"MXBEMB01"
_VERSION = 1
_HEADER = struct.Struct("<8sIQIIQ32s")
_ARRAY_ENTRY = struct.Struct("<Q32s")
_ARRAY_ORDER = ("base", "row_offsets", "block_offsets", "sign_mantissa", "planes", "exceptions")
_HASH_CHUNK = 1 << 24


def _valid_block_size(block_size: int) -> bool:
    """Return whether the block size matches the base codec constraints."""
    return 32 <= block_size <= 4096 and block_size.bit_count() == 1


@dataclass(frozen=True)
class Bf16RowPacked:
    """Row-aligned compressed BF16 rows with per-row block exception indexes."""

    rows: int
    width: int
    block_size: int
    source_sha256: str
    base: np.ndarray
    row_offsets: np.ndarray
    block_offsets: np.ndarray
    sign_mantissa: np.ndarray
    planes: np.ndarray
    exceptions: np.ndarray

    def __post_init__(self) -> None:
        """Validate shapes, dtypes and index consistency before any decode."""
        if not _valid_block_size(self.block_size):
            raise ValueError("block_size must be a power of two from 32 through 4096")
        if self.rows < 0 or self.rows > 2**31 - 1:
            raise ValueError("invalid row count")
        if self.width <= 0 or self.width % self.block_size:
            raise ValueError("row width must be a positive multiple of the codec block size")
        blocks = self.blocks_per_row
        plane_bytes = self.plane_bytes
        if len(self.source_sha256) != 64 or set(self.source_sha256) - set("0123456789abcdef"):
            raise ValueError("source_sha256 must be a 64-character lowercase hex digest")
        if self.base.shape != (self.rows,) or self.base.dtype != np.uint8:
            raise ValueError("invalid base exponent array")
        if self.row_offsets.shape != (self.rows + 1,) or self.row_offsets.dtype != np.uint32:
            raise ValueError("invalid row exception offset array")
        if self.row_offsets[0] != 0 or int(self.row_offsets[-1]) != len(self.exceptions):
            raise ValueError("row exception offsets must cover the exception stream")
        if (
            self.block_offsets.shape != (self.rows, blocks)
            or self.block_offsets.dtype != np.uint32
        ):
            raise ValueError("invalid block exception offset array")
        if (
            self.sign_mantissa.shape != (self.rows * self.width,)
            or self.sign_mantissa.dtype != np.uint8
        ):
            raise ValueError("invalid sign/mantissa array")
        if self.planes.shape != (self.rows, 3 * plane_bytes) or self.planes.dtype != np.uint8:
            raise ValueError("invalid bit-plane array")
        if self.exceptions.dtype != np.uint8:
            raise ValueError("exception stream must be uint8")
        if len(self.exceptions) > 2**32 - 1:
            raise ValueError("exception stream exceeds uint32 indexing")
        if self.rows and (
            bool((self.block_offsets > self.width).any())
            or bool((np.diff(self.block_offsets, axis=1) < 0).any())
            or bool(
                (
                    self.block_offsets[:, -1]
                    > self.row_offsets[1:] - self.row_offsets[:-1]
                ).any()
            )
        ):
            raise ValueError("block exception offsets must be monotonic within each row")

    @property
    def blocks_per_row(self) -> int:
        """Number of codec blocks in one row."""
        return self.width // self.block_size

    @property
    def plane_bytes(self) -> int:
        """Byte length of one exponent bit plane for one row."""
        return (self.width + 7) // 8

    @property
    def original_bytes(self) -> int:
        """Uncompressed BF16 byte count for the represented table."""
        return self.rows * self.width * 2

    @property
    def retained_bytes(self) -> int:
        """Every retained byte: arrays, block indexes and the format header."""
        return _HEADER.size + len(_ARRAY_ORDER) * _ARRAY_ENTRY.size + sum(
            self.array_sizes().values()
        )

    def array_sizes(self) -> dict[str, int]:
        """Byte sizes of the serialized arrays in on-disk order."""
        blocks = self.blocks_per_row
        plane_bytes = self.plane_bytes
        return {
            "base": self.rows,
            "row_offsets": 4 * (self.rows + 1),
            "block_offsets": 4 * self.rows * blocks,
            "sign_mantissa": self.rows * self.width,
            "planes": self.rows * 3 * plane_bytes,
            "exceptions": len(self.exceptions),
        }

    def row_words(self, row_ids: np.ndarray) -> np.ndarray:
        """Decode raw BF16 rows by id; the independent CPU reference path."""
        ids = np.asarray(row_ids, dtype=np.int64).reshape(-1)
        if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= self.rows):
            raise ValueError("row id outside the packed table")
        selected_planes = self.planes.reshape(self.rows, 3, self.plane_bytes)[ids]
        bits = np.unpackbits(selected_planes, axis=-1, count=self.width, bitorder="little")
        codes = bits[:, 0] | (bits[:, 1] << 1) | (bits[:, 2] << 2)
        rare = codes == 0
        exponent = self.base[ids][:, None].astype(np.int16) + codes.astype(np.int16) - 1
        if len(self.exceptions):
            rank = np.cumsum(rare, axis=1, dtype=np.int64)
            index = self.row_offsets[ids][:, None].astype(np.int64) + rank - 1
            exponent[rare] = self.exceptions[index[rare]].astype(np.int16)
        sm = self.sign_mantissa.reshape(self.rows, self.width)[ids].astype(np.uint16)
        return (((sm & 128) << 8) | (exponent.astype(np.uint16) << 7) | (sm & 127)).astype("<u2")

    def array_hashes(self) -> dict[str, str]:
        """SHA256 of every serialized array plus the bound source digest."""
        result: dict[str, str] = {}
        for name in _ARRAY_ORDER:
            flat = np.ascontiguousarray(getattr(self, name)).reshape(-1)
            digest = hashlib.sha256()
            for start in range(0, flat.size, _HASH_CHUNK):
                digest.update(flat[start : start + _HASH_CHUNK].tobytes())
            result[name] = digest.hexdigest()
        result["source"] = self.source_sha256
        return result

    def iter_chunks(self, chunk_rows: int = 4096) -> Iterator[Bf16RowPacked]:
        """Yield bounded row slices sharing the same codec parameters."""
        if chunk_rows <= 0:
            raise ValueError("chunk_rows must be positive")
        for start in range(0, self.rows, chunk_rows):
            end = min(start + chunk_rows, self.rows)
            offset = int(self.row_offsets[start])
            yield Bf16RowPacked(
                rows=end - start,
                width=self.width,
                block_size=self.block_size,
                source_sha256=self.source_sha256,
                base=self.base[start:end],
                row_offsets=self.row_offsets[start : end + 1] - offset,
                block_offsets=self.block_offsets[start:end],
                sign_mantissa=self.sign_mantissa[start * self.width : end * self.width],
                planes=self.planes[start:end],
                exceptions=self.exceptions[int(self.row_offsets[start]) : int(self.row_offsets[end])],
            )


def pack_rows(
    values: np.ndarray,
    *,
    block_size: int = 512,
    chunk_rows: int = 4096,
    source_sha256: str = "0" * 64,
) -> Bf16RowPacked:
    """Encode a ``[rows, width]`` little-endian BF16 word matrix row by row.

    Each row is encoded exactly as ``pack_bf16_bytes(row_bytes, block_size)``
    from :mod:`mxwave.lossless_bf16` would encode it: same base-exponent
    selection, same bit planes and the same per-block exception indexing. The
    vectorized implementation never converts values through a float type.
    """
    if values.ndim != 2 or values.dtype != np.dtype("<u2") or not values.flags["C_CONTIGUOUS"]:
        raise ValueError("values must be a C-contiguous [rows, width] '<u2' array")
    if not _valid_block_size(block_size):
        raise ValueError("block_size must be a power of two from 32 through 4096")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    rows, width = values.shape
    if width % block_size:
        raise ValueError("row width must be a multiple of the codec block size")
    blocks = width // block_size
    plane_bytes = (width + 7) // 8
    base = np.empty(rows, dtype=np.uint8)
    row_offsets = np.empty(rows + 1, dtype=np.uint32)
    row_offsets[0] = 0
    block_offsets = np.empty((rows, blocks), dtype=np.uint32)
    sign_mantissa = np.empty(rows * width, dtype=np.uint8)
    planes = np.empty((rows, 3 * plane_bytes), dtype=np.uint8)
    exception_chunks: list[np.ndarray] = []
    done = 0
    for start in range(0, rows, chunk_rows):
        end = min(start + chunk_rows, rows)
        chunk = values[start:end]
        count = end - start
        exponent = ((chunk >> 7) & 255).astype(np.uint8)
        keys = (np.arange(count, dtype=np.int32)[:, None] * 256 + exponent.astype(np.int32)).ravel()
        frequencies = np.bincount(keys, minlength=count * 256).reshape(count, 256)
        cumulative = np.zeros((count, 257), dtype=np.int64)
        np.cumsum(frequencies, axis=1, out=cumulative[:, 1:])
        windows = cumulative[:, 7:] - cumulative[:, :-7]
        chunk_base = windows.argmax(axis=1).astype(np.uint8)
        low = chunk_base.astype(np.int16)[:, None]
        signed = exponent.astype(np.int16)
        common = (signed >= low) & (signed < low + 7)
        codes = np.where(common, signed - low + 1, 0).astype(np.uint8)
        bits = ((codes[:, None, :] >> np.arange(3, dtype=np.uint8)[None, :, None]) & 1).astype(
            np.uint8
        )
        planes[start:end] = np.packbits(bits, axis=2, bitorder="little").reshape(
            count, 3 * plane_bytes
        )
        sign_mantissa[start * width : end * width] = (
            (((chunk >> 8) & 128) | (chunk & 127)).astype(np.uint8)
        ).ravel()
        rare = ~common
        padded = np.pad(rare, ((0, 0), (0, (-width) % block_size)), constant_values=False)
        counts = padded.reshape(count, blocks, block_size).sum(axis=2, dtype=np.uint32)
        block_offsets[start:end] = np.cumsum(counts, axis=1, dtype=np.uint32) - counts
        row_counts = counts.sum(axis=1, dtype=np.uint64)
        row_offsets[start + 1 : end + 1] = (done + np.cumsum(row_counts)).astype(np.uint32)
        exception_chunks.append(exponent[rare])
        base[start:end] = chunk_base
        done += int(row_counts.sum())
    if done > 2**32 - 1:
        raise ValueError("exception stream exceeds uint32 indexing")
    exceptions = (
        np.concatenate(exception_chunks) if exception_chunks else np.empty(0, dtype=np.uint8)
    )
    return Bf16RowPacked(
        rows=rows,
        width=width,
        block_size=block_size,
        source_sha256=source_sha256,
        base=base,
        row_offsets=row_offsets,
        block_offsets=block_offsets,
        sign_mantissa=sign_mantissa,
        planes=planes,
        exceptions=exceptions,
    )


def write_to(packed: Bf16RowPacked, path: str | Path) -> None:
    """Serialize the row layout into a versioned, hash-bound artifact."""
    sizes = packed.array_sizes()

    def flush(handle: BinaryIO) -> None:
        handle.flush()
        os.fsync(handle.fileno())

    header_size = _HEADER.size + len(_ARRAY_ORDER) * _ARRAY_ENTRY.size
    with open(path, "wb") as handle:
        handle.truncate(header_size)
        handle.seek(header_size)
        handle.write(packed.base.tobytes())
        handle.write(packed.row_offsets.tobytes())
        handle.write(packed.block_offsets.tobytes())
        handle.write(packed.sign_mantissa.tobytes())
        handle.write(packed.planes.tobytes())
        handle.write(packed.exceptions.tobytes())
        flush(handle)
        payload_size = handle.tell() - header_size
    if header_size + payload_size != packed.retained_bytes:
        raise ValueError("serialized size does not match the layout accounting")
    with open(path, "r+b") as header_handle:
        header_handle.write(
            _HEADER.pack(
                _MAGIC,
                _VERSION,
                packed.rows,
                packed.width,
                packed.block_size,
                len(packed.exceptions),
                bytes.fromhex(packed.source_sha256),
            )
            + b"".join(
                _ARRAY_ENTRY.pack(sizes[name], bytes.fromhex(packed.array_hashes()[name]))
                for name in _ARRAY_ORDER
            )
        )
        flush(header_handle)


def read_from(path: str | Path, *, verify_hashes: bool = False) -> Bf16RowPacked:
    """Load a row-layout artifact, optionally verifying every recorded hash."""
    data = Path(path)
    size = data.stat().st_size
    header_size = _HEADER.size + len(_ARRAY_ORDER) * _ARRAY_ENTRY.size
    with data.open("rb") as handle:
        header = handle.read(_HEADER.size)
        if len(header) != _HEADER.size:
            raise ValueError("truncated row-layout header")
        magic, version, rows, width, block_size, exceptions_count, source = _HEADER.unpack(header)
        if magic != _MAGIC or version != _VERSION:
            raise ValueError("unsupported row-layout artifact")
        entries = {
            name: _ARRAY_ENTRY.unpack(handle.read(_ARRAY_ENTRY.size)) for name in _ARRAY_ORDER
        }
    if not _valid_block_size(block_size) or width <= 0 or width % block_size:
        raise ValueError("unsupported row-layout width or block size")
    plane_bytes = (width + 7) // 8
    blocks = width // block_size
    expected = {
        "base": rows,
        "row_offsets": 4 * (rows + 1),
        "block_offsets": 4 * rows * blocks,
        "sign_mantissa": rows * width,
        "planes": rows * 3 * plane_bytes,
        "exceptions": int(exceptions_count),
    }
    for name in _ARRAY_ORDER:
        if entries[name][0] != expected[name]:
            raise ValueError(f"row-layout array size mismatch: {name}")
    if header_size + sum(expected.values()) != size:
        raise ValueError("row-layout artifact length mismatch")

    shapes: dict[str, tuple[int, ...]] = {
        "base": (rows,),
        "row_offsets": (rows + 1,),
        "block_offsets": (rows, blocks),
        "sign_mantissa": (rows * width,),
        "planes": (rows, 3 * plane_bytes),
        "exceptions": (int(exceptions_count),),
    }
    arrays: dict[str, np.ndarray] = {}
    offset = header_size
    for name in _ARRAY_ORDER:
        dtype = np.uint32 if name.endswith("offsets") else np.uint8
        arrays[name] = np.memmap(data, dtype=dtype, mode="r", offset=offset, shape=shapes[name])
        offset += expected[name]
    packed = Bf16RowPacked(
        rows=rows,
        width=width,
        block_size=block_size,
        source_sha256=source.hex(),
        base=arrays["base"],
        row_offsets=arrays["row_offsets"],
        block_offsets=arrays["block_offsets"],
        sign_mantissa=arrays["sign_mantissa"],
        planes=arrays["planes"],
        exceptions=arrays["exceptions"],
    )
    if verify_hashes:
        recorded = {name: entries[name][1].hex() for name in _ARRAY_ORDER}
        recorded["source"] = source.hex()
        if packed.array_hashes() != recorded:
            raise ValueError("row-layout array hash mismatch")
    return packed
