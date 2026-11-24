"""Experimental model-independent, random-addressable U8 scale tiles with exact exceptions.

No exponent is approximated. The tensor chooses its own most frequent values;
the GPU integration consumes these buffers without materializing a scale tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

Mode = Literal["mask128", "palette64"]
U8 = NDArray[np.uint8]
U32 = NDArray[np.uint32]


@dataclass(frozen=True)
class ScaleTiles:
    """Four U32 selector words per tile, aligned exception offsets and a tensor-wide palette."""

    count: int
    mode: Mode
    palette: tuple[int, ...]
    selectors: U32
    offsets: U32
    exceptions: U8

    @property
    def block(self) -> int:
        """Return logical scale values per independently decodable tile."""
        return 128 if self.mode == "mask128" else 64

    @property
    def descriptor(self) -> int:
        """Pack exact palette bytes and the codec mode into the descriptor's final U64."""
        return sum(v << (8 * i) for i, v in enumerate(self.palette)) | (
            int(self.mode == "palette64") << 32)

    @property
    def retained_bytes(self) -> int:
        """Include all three data arrays and the 32-byte GPU pointer/palette descriptor."""
        return self.selectors.nbytes + self.offsets.nbytes + self.exceptions.nbytes + 32

    def allocated_bytes(self, alignment: int = 512) -> int:
        """Account for minimum per-buffer allocator rounding without claiming measured residency."""
        if alignment < 1:
            raise ValueError("Require positive allocation alignment")
        return sum((size + alignment - 1) // alignment * alignment for size in
                   (self.selectors.nbytes, self.offsets.nbytes, self.exceptions.nbytes, 32))


def pack(raw: U8, mode: Mode) -> ScaleTiles:
    """Pack arbitrary U8 bytes in bounded CPU chunks, retaining every outlier exactly."""
    if raw.dtype != np.uint8 or raw.ndim != 1 or not raw.flags.c_contiguous or not raw.size:
        raise ValueError("Require nonempty contiguous one-dimensional U8 data")
    if mode not in ("mask128", "palette64"):
        raise ValueError("Unsupported scale tile mode")
    block = 128 if mode == "mask128" else 64
    count = int(raw.size)
    blocks = (count + block - 1) // block
    if blocks * block >= 2**32:
        raise ValueError("Tile payload exceeds the U32 offset format")
    histogram = np.bincount(raw, minlength=256)
    frequent = np.lexsort((np.arange(256), -histogram))[:1 if mode == "mask128" else 3]
    palette = tuple(int(v) for v in frequent)
    lookup = np.full(256, 3, dtype=np.uint8)
    for i, value in enumerate(palette):
        lookup[value] = i
    selectors = np.empty((blocks, 4), dtype=np.uint32)
    offsets = np.empty(blocks + 1, dtype=np.uint32)
    offsets[0] = 0
    chunks = []
    total = 0
    shifts = np.arange(16, dtype=np.uint32) * 2
    for first in range(0, count, 1024**2):
        data = raw[first:first + 1024**2]
        if data.size % block:
            data = np.pad(data, (0, block - data.size % block), constant_values=palette[0])
        values = data.reshape(-1, block)
        if mode == "mask128":
            escaped = values != palette[0]
            words = np.packbits(escaped, axis=1, bitorder="little").view("<u4")
        else:
            codes = lookup[values]
            escaped = codes == 3
            words = np.bitwise_or.reduce(
                codes.reshape(-1, 16).astype(np.uint32) << shifts, axis=1).reshape(-1, 4)
        counts = escaped.sum(axis=1, dtype=np.uint32)
        padded = (counts + 3) // 4 * 4
        local = np.concatenate((np.zeros(1, dtype=np.uint32), np.cumsum(padded, dtype=np.uint32)))
        if total + int(local[-1]) >= 2**32:
            raise ValueError("Exception payload exceeds the U32 offset format")
        payload = np.zeros(int(local[-1]), dtype=np.uint8)
        rows, columns = np.nonzero(escaped)
        ranks = np.cumsum(escaped, axis=1, dtype=np.uint16)[rows, columns] - 1
        payload[local[rows] + ranks] = values[rows, columns]
        index = first // block
        selectors[index:index + len(values)] = words
        offsets[index + 1:index + len(values) + 1] = local[1:] + total
        chunks.append(payload)
        total += payload.size
    return ScaleTiles(count, mode, palette, selectors, offsets, np.concatenate(chunks))


def unpack(packed: ScaleTiles) -> U8:
    """Independently decode every byte, validating tile offsets and exception padding."""
    if packed.mode not in ("mask128", "palette64") or packed.count < 1:
        raise ValueError("Invalid tile shape or mode")
    block, count = packed.block, packed.count
    blocks = (count + block - 1) // block
    palette = np.asarray(packed.palette, dtype=np.uint8)
    if (len(palette) != (1 if packed.mode == "mask128" else 3)
            or len(set(packed.palette)) != len(palette)
            or any(v < 0 or v > 255 for v in packed.palette)
            or packed.selectors.dtype != np.uint32 or packed.selectors.shape != (blocks, 4)
            or packed.offsets.dtype != np.uint32 or packed.offsets.shape != (blocks + 1,)
            or packed.exceptions.dtype != np.uint8 or packed.exceptions.ndim != 1
            or packed.offsets[0] != 0 or int(packed.offsets[-1]) != len(packed.exceptions)
            or np.any(packed.offsets % 4) or np.any(packed.offsets[1:] < packed.offsets[:-1])):
        raise ValueError("Invalid tile buffers, palette or exception bounds")
    output = np.empty(blocks * block, dtype=np.uint8)
    shifts = np.arange(16, dtype=np.uint32) * 2
    for first in range(0, blocks, 8192):
        words = packed.selectors[first:first + 8192]
        if packed.mode == "mask128":
            escaped = np.unpackbits(words.astype("<u4").view(np.uint8), axis=1,
                                    bitorder="little").astype(bool)
            values = np.full(escaped.shape, palette[0], dtype=np.uint8)
        else:
            codes = ((words[:, :, None] >> shifts) & 3).reshape(-1, block).astype(np.uint8)
            escaped = codes == 3
            values = np.append(palette, np.uint8(0))[codes]
        counts = escaped.sum(axis=1, dtype=np.uint32)
        lo = packed.offsets[first:first + len(words)]
        hi = packed.offsets[first + 1:first + len(words) + 1]
        if np.any(hi - lo != (counts + 3) // 4 * 4):
            raise ValueError("Exception counts differ from tile offsets")
        rows, columns = np.nonzero(escaped)
        ranks = np.cumsum(escaped, axis=1, dtype=np.uint16)[rows, columns] - 1
        values[rows, columns] = packed.exceptions[lo[rows] + ranks]
        for padding in range(3):
            pos = lo + counts + padding
            valid = pos < hi
            if np.any(packed.exceptions[pos[valid]]):
                raise ValueError("Nonzero exception padding")
        output[first * block:(first + len(words)) * block] = values.reshape(-1)
    if np.any(output[count:] != palette[0]):
        raise ValueError("Noncanonical final tile padding")
    return output[:count]


def choose(raw: U8) -> ScaleTiles:
    """Choose the smaller exact tiled representation without model- or layer-specific rules."""
    candidates = (pack(raw, "mask128"), pack(raw, "palette64"))
    return min(candidates, key=lambda candidate: candidate.allocated_bytes())


@dataclass(frozen=True)
class SparseScales:
    """Two-bit selectors or a tagged direct index into an exact 16-byte exception vector."""

    count: int
    palette: tuple[int, int, int]
    selectors: U32
    exceptions: U8

    @property
    def descriptor(self) -> int:
        """Encode a source-selected palette plus verified affine/alignment fast-path flags."""
        first, second, third = self.palette
        affine = second == first + 1 and third == first + 2
        return (first | second << 8 | third << 16 | int(affine) << 24
                | int(affine and first % 4 == 0) << 25)

    @property
    def retained_bytes(self) -> int:
        """Include both GPU buffers and the 24-byte descriptor."""
        return self.selectors.nbytes + self.exceptions.nbytes + 24

    def allocated_bytes(self, alignment: int = 512) -> int:
        """Estimate per-buffer rounding, including an otherwise tiny descriptor."""
        if alignment < 1:
            raise ValueError("Require positive allocation alignment")
        return sum((size + alignment - 1) // alignment * alignment for size in
                   (self.selectors.nbytes, self.exceptions.nbytes, 24))


def pack_sparse(raw: U8) -> SparseScales:
    """Keep frequent values on a two-bit path and preserve every exceptional vector verbatim."""
    if (raw.dtype != np.uint8 or raw.ndim != 1 or not raw.flags.c_contiguous
            or not raw.size or raw.size % 16 or raw.size >= 2**32):
        raise ValueError("Require contiguous U8 whole 16-byte vectors within U32 bounds")
    histogram = np.bincount(raw, minlength=256)
    frequent = np.lexsort((np.arange(256), -histogram))[:3]
    palette = tuple(int(v) for v in sorted(frequent))
    lookup = np.full(256, 3, dtype=np.uint8)
    for i, value in enumerate(palette):
        lookup[value] = i
    order = np.array([0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15])
    shifts = np.arange(16, dtype=np.uint32) * 2
    selectors = np.empty(raw.size // 16, dtype=np.uint32)
    escaped_values = []
    exceptions_seen = 0
    for first in range(0, raw.size, 1024**2):
        values = raw[first:first + 1024**2].reshape(-1, 16)
        codes = lookup[values]
        words = np.bitwise_or.reduce(codes[:, order].astype(np.uint32) << shifts, axis=1)
        escaped = np.any(codes == 3, axis=1)
        # Normal selectors never contain code 3. Reserve code 3 in the final
        # position as a whole-vector escape, with a direct 30-bit payload index.
        escaped_count = int(escaped.sum())
        if exceptions_seen + escaped_count >= 2**30:
            raise ValueError("Exception vector count exceeds its 30-bit tag")
        words[escaped] = 0xc0000000 | (np.arange(escaped_count, dtype=np.uint32) + exceptions_seen)
        selectors[first // 16:(first + values.size) // 16] = words
        escaped_values.append(values[escaped].copy())
        exceptions_seen += escaped_count
    exceptions = np.concatenate(escaped_values)
    return SparseScales(int(raw.size), palette, selectors, exceptions)


def unpack_sparse(packed: SparseScales) -> U8:
    """Independently restore all bytes, rejecting invalid, repeated or unreferenced escapes."""
    count = packed.count
    palette = packed.palette
    if (count < 16 or count % 16 or len(palette) != 3 or len(set(palette)) != 3
            or any(v < 0 or v > 255 for v in palette)
            or packed.selectors.dtype != np.uint32 or packed.selectors.shape != (count // 16,)
            or packed.exceptions.dtype != np.uint8 or packed.exceptions.ndim != 2
            or packed.exceptions.shape[1] != 16):
        raise ValueError("Invalid sparse scale buffers")
    output = np.empty(count, dtype=np.uint8)
    order = np.array([0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15])
    values = np.array([*palette, 0], dtype=np.uint8)
    shifts = np.arange(16, dtype=np.uint32) * 2
    exceptions_seen = 0
    for first in range(0, count // 16, 65536):
        words = packed.selectors[first:first + 65536]
        codes = ((words[:, None] >> shifts) & 3).astype(np.uint8)[:, order]
        decoded = values[codes]
        escaped = words >> 30 == 3
        if np.any(codes[~escaped] == 3):
            raise ValueError("Noncanonical escape selector")
        indices = words[escaped] & 0x3fffffff
        expected = np.arange(len(indices), dtype=np.uint32) + exceptions_seen
        if (not np.array_equal(indices, expected)
                or (len(indices) and indices[-1] >= len(packed.exceptions))):
            raise ValueError("Invalid or repeated exception index")
        decoded[escaped] = packed.exceptions[indices]
        exceptions_seen += len(indices)
        output[first * 16:(first + len(words)) * 16] = decoded.reshape(-1)
    if exceptions_seen != len(packed.exceptions):
        raise ValueError("Unreferenced sparse exception vectors")
    return output
