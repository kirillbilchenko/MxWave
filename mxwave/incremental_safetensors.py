"""Bounded-memory safetensors emission helpers.

The upstream safetensors Python API serializes a complete mapping at once.  A
model converter cannot use that API without keeping every output tensor in a
shard resident in host memory.  This module writes the same simple, contiguous
safetensors layout incrementally: the complete header is known from the model
plan, while tensor payloads are supplied in bounded chunks.
"""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import BinaryIO

import torch

from .shard import TensorInfo

__all__ = ["IncrementalSafeTensorsWriter", "source_data_start"]


_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def source_data_start(stream: BinaryIO, path: Path) -> int:
    """Return the absolute byte offset of a safetensors payload section."""
    stream.seek(0)
    raw_length = stream.read(8)
    if len(raw_length) != 8:
        raise ValueError(f"Invalid safetensors header in {path}")
    header_length = int(struct.unpack("<Q", raw_length)[0])
    if header_length > 100_000_000:
        raise ValueError(f"Unreasonably large safetensors header in {path}")
    data_start = 8 + header_length
    if data_start > path.stat().st_size:
        raise ValueError(f"Truncated safetensors header in {path}")
    return data_start


class IncrementalSafeTensorsWriter:
    """Write a pre-planned safetensors file without retaining full tensors.

    Tensor payloads must be written sequentially within each tensor. Different
    tensors may be interleaved, which lets a quantizer write packed values and
    scales for each row chunk without concatenating either full matrix.
    """

    def __init__(
        self,
        stream: BinaryIO,
        specs: Mapping[str, tuple[tuple[int, ...], str]],
    ) -> None:
        if not specs:
            raise ValueError("A safetensors file must contain at least one tensor")
        if "__metadata__" in specs:
            raise ValueError("__metadata__ is reserved by the safetensors format")

        header: dict[str, dict[str, object]] = {}
        tensor_bytes: dict[str, int] = {}
        data_offsets: dict[str, int] = {}
        offset = 0
        # safetensors.torch.save_file emits tensors in lexical key order. Keep
        # that order for deterministic, byte-compatible files.
        for name in sorted(specs):
            shape, dtype = specs[name]
            dtype_bytes = _DTYPE_BYTES.get(dtype)
            if dtype_bytes is None:
                raise ValueError(f"Unsupported safetensors dtype {dtype!r} for {name!r}")
            if any(not isinstance(dim, int) or dim < 0 for dim in shape):
                raise ValueError(f"Invalid shape {shape!r} for {name!r}")
            size = math.prod(shape) * dtype_bytes
            tensor_bytes[name] = size
            data_offsets[name] = offset
            header[name] = {
                "dtype": dtype,
                "shape": list(shape),
                "data_offsets": [offset, offset + size],
            }
            offset += size

        raw_header = json.dumps(
            header,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        raw_header += b" " * (-len(raw_header) % 8)
        self._stream = stream
        self._data_start = 8 + len(raw_header)
        self._specs = dict(specs)
        self._tensor_bytes = tensor_bytes
        self._written = {name: 0 for name in header}
        self._data_offsets = data_offsets
        stream.seek(0)
        stream.write(struct.pack("<Q", len(raw_header)))
        stream.write(raw_header)
        stream.truncate(self._data_start + offset)

    def _write_bytes(self, name: str, payload: bytes | bytearray | memoryview) -> None:
        """Append raw bytes to one tensor payload."""
        if name not in self._tensor_bytes:
            raise KeyError(f"Tensor {name!r} was not declared in the output header")
        payload_view = memoryview(payload).cast("B")
        start = self._written[name]
        stop = start + payload_view.nbytes
        expected = self._tensor_bytes[name]
        if stop > expected:
            raise ValueError(
                f"Tensor {name!r} received {stop} bytes, above its planned {expected} bytes"
            )
        self._stream.seek(self._data_start + self._data_offsets[name] + start)
        written = self._stream.write(payload_view)
        if written != payload_view.nbytes:
            raise OSError(
                f"Short write for tensor {name!r}: wrote {written} of {payload_view.nbytes} bytes"
            )
        self._written[name] = stop

    def write_u8_chunk(self, name: str, tensor: torch.Tensor) -> None:
        """Append one contiguous uint8 tensor chunk to a declared U8 tensor."""
        if self._tensor_bytes.get(name) is None:
            raise KeyError(f"Tensor {name!r} was not declared in the output header")
        _shape, declared_dtype = self._specs[name]
        if declared_dtype != "U8":
            raise TypeError(
                f"Tensor {name!r} is declared as {declared_dtype}, not U8"
            )
        if tensor.dtype != torch.uint8:
            raise TypeError(f"Tensor {name!r} must be torch.uint8, found {tensor.dtype}")
        value = tensor.detach().cpu().contiguous().reshape(-1)
        self._write_bytes(name, memoryview(value.numpy()))

    def copy_tensor_payload(
        self,
        name: str,
        source: BinaryIO,
        source_path: Path,
        source_info: TensorInfo,
        *,
        source_payload_start: int,
        chunk_size_bytes: int = 8 * 1024**2,
    ) -> None:
        """Copy one unchanged tensor payload in bounded raw-byte chunks."""
        if chunk_size_bytes <= 0:
            raise ValueError("chunk_size_bytes must be positive")
        if name != source_info.name:
            raise ValueError(
                f"Raw tensor copy cannot rename {source_info.name!r} to {name!r}"
            )
        planned_shape, planned_dtype = self._specs[name]
        if planned_shape != source_info.shape or planned_dtype != source_info.dtype:
            raise ValueError(
                f"Output spec for {name!r} is {planned_shape}/{planned_dtype}, but "
                f"the source is {source_info.shape}/{source_info.dtype}"
            )
        expected_size = source_info.data_offsets[1] - source_info.data_offsets[0]
        if self._tensor_bytes.get(name) != expected_size:
            raise ValueError(
                f"Output payload size for {name!r} does not match {source_path.name}"
            )
        source.seek(source_payload_start + source_info.data_offsets[0])
        remaining = expected_size
        while remaining:
            payload = source.read(min(remaining, chunk_size_bytes))
            if not payload:
                raise ValueError(
                    f"Unexpected end of safetensors payload for {name!r} in {source_path}"
                )
            self._write_bytes(name, payload)
            remaining -= len(payload)

    def finish(self) -> None:
        """Validate complete coverage and flush the output payload."""
        incomplete = {
            name: (self._written[name], expected)
            for name, expected in self._tensor_bytes.items()
            if self._written[name] != expected
        }
        if incomplete:
            sample = list(incomplete.items())[:5]
            raise ValueError(f"Incomplete safetensors payloads: {sample}")
        self._stream.flush()
