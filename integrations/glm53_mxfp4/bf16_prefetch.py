"""Bounded, stream-ordered BF16 prefetch for an isolated projection-chain component."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class Projection:
    """Immutable packed matrix and its unpadded logical row count."""

    name: str
    packed: Any
    rows: int


def decode_into(projection: Projection, output: torch.Tensor) -> None:
    """Decode into caller-owned storage on the current stream without allocating a weight."""
    from mxwave_zipserv import runtime

    packed = projection.packed
    device = packed.buffers[0].device
    if (device.type != "cuda" or output.device != device or output.dtype != torch.bfloat16
            or tuple(output.shape) != (packed.m, packed.k) or not output.is_contiguous()):
        raise ValueError("Decode destination must be a matching contiguous CUDA BF16 matrix")
    runtime._validate_buffers(list(packed.buffers), device)
    with torch.cuda.device(device):
        runtime._check(runtime._library().mxzip_decode(
            torch.cuda.current_stream(device).cuda_stream,
            *(buffer.data_ptr() for buffer in packed.buffers),
            packed.max_common, packed.max_rare, packed.start_exp,
            output.data_ptr(), packed.m, packed.k))


class PrefillPipeline:
    """Own two scratch slots, readiness/consumption events and an optional cached first weight.

    The projection-chain fixture passes the first input-width output columns to
    the next projection. This is a dependent component, not a GLM forward pass.
    Callers serialize invocations on one current stream; graph replay joins the
    decoder stream before returning. Simultaneous requests need separate arenas.
    """

    def __init__(self, projections: tuple[Projection, ...], *, prime_first: bool = False,
                 memory_limit_bytes: int = 256 * 1024**2) -> None:
        if len(projections) < 3:
            raise ValueError("The buffer-reuse component requires at least three projections")
        width = projections[0].packed.k
        device = projections[0].packed.buffers[0].device
        if (any(p.packed.k != width or not 0 < p.rows <= p.packed.m
                or p.packed.buffers[0].device != device for p in projections)
                or any(p.rows < width for p in projections[:-1])):
            raise ValueError("Projections must support the declared dependent activation bridge")
        self.projections = projections
        self.width = width
        self.device = device
        self.start = int(prime_first)
        words = max(p.packed.m * p.packed.k for p in projections[self.start:])
        cached_words = projections[0].packed.m * width if prime_first else 0
        self.weight_storage_bytes = (2 * words + cached_words) * 2
        if self.weight_storage_bytes > memory_limit_bytes:
            raise ValueError("Declared prefetch weight-storage budget exceeded")
        self.decode_stream = torch.cuda.Stream(device=device)
        self.buffers = tuple(torch.empty(words, dtype=torch.bfloat16, device=device)
                             for _ in range(2))
        self.ready = tuple(torch.cuda.Event() for _ in self.buffers)
        self.consumed = tuple(torch.cuda.Event() for _ in self.buffers)
        self.first = (torch.empty((projections[0].packed.m, width),
                                 dtype=torch.bfloat16, device=device) if prime_first else None)
        self.prime_milliseconds = 0.0
        if self.first is not None:
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            decode_into(projections[0], self.first)
            end.record()
            end.synchronize()
            self.prime_milliseconds = begin.elapsed_time(end)
        for buffer in self.buffers:
            buffer.record_stream(self.decode_stream)
        for projection in projections:
            for buffer in projection.packed.buffers:
                buffer.record_stream(self.decode_stream)

    def view(self, index: int) -> torch.Tensor:
        """Return the fixed slot view for one matrix without retaining an extra allocation."""
        projection = self.projections[index]
        slot = (index - self.start) % len(self.buffers)
        return self.buffers[slot][:projection.packed.m * projection.packed.k].view(
            projection.packed.m, projection.packed.k)

    def forward(self, x: torch.Tensor, *, overlap: bool, annotate: bool = False
                ) -> tuple[torch.Tensor, ...]:
        """Decode one matrix ahead, waiting both for readiness and safe scratch reuse."""
        if x.device != self.device or x.dtype != torch.bfloat16 or x.shape[-1] != self.width:
            raise ValueError("The component requires matching CUDA BF16 activations")
        main = torch.cuda.current_stream(self.device)
        for buffer in self.buffers:
            buffer.record_stream(main)
        if self.first is not None:
            self.first.record_stream(main)

        def enqueue(index: int) -> None:
            slot = (index - self.start) % len(self.buffers)
            with torch.cuda.stream(self.decode_stream):
                if index >= self.start + len(self.buffers):
                    self.decode_stream.wait_event(self.consumed[slot])
                scope = (torch.profiler.record_function("decode/" + self.projections[index].name)
                         if annotate else nullcontext())
                with scope:
                    decode_into(self.projections[index], self.view(index))
                self.ready[slot].record(self.decode_stream)

        if overlap:
            # Fork from the caller stream for safe eager use and multi-stream capture.
            self.decode_stream.wait_stream(main)
            enqueue(self.start)
        outputs = []
        for index, projection in enumerate(self.projections):
            slot = (index - self.start) % len(self.buffers)
            if index < self.start:
                assert self.first is not None
                weight = self.first
            else:
                weight = self.view(index)
                if overlap:
                    main.wait_event(self.ready[slot])
                else:
                    scope = (torch.profiler.record_function("decode/" + projection.name)
                             if annotate else nullcontext())
                    with scope:
                        decode_into(projection, weight)
            scope = (torch.profiler.record_function("gemm/" + projection.name)
                     if annotate else nullcontext())
            with scope:
                output = torch.nn.functional.linear(x, weight[:projection.rows])
            if overlap and index >= self.start:
                self.consumed[slot].record(main)
            if overlap and self.start < index + 1 < len(self.projections):
                # Queue compute first: a large decoder grid must not get first
                # access to every SM before the current GEMM is submitted.
                enqueue(index + 1)
            outputs.append(output)
            if index + 1 < len(self.projections):
                x = output[..., :self.width].contiguous()
        if overlap:
            # Join every auxiliary node before a graph or eager invocation completes.
            main.wait_stream(self.decode_stream)
        return tuple(outputs)


def interval_overlap(left: list[tuple[float, float]], right: list[tuple[float, float]]) -> float:
    """Measure intersection of interval unions without double-counting nested kernels."""
    def union(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
        merged: list[tuple[float, float]] = []
        for first, last in sorted(intervals):
            if last < first:
                raise ValueError("An interval cannot end before it starts")
            if merged and first <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(last, merged[-1][1]))
            else:
                merged.append((first, last))
        return merged

    a, b = union(left), union(right)
    return sum(max(0.0, min(a1, b1) - max(a0, b0))
               for a0, a1 in a for b0, b1 in b)
