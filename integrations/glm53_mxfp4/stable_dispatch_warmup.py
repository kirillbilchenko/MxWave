"""Prepare every supported deterministic expert-order specialization before serving."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from statistics import median
from typing import Any

BLOCK_SIZES = (8, 16, 32, 48, 64)


def specializations(max_tokens: int = 2048) -> list[dict[str, int]]:
    """Cover the complete existing router contract, including padded block-48 overflow."""
    if not isinstance(max_tokens, int) or not 1 <= max_tokens <= 2048:
        raise ValueError("Stable-dispatch warmup requires a token limit from 1 through 2048")
    cases: dict[tuple[int, int], dict[str, int]] = {}
    for block in BLOCK_SIZES:
        for tokens in range(2, max_tokens + 1):
            padded = ((tokens + block - 1) // block) * block
            capacity = 1 << (max(32, padded) - 1).bit_length()
            key = block, capacity
            if key not in cases:
                cases[key] = {"block": block, "capacity": capacity, "first_tokens": tokens}
            cases[key]["last_tokens"] = tokens
            cases[key]["padded_tokens"] = padded
    return list(cases.values())


def warmup(max_tokens: int = 2048) -> dict[str, Any]:
    """Compile unchanged kernels on guarded scratch metadata and verify integer results.

    Invoke only between model requests, outside graph capture. The fixtures use
    256-byte-aligned views like the native producer. No model weights or live
    routing metadata are passed to this function.
    """
    import stable_dispatch
    import torch

    if not torch.cuda.is_available() or torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Warmup requires an initialized CUDA device outside graph capture")
    cases = specializations(max_tokens)
    source_paths = (
        Path(stable_dispatch.__file__),
        Path(stable_dispatch.__file__).with_name("stable_dispatch_gpu.py"),
        Path(__file__),
    )
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    device = torch.device("cuda", torch.cuda.current_device())
    before = torch.cuda.memory_allocated(device)
    started = time.perf_counter()
    guard_size, guard_value, expert = 64, -123456789, 143

    def fixture(case: dict[str, int]) -> dict[str, Any]:
        tokens, block, count = case["last_tokens"], case["block"], case["padded_tokens"]
        guards = [guard_value] * guard_size
        # Unique top-k permits each expert at most one flattened id per token.
        unsorted = [i * 8 for i in reversed(range(tokens))] + [tokens * 8] * (count - tokens)
        expected = [guards + sorted(unsorted) + guards,
                    guards + [expert] * (count // block) + guards,
                    guards + [count] + guards]
        storage = [
            torch.tensor(guards + unsorted + guards, dtype=torch.int32, device=device),
            torch.tensor(expected[1], dtype=torch.int32, device=device),
            torch.tensor(expected[2], dtype=torch.int32, device=device),
        ]
        lengths = (count, count // block, 1)
        views = [v[guard_size:guard_size + n] for v, n in zip(storage, lengths, strict=True)]
        if any(v.data_ptr() % 256 for v in views):
            raise RuntimeError("Scratch pointers must retain native allocation alignment")
        torch.cuda.synchronize(device)

        def invoke() -> float:
            start = time.perf_counter()
            stable_dispatch.canonicalize(
                *views, num_tokens=tokens, num_experts=288, block=block,
            )
            torch.cuda.synchronize(device)
            return time.perf_counter() - start

        first = invoke()
        if [v.cpu().tolist() for v in storage] != expected:
            raise RuntimeError("Canonicalization changed metadata, guards or integer order")
        repeats = [invoke() for _ in range(3)]
        if [v.cpu().tolist() for v in storage] != expected:
            raise RuntimeError("Repeated canonicalization changed verified integer results")
        return {
            **case,
            "first_call_seconds": first,
            "repeat_seconds": repeats,
            "median_repeat_seconds": median(repeats),
            "fixture_bytes": sum(v.numel() * v.element_size() for v in storage),
            "guards_and_metadata_exact": True,
            "sorted_integer_ids_exact": True,
            "scratch_pointers_aligned_to_256_bytes": True,
        }

    with torch.inference_mode():
        results = [fixture(case) for case in cases]
    return {
        "status": "complete",
        "max_tokens": max_tokens,
        "block_sizes": list(BLOCK_SIZES),
        "specializations": results,
        "specialization_count": len(results),
        "seconds": time.perf_counter() - started,
        "source_sha256": hashes,
        "cuda_allocated_before": before,
        "cuda_allocated_after": torch.cuda.memory_allocated(device),
        "maximum_fixture_bytes": max((r["fixture_bytes"] for r in results), default=0),
        "model_weights_or_live_routing_touched": False,
        "existing_kernel_implementation_changed": False,
        "qualification_timing": False,
    }
