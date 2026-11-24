"""Check the existing production slab reader against imported MXFP4 checkpoint bytes on CPU."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import resource
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import torch
from kda_policy import load_policy
from safetensors import safe_open


def digest(tensor: torch.Tensor) -> str:
    """Hash raw CPU storage without a floating-point conversion or a large temporary copy."""
    flat = tensor.contiguous().view(torch.uint8).reshape(-1)
    result = hashlib.sha256()
    for start in range(0, flat.numel(), 16 * 1024**2):
        result.update(flat[start:start + 16 * 1024**2].numpy())
    return result.hexdigest()


def main() -> None:
    """Verify first, middle and last shards, including every yielded tensor and its order."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--reader", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("Preserve loader verification evidence")
    load_policy(args.source)
    torch.set_num_threads(1)
    os.environ.update(GLM_FAST_LOAD_PINNED="0", GLM_FAST_LOAD_DROP_CACHE="0",
        GLM_FAST_LOAD_THREADS="4", GLM_FAST_LOAD_SLAB_MB="8", GLM_FAST_LOAD_AHEAD_MB="32",
        GLM_FAST_LOAD_VERIFY="0")
    spec = importlib.util.spec_from_file_location("production_slab_reader", args.reader)
    assert spec and spec.loader
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    index = json.loads((args.source / "model.safetensors.index.json").read_bytes())["weight_map"]
    files = sorted(set(index.values()))
    chosen = [args.source / files[i] for i in (0, len(files) // 2, len(files) - 1)]
    stats = reader._Stats()
    started = time.monotonic()
    result: dict[str, Any] = {"status": "running", "gpu_used": False, "pinned": False,
        "reader_sha256": hashlib.sha256(args.reader.read_bytes()).hexdigest(),
        "cases": [], "limits": ["Three full representative shards, not all 120 files.",
            "CPU slab bytes and ordering; not a measured pinned H2D or model startup speedup.",
            "The existing production reader is used read-only; its implementation is not copied."]}
    with ExitStack() as stack:
        handles = {str(path): stack.enter_context(safe_open(path, framework="pt", device="cpu"))
                   for path in chosen}
        order = [(str(path), name) for path in chosen for name in list(handles[str(path)].keys())]
        wanted = iter(order)
        rows = {str(path): {"file": path.name, "tensors": 0, "bytes": 0, "dtypes": {},
                           "tensor_sha256_digest": hashlib.sha256()} for path in chosen}
        count = 0
        iterator = reader.fast_safetensors_iterator([str(p) for p in chosen],
            keys_of=lambda path: list(handles[path].keys()),
            open_stock=lambda path: safe_open(path, framework="pt", device="cpu"), stats=stats)
        for name, tensor in iterator:
            path, expected_name = next(wanted)
            reference = handles[path].get_tensor(name)
            actual_sha = digest(tensor)
            if (name != expected_name or tensor.dtype != reference.dtype
                    or tensor.shape != reference.shape or actual_sha != digest(reference)):
                raise RuntimeError(f"Production reader changed imported checkpoint bytes: {name}")
            row = rows[path]
            row["tensors"] += 1
            row["bytes"] += tensor.numel() * tensor.element_size()
            dtype = str(tensor.dtype)
            row["dtypes"][dtype] = row["dtypes"].get(dtype, 0) + 1
            row["tensor_sha256_digest"].update(name.encode() + bytes.fromhex(actual_sha))
            count += 1
            del tensor, reference
        if next(wanted, None) is not None or count != len(order):
            raise ValueError("The production reader omitted tensors")
        for row in rows.values():
            row["tensor_sha256_digest"] = row["tensor_sha256_digest"].hexdigest()
            result["cases"].append(row)
    result.update(status="passed", seconds=time.monotonic() - started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        tensors=count, eager_tensors=stats.eager, passthrough_tensors=stats.passthrough)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("cases", "limits")}))


if __name__ == "__main__":
    main()
