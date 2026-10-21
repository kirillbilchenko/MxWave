"""Compare stock CPU loader bytes on complete representative checkpoint shards."""

from __future__ import annotations

import argparse
import faulthandler
import gc
import hashlib
import json
import os
import resource
import time
from pathlib import Path
from typing import Any


def fingerprint(tensor: Any) -> dict[str, Any]:
    """Hash tensor storage without interpreting FP8 scale bytes as floats."""
    import torch

    raw = tensor.contiguous().reshape(-1).view(torch.uint8)
    digest = hashlib.sha256()
    for start in range(0, raw.numel(), 8 * 1024**2):
        digest.update(raw[start:start + 8 * 1024**2].numpy())
    return {"dtype": str(tensor.dtype), "shape": list(tensor.shape),
            "bytes": raw.numel(), "sha256": digest.hexdigest()}


def main() -> None:
    """Run one fresh CPU process per strategy; model startup remains a separate gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strategy", choices=("lazy", "eager", "runai_streamer"), required=True)
    parser.add_argument("--read-buffer-mib", type=int, default=1024)
    args = parser.parse_args()
    faulthandler.dump_traceback_later(60, repeat=True)
    if args.output.exists():
        raise FileExistsError("Preserve previous loader evidence")
    if args.read_buffer_mib < 1:
        parser.error("Require a positive read buffer")
    read_buffer_bytes = args.read_buffer_mib * 1024**2
    os.environ.update(RUNAI_STREAMER_MEMORY_LIMIT=str(read_buffer_bytes),
                      RUNAI_STREAMER_CONCURRENCY="8")
    import torch
    from safetensors import safe_open
    from vllm.model_executor.model_loader.weight_utils import (
        runai_safetensors_weights_iterator,
        safetensors_weights_iterator,
    )

    if torch.cuda.is_available():
        raise RuntimeError("The loader screen must run without GPU access")
    torch.set_num_threads(1)
    index = json.loads((args.model / "model.safetensors.index.json").read_bytes())["weight_map"]
    shards = sorted(set(index.values()))
    chosen = [args.model / shards[i] for i in (0, len(shards) // 2, len(shards) - 1)]
    expected = {}
    for path in chosen:
        print(json.dumps({"phase": "reference", "file": path.name}), flush=True)
        with safe_open(path, framework="pt", device="cpu") as handle:
            names = handle.keys()
            for name in names:
                if name in expected:
                    raise ValueError("Duplicate tensor name")
                expected[name] = fingerprint(handle.get_tensor(name))
    gc.collect()
    print(json.dumps({"phase": "iterator", "strategy": args.strategy}), flush=True)
    started = time.monotonic()
    files = [str(path) for path in chosen]
    iterator = (runai_safetensors_weights_iterator(files, False)
                if args.strategy == "runai_streamer"
                else safetensors_weights_iterator(files, False, args.strategy))
    seen = set()
    for name, tensor in iterator:
        if name in seen or name not in expected or fingerprint(tensor) != expected[name]:
            raise RuntimeError(f"Loader changed/duplicated an operand: {name}")
        seen.add(name)
        if len(seen) % 100 == 0:
            print(json.dumps({"phase": "checking", "tensors": len(seen)}), flush=True)
        del tensor
    if seen != set(expected):
        raise RuntimeError("Loader omitted checkpoint tensors")
    result = {
        "status": "passed", "strategy": args.strategy, "files": [p.name for p in chosen],
        "tensors": len(seen), "bytes": sum(v["bytes"] for v in expected.values()),
        "elapsed_including_hashing_seconds": time.monotonic() - started,
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "runai_read_buffer_limit_bytes": read_buffer_bytes, "runai_concurrency": 8,
        "inventory_sha256": hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest(),
        "limits": ["Three complete representative shards, not the whole model.",
                   "Reference read warms the file cache; elapsed time is not cold-start performance.",
                   "Tensor inventory and bytes checked; iteration order may differ.",
                   "No GPU, H2D, model loader callbacks or graph capture exercised."],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    faulthandler.cancel_dump_traceback_later()
    print(json.dumps(result))


if __name__ == "__main__":
    main()
