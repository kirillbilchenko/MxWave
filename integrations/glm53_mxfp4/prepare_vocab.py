"""Prepare hash-bound head and embedding artifacts on CPU without stopping GLM."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import resource
import time
from pathlib import Path

import torch
from prepare_bf16_attention import LIBRARY_SHA, RUNTIME_SHA, tensor_digest
from safetensors import safe_open
from safetensors.torch import save_file
from vocab_policy import load_targets
from zipserv_contract import require_qualified_exponent_base

from mxwave.lossless_bf16_rows import pack_rows, write_to


def main() -> None:
    """Pack four full TP2 shards, retaining source hashes and every serialized byte."""
    from mxwave_zipserv import runtime

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--chunk-rows", type=int, default=128)
    args = parser.parse_args()
    targets = load_targets(args.source)
    if (hashlib.sha256(Path(os.environ["MXWAVE_ZIPSERV_LIBRARY"]).read_bytes()).hexdigest()
            != LIBRARY_SHA or hashlib.sha256(Path(runtime.__file__).read_bytes()).hexdigest()
            != RUNTIME_SHA):
        raise ValueError("Use the qualified ZipServ native and Python ABI")
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "prepared.json"
    if report_path.exists():
        raise FileExistsError("Preserve vocabulary artifacts and preparation evidence")
    torch.set_num_threads(2)
    index = json.loads((args.source / "model.safetensors.index.json").read_bytes())["weight_map"]
    started = time.monotonic()
    report = {"status": "packing", "cases": [], "device": "cpu", "gpu_used": False}
    for rank in (0, 1):
        for target in targets.values():
            with safe_open(args.source / index[target.source_key], framework="pt",
                           device="cpu") as handle:
                weight = handle.get_slice(target.source_key)[
                    rank * target.rows:(rank + 1) * target.rows].contiguous()
            if weight.dtype != torch.bfloat16 or tuple(weight.shape) != (target.rows, target.width):
                raise ValueError("Source vocabulary storage must be BF16 without casts")
            digest = tensor_digest(weight)
            name = f"{target.kind}-r{rank}"
            if target.kind == "head":
                packed = runtime.compress(weight)
                require_qualified_exponent_base(packed.start_exp, where=f"head writer r{rank}")
                path = args.output / (name + ".safetensors")
                save_file(dict(zip(runtime.BUFFER_NAMES, packed.buffers, strict=True)), path)
                metadata = [packed.m, packed.k, packed.max_common,
                            packed.max_rare, packed.start_exp]
                retained = packed.storage_bytes
            else:
                packed = pack_rows(weight.view(torch.uint16).numpy(), source_sha256=digest,
                                   chunk_rows=args.chunk_rows)
                path = args.output / (name + ".mxb")
                write_to(packed, path)
                metadata = [packed.rows, packed.width, packed.block_size]
                retained = sum(packed.array_sizes().values())
            if retained >= weight.numel() * 2:
                raise ValueError("A selected vocabulary shard does not shrink")
            record = {"kind": target.kind, "rank": rank, "prefix": target.prefix,
                "source_key": target.source_key, "shape": list(weight.shape),
                "source_sha256": digest, "raw_bytes": weight.numel() * 2,
                "retained_bytes": retained, "metadata": metadata, "artifact": path.name,
                "artifact_bytes": path.stat().st_size,
                "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            report["cases"].append(record)
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(record), flush=True)
            del packed, weight
            gc.collect()
    report.update(status="packed_gpu_verification_pending", seconds=time.monotonic() - started,
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    report_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
