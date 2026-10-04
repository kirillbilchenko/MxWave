"""Measure CUDA quantization chunk budgets on real checkpoint matrices."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import time
from collections.abc import Iterable
from functools import partial
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from mxwave.checkpoint import _atomic_write_text, iter_quantized_row_chunks, tensor_chunk_rows
from mxwave.core import QUANTIZATION_REVISION, quantize_mxfp4
from mxwave.incremental_safetensors import IncrementalSafeTensorsWriter
from mxwave.shard import ShardFile, read_tensor_row_range


def source_digest() -> str:
    """Hash the exact executed Python sources, including this benchmark."""
    root = Path(__file__).resolve().parent.parent
    paths = sorted((root / "mxwave").rglob("*.py")) + [Path(__file__).resolve()]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode() + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _read_rows(weight: Any, device: torch.device, start: int, stop: int) -> torch.Tensor:
    return weight[start:stop].to(device)


def _consume_chunks(
    chunks: Iterable[tuple[int, int, torch.Tensor, torch.Tensor]],
    output_path: Path | None,
    rows: int,
    columns: int,
) -> None:
    if output_path is None:
        for _start, _stop, packed, scales in chunks:
            del packed, scales
        return
    specs = {
        "weight_packed": ((rows, columns // 2), "U8"),
        "weight_scale": ((rows, columns // 32), "U8"),
    }
    with output_path.open("w+b") as stream:
        writer = IncrementalSafeTensorsWriter(stream, specs)
        for _start, _stop, packed, scales in chunks:
            writer.write_u8_chunk("weight_packed", packed)
            writer.write_u8_chunk("weight_scale", scales)
            del packed, scales
        writer.finish()
        os.fsync(stream.fileno())


def run(args: argparse.Namespace) -> None:
    """Time fixed matrices across budgets and require identical output hashes."""
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    if args.repetitions < 2 or args.warmups < 1:
        raise ValueError("Use at least one warmup and two measured repetitions")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite benchmark: {output}")
    emission_dir = Path(args.emission_dir) if args.emission_dir else None
    if emission_dir is not None:
        emission_dir.mkdir(parents=True, exist_ok=False)
    model_dir = Path(args.model_dir)
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    report: dict[str, Any] = {
        "schema_version": 1,
        "quantization_revision": QUANTIZATION_REVISION,
        "executed_source_sha256": source_digest(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "platform": platform.platform(),
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "row_limit": args.max_rows,
        "scope": (
            "real checkpoint matrix reads, host-to-device transfer, finite checks and "
            "quantization; excludes output disk writes and output hashing; "
            "not inference throughput or whole-checkpoint conversion time"
        ),
        "results": [],
    }
    if emission_dir is not None:
        report["scope"] = (
            "production row reads, host-to-device transfers, finite checks, quantization, "
            "GPU-to-CPU output copies and incremental safetensors writes including fsync; "
            "excludes source/shard hashes, rename journal, passthrough tensors and SQNR; "
            "not inference throughput or whole-checkpoint conversion time"
        )
    device = torch.device("cuda")
    for projection in args.projections:
        suffix = f"layers.0.mlp.{projection}.weight"
        names = [name for name in index if name.startswith("model.") and name.endswith(suffix)]
        if len(names) != 1:
            raise ValueError(f"Expected one source matrix ending in {suffix!r}: {names}")
        name = names[0]
        with (
            safe_open(str(model_dir / index[name]), framework="pt", device="cpu") as source,
            safe_open(args.calibration, framework="pt", device="cpu") as calibration,
        ):
            weight = source.get_slice(name)
            rows, columns = weight.get_shape()
            hessian = calibration.get_tensor(f"block-hessian::{name}").to(device)
            report.setdefault("input_matrices", {})[name] = {
                "shape": [rows, columns],
                "hessian_sha256": hashlib.sha256(hessian.cpu().numpy().tobytes()).hexdigest(),
            }

            read_rows = partial(_read_rows, weight, device)
            if emission_dir is not None:
                shard = ShardFile(model_dir / index[name], index)
                read_rows = partial(read_tensor_row_range, shard, name, device=device)
            for depth in args.clip_depths:
                baseline: tuple[str, str] | None = None
                for budget in args.element_budgets:
                    output_path = (
                        emission_dir / f"{projection}-depth{depth}-budget{budget}.safetensors"
                        if emission_dir is not None
                        else None
                    )
                    quantize = partial(quantize_mxfp4, hessian=hessian, mse_clip_depth=depth)

                    quantized_chunks = partial(
                        iter_quantized_row_chunks,
                        read_rows,
                        quantize,
                        rows=rows,
                        columns=columns,
                        max_rows=args.max_rows,
                        max_elements=budget,
                        name=name,
                    )

                    for _ in range(args.warmups):
                        _consume_chunks(quantized_chunks(), output_path, rows, columns)
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    samples = []
                    for _ in range(args.repetitions):
                        started = time.perf_counter()
                        _consume_chunks(quantized_chunks(), output_path, rows, columns)
                        torch.cuda.synchronize()
                        samples.append(time.perf_counter() - started)
                    peak_allocated = torch.cuda.max_memory_allocated()
                    peak_reserved = torch.cuda.max_memory_reserved()
                    packed_digest, scale_digest = hashlib.sha256(), hashlib.sha256()
                    if output_path is not None:
                        with safe_open(str(output_path), framework="pt", device="cpu") as emitted:
                            packed = emitted.get_tensor("weight_packed")
                            scales = emitted.get_tensor("weight_scale")
                            packed_digest.update(packed.numpy().tobytes())
                            scale_digest.update(scales.numpy().tobytes())
                            del packed, scales
                    else:
                        for _start, _stop, packed, scales in quantized_chunks():
                            packed_digest.update(packed.cpu().numpy().tobytes())
                            scale_digest.update(scales.cpu().numpy().tobytes())
                            del packed, scales
                    hashes = packed_digest.hexdigest(), scale_digest.hexdigest()
                    if baseline is None:
                        baseline = hashes
                    elif hashes != baseline:
                        raise AssertionError(f"Chunk budget changed quantized bytes: {name}")
                    result = {
                        "matrix": name,
                        "shape": [rows, columns],
                        "mse_clip_depth": depth,
                        "element_budget": budget,
                        "effective_rows": tensor_chunk_rows(columns, args.max_rows, budget),
                        "seconds": samples,
                        "median_seconds": statistics.median(samples),
                        "million_elements_per_second": (
                            rows * columns / statistics.median(samples) / 1e6
                        ),
                        "peak_cuda_allocated_bytes": peak_allocated,
                        "peak_cuda_reserved_bytes": peak_reserved,
                        "packed_sha256": hashes[0],
                        "scales_sha256": hashes[1],
                        "matches_smallest_budget": True,
                    }
                    report["results"].append(result)
                    _atomic_write_text(output, json.dumps(report, indent=2) + "\n")
                    print(json.dumps(result), flush=True)
                    torch.cuda.empty_cache()
            del hessian


def main() -> None:
    """Parse and run the isolated chunk-budget benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--emission-dir",
        help="New directory for production row reads and timed incremental output writes",
    )
    parser.add_argument("--projections", nargs="+", default=["down_proj", "gate_proj"])
    parser.add_argument("--clip-depths", type=int, nargs="+", default=[4, 8])
    parser.add_argument(
        "--element-budgets",
        type=int,
        nargs="+",
        default=[1_048_576, 4_194_304, 8_388_608, 33_554_432],
    )
    parser.add_argument("--max-rows", type=int, default=1024)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
