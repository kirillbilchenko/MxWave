"""CPU-only comparison of generic scale tiles with actual TP2 Marlin-aligned nibble storage."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np
from scale_tiles import pack, pack_sparse, unpack, unpack_sparse

CONFIG_SHA = "194d290ebb0f19aedc60d24ea9e236e33c86616e460f4a108a842f6fd5b8b198"
INDEX_SHA = "6d1364a27eef78675390422b1b5230601817fb7460ce816951b2cd515934c85d"


def marlin_order(raw: np.ndarray) -> np.ndarray:
    """Express the stock BF16-activation permutation as transpose and four-value swaps."""
    transposed = raw.T.copy()
    permuted = transposed.reshape(-1, 8, 8).transpose(0, 2, 1).reshape(-1, 4)
    return permuted[:, [0, 2, 1, 3]].reshape(raw.shape[1], raw.shape[0]).copy()


def write(path: Path, value: Any) -> None:
    """Atomically retain progress without replacing earlier completed evidence."""
    temporary = path.with_suffix(".incomplete")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    """Read complete expert groups, verify both codecs, and preserve selected GPU fixtures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", nargs="+", type=int, default=[3, 23, 44])
    parser.add_argument("--palette-only", action="store_true",
                        help="Audit exact direct-palette records; do not emit large artifacts")
    args = parser.parse_args()
    if (len(set(args.layers)) != len(args.layers)
            or any(layer not in range(3, 45) for layer in args.layers)):
        parser.error("Require distinct main routed-expert layers")
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "screen.json"
    if report_path.exists():
        raise FileExistsError("Preserve prior CPU screen")
    config_bytes = (args.source / "config.json").read_bytes()
    index_bytes = (args.source / "model.safetensors.index.json").read_bytes()
    if (hashlib.sha256(config_bytes).hexdigest() != CONFIG_SHA
            or hashlib.sha256(index_bytes).hexdigest() != INDEX_SHA):
        raise ValueError("Audited model metadata changed")
    config = json.loads(config_bytes)["quantization_config"]
    groups = [g for g in config["config_groups"].values()
              if g.get("format") == "mxfp4-pack-quantized"]
    if config["quant_method"] != "compressed-tensors" or len(groups) != 1:
        raise ValueError("Require config-declared MXFP4")
    targets = set(groups[0]["targets"])
    index = json.loads(index_bytes)["weight_map"]
    headers = {}
    provenance = hashlib.sha256()
    source_tensors = 0

    def read(layer: int, expert: int, projection: str) -> np.ndarray:
        nonlocal source_tensors
        module = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}"
        if module not in targets:
            raise ValueError("Source module is not targeted MXFP4")
        name = module + ".weight_scale"
        shard = index[name]
        path = args.source / shard
        if path.name != shard:
            raise ValueError("Unsafe shard path")
        with path.open("rb") as f:
            if shard not in headers:
                size = struct.unpack("<Q", f.read(8))[0]
                if size > 32 * 1024**2:
                    raise ValueError("Oversized tensor header")
                payload = f.read(size)
                headers[shard] = size, json.loads(payload), hashlib.sha256(payload).hexdigest()
            size, header, _ = headers[shard]
            spec = header[name]
            shape = [4096, 64] if projection == "down_proj" else [2048, 128]
            lo, hi = spec["data_offsets"]
            if spec["dtype"] != "U8" or spec["shape"] != shape or hi - lo != 262144:
                raise ValueError("Scale payload does not match audited shape")
            f.seek(8 + size + lo)
            payload = f.read(hi - lo)
            if len(payload) != hi - lo:
                raise ValueError("Truncated scale payload")
        provenance.update(name.encode() + b"\0" + payload)
        source_tensors += 1
        return np.frombuffer(payload, dtype=np.uint8).reshape(shape)

    report: dict[str, Any] = {"status": "running", "layers": args.layers, "cases": [],
        "palette_only": args.palette_only,
        "all_main_expert_layers": set(args.layers) == set(range(3, 45)),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "codec_sha256": hashlib.sha256(Path(pack.__code__.co_filename).read_bytes()).hexdigest(),
        "config_sha256": CONFIG_SHA, "index_sha256": INDEX_SHA,
        "gate": {"minimum_incremental_scale_saving": .20,
                 "gpu_component_maximum_latency_ratio": 1.02},
        "limits": ["CPU reconstruction of actual Marlin order; GPU equality still pending.",
            "Complete scale groups for listed layers; other model tensors are unchanged and unaudited here.",
            "Exact CPU weight bytes do not prove full-model arithmetic equivalence.",
            "Allocator estimates use 512-byte rounding; actual CUDA allocation must be measured."]}
    started = time.monotonic()
    write(report_path, report)
    for layer in args.layers:
        if layer not in range(3, 45):
            raise ValueError("Main routed-expert layer required")
        arrays = {(rank, family): np.empty((288, 128 if family == "w13" else 32,
                                            2048 if family == "w13" else 4096), dtype=np.uint8)
                  for rank in (0, 1) for family in ("w13", "w2")}
        for expert in range(288):
            gate, up, down = (read(layer, expert, p) for p in ("gate_proj", "up_proj", "down_proj"))
            for rank in (0, 1):
                part = slice(rank * 1024, (rank + 1) * 1024)
                arrays[rank, "w13"][expert] = marlin_order(np.concatenate((gate[part], up[part])))
                arrays[rank, "w2"][expert] = marlin_order(down[:, rank * 32:(rank + 1) * 32])
        for (rank, family), raw in arrays.items():
            flat = raw.reshape(-1)
            minimum, maximum = int(flat.min()), int(flat.max())
            aligned = minimum // 16 == maximum // 16
            baseline = raw.nbytes // 2 + 16 if aligned else raw.nbytes
            row: dict[str, Any] = {"layer": layer, "rank": rank, "family": family,
                "shape": list(raw.shape), "raw_bytes": raw.nbytes,
                "raw_sha256": hashlib.sha256(flat).hexdigest(),
                "baseline_encoding": "aligned_nibbles" if aligned else "raw_e8m0",
                "baseline_retained_bytes": baseline,
                "aligned_nibble_retained_bytes": baseline if aligned else None,
                "aligned_nibble_allocated_estimate": ((raw.nbytes // 2 + 511) // 512 * 512 + 512
                                                      if aligned else None),
                "minimum": minimum, "maximum": maximum,
                "formats": []}
            for mode in (("palette16",) if args.palette_only else ("mask128", "palette64")):
                packed = pack_sparse(flat) if args.palette_only else pack(flat, mode)
                decoded = unpack_sparse(packed) if args.palette_only else unpack(packed)
                if not np.array_equal(decoded, flat):
                    raise RuntimeError("Scale tile CPU decoder changed source bytes")
                item = {"mode": mode, "palette": packed.palette,
                    "retained_bytes": packed.retained_bytes, "allocated_estimate": packed.allocated_bytes(),
                    "incremental_saving": 1 - packed.retained_bytes / baseline,
                    "exceptions_bytes": packed.exceptions.nbytes,
                    "selectors_bytes": packed.selectors.nbytes,
                    "offsets_bytes": 0 if args.palette_only else packed.offsets.nbytes,
                    "cpu_roundtrip_exact": True}
                if layer == 23 and not args.palette_only:
                    path = args.output / f"l{layer}-r{rank}-{family}-{mode}.npz"
                    np.savez(path, selectors=packed.selectors, offsets=packed.offsets,
                             exceptions=packed.exceptions, palette=np.array(packed.palette, dtype=np.uint8),
                             count=np.array(packed.count, dtype=np.int64), mode=np.array(mode))
                    item.update(artifact=path.name, artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                row["formats"].append(item)
                del decoded, packed
            best = min(row["formats"], key=lambda item: item["allocated_estimate"])
            row["selected"] = best["mode"] if best["retained_bytes"] < baseline else row["baseline_encoding"]
            report["cases"].append(row)
            write(report_path, report)
            print(json.dumps({k: row[k] for k in ("layer", "rank", "family", "selected")}))
        del arrays
    if source_tensors != len(args.layers) * 288 * 3 or len(report["cases"]) != len(args.layers) * 4:
        raise RuntimeError("Expert-scale coverage is incomplete")
    report.update(status="passed", seconds=time.monotonic() - started,
                  source_scale_tensors=source_tensors,
                  source_scale_bytes_sha256=provenance.hexdigest(),
                  shard_headers={k: v[2] for k, v in headers.items()})
    write(report_path, report)


if __name__ == "__main__":
    main()
