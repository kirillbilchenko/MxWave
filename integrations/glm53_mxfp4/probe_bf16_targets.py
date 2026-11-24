"""Bounded CPU sample of lossless BF16 coding on active, indexed GLM text weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import resource
import runpy
import struct
import time
from collections import Counter
from pathlib import Path
from typing import Any


def family(name: str, layers: int) -> str:
    """Classify checkpoint tensors without interpreting a quantization format."""
    if ".visual." in name:
        return "vision"
    match = re.search(r"\.layers\.(\d+)\.", name)
    if match and int(match[1]) >= layers:
        return "mtp"
    if "embed_tokens" in name:
        return "embedding"
    if "lm_head" in name:
        return "head"
    if ".self_attn." in name:
        return "attention"
    if ".mlp.experts." in name:
        return "experts"
    if ".mlp.shared_experts." in name:
        return "shared"
    if ".mlp.gate." in name:
        return "router"
    return "other"


def sample_regions(rows: int, count: int = 32) -> list[tuple[int, int]]:
    """Select disjoint merged first/middle/last row regions deterministically."""
    count = min(rows, count)
    result: list[tuple[int, int]] = []
    for start in sorted({0, (rows - count) // 2, rows - count}):
        end = start + count
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def main() -> None:
    """Measure both existing codecs; report sampled bytes separately from extrapolation."""
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--codec-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("Preserve prior feasibility reports")
    started = time.monotonic()
    config = json.loads((args.model / "config.json").read_bytes())
    index = json.loads((args.model / "model.safetensors.index.json").read_bytes())["weight_map"]
    layers = config["text_config"]["num_hidden_layers"]
    headers: dict[str, Any] = {}
    physical_bytes, superseded_bytes, physical_entries = 0, 0, 0
    header_hashes = {}
    for shard in sorted(set(index.values())):
        with (args.model / shard).open("rb") as handle:
            length = struct.unpack("<Q", handle.read(8))[0]
            if not 0 < length <= 64 * 1024**2:
                raise ValueError("Invalid bounded safetensors header")
            raw = handle.read(length)
        header_hashes[shard] = hashlib.sha256(raw).hexdigest()
        for name, spec in json.loads(raw).items():
            if name == "__metadata__":
                continue
            size = spec["data_offsets"][1] - spec["data_offsets"][0]
            physical_bytes += size
            physical_entries += 1
            if index.get(name) != shard:
                superseded_bytes += size
                continue
            headers[name] = {**spec, "shard": shard, "payload_offset": 8 + length}
    if set(headers) != set(index):
        raise ValueError("Active index/header coverage differs")
    codec_paths = {key: args.codec_dir / file for key, file in {
        "flat": "lossless_bf16.py", "rows": "lossless_bf16_rows.py"}.items()}
    codecs = {key: runpy.run_path(str(path)) for key, path in codec_paths.items()}
    totals: Counter[str] = Counter()
    cases = []
    for name, spec in sorted(headers.items()):
        group = family(name, layers)
        size = spec["data_offsets"][1] - spec["data_offsets"][0]
        totals[group + "/" + spec["dtype"]] += size
        shape = spec["shape"]
        if (group not in ("attention", "head", "embedding", "router")
                or spec["dtype"] != "BF16" or len(shape) != 2 or size < 65536):
            continue
        rows, width = shape
        if size != rows * width * 2:
            raise ValueError("BF16 matrix payload size differs")
        regions = sample_regions(rows)
        parts = []
        with (args.model / spec["shard"]).open("rb") as handle:
            for start, end in regions:
                offset = spec["payload_offset"] + spec["data_offsets"][0] + start * width * 2
                handle.seek(offset)
                raw = handle.read((end - start) * width * 2)
                if len(raw) != (end - start) * width * 2:
                    raise ValueError("Truncated BF16 sample")
                parts.append(raw)
        raw = b"".join(parts)
        sample_sha = hashlib.sha256(raw).hexdigest()
        packed = codecs["flat"]["pack_bf16_bytes"](raw, block_size=512)
        if codecs["flat"]["unpack_bf16_bytes"](packed) != raw:
            raise ValueError("Flat codec changed BF16 sample bits")
        case = {"name": name, "family": group, "shape": shape, "source_bytes": size,
            "shard": spec["shard"], "row_regions": regions,
            "sample_sha256": sample_sha, "sample_bytes": len(raw),
            "flat_encoded_bytes": packed.storage_bytes, "flat_byte_exact": True,
            "flat_saving_fraction": 1 - packed.storage_bytes / len(raw)}
        if width % 512 == 0:
            words = np.frombuffer(raw, dtype="<u2").reshape(-1, width)
            row_packed = codecs["rows"]["pack_rows"](
                words, block_size=512, chunk_rows=32, source_sha256=sample_sha)
            if row_packed.row_words(np.arange(len(words))).tobytes() != raw:
                raise ValueError("Row codec changed BF16 sample bits")
            case.update(rows_encoded_bytes=row_packed.retained_bytes, rows_byte_exact=True,
                        rows_saving_fraction=1 - row_packed.retained_bytes / len(raw))
        cases.append(case)
    summaries = {}
    for group in ("attention", "head", "embedding", "router"):
        selected = [case for case in cases if case["family"] == group]
        if not selected:
            continue
        sample_bytes = sum(case["sample_bytes"] for case in selected)
        summaries[group] = {"matrices": len(selected), "source_bytes": sum(
            case["source_bytes"] for case in selected), "sample_bytes": sample_bytes}
        for key in ("flat", "rows"):
            eligible = [case for case in selected if key + "_encoded_bytes" in case]
            measured = sum(case["sample_bytes"] for case in eligible)
            coded = sum(case[key + "_encoded_bytes"] for case in eligible)
            summaries[group][key] = {"matrices": len(eligible),
                "sample_saving_fraction": 1 - coded / measured if measured else None,
                "sample_encoded_bytes": coded,
                "estimated_saved_source_bytes": sum(case["source_bytes"] *
                    case[key + "_saving_fraction"] for case in eligible),
                "minimum_matrix_sample_saving": min(
                    (case[key + "_saving_fraction"] for case in eligible), default=None),
                "maximum_matrix_sample_saving": max(
                    (case[key + "_saving_fraction"] for case in eligible), default=None)}
    report = {"status": "complete", "device": "cpu", "model": str(args.model),
        "quant_method": config["quantization_config"]["quant_method"],
        "source_sha256": {name: hashlib.sha256((args.model / name).read_bytes()).hexdigest()
                          for name in ("config.json", "model.safetensors.index.json")},
        "code_sha256": {str(path.name): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in (*codec_paths.values(), Path(__file__))},
        "header_sha256": header_hashes, "active_tensor_count": len(headers),
        "physical_tensor_entries": physical_entries, "physical_payload_bytes": physical_bytes,
        "superseded_or_unindexed_payload_bytes": superseded_bytes,
        "active_payload_bytes": sum(totals.values()), "active_bytes_by_family_dtype": dict(totals),
        "summary": summaries, "cases": cases, "seconds": time.monotonic() - started,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "limitations": ["Deterministic first/middle/last rows, not a random statistical estimate.",
            "Extrapolation concerns source tensors, not TP2 resident allocation or runtime speed.",
            "Each codec size includes its sample header/indexes; full tensor sizing is unmeasured.",
            "BF16 linear adapters still require GPU arithmetic, scratch and graph qualification."]}
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "summary": summaries,
                      "seconds": report["seconds"], "peak_rss_bytes": report["peak_rss_bytes"]}))


if __name__ == "__main__":
    main()
