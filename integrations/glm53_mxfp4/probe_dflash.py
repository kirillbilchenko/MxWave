"""CPU-only lossless feasibility check of the already deployed DFlash checkpoint.

Keep serving untouched. Read BF16 tensors in bounded chunks, test row-addressable
selector codebooks, and screen small FP8 slices without inventing runtime savings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import runpy
import struct
import time
import zlib
from pathlib import Path
from typing import Any

from audit_checkpoint import digest, write_json


def measure(model: Path, codec_root: Path, output: Path) -> None:
    """Measure exact CPU coding and report byte counts independently of CUDA memory."""
    import numpy as np

    started = time.monotonic()
    config_raw = (model / "config.json").read_bytes()
    config = json.loads(config_raw)
    if config["architectures"] != ["DFlash2DraftModel"]:
        raise ValueError("Expected the deployed DFlash2 architecture")
    if config["quantization_config"]["quant_method"] != "fp8":
        raise ValueError("This probe expects the deployed mixed BF16/FP8 checkpoint")
    codec_paths = {
        "bf16": codec_root / "mxwave/lossless_bf16.py",
        "rows": codec_root / "mxwave/lossless_bf16_rows.py",
    }
    codecs = {name: runpy.run_path(str(path)) for name, path in codec_paths.items()}
    report: dict[str, Any] = {
        "status": "running",
        "device": "cpu",
        "model": str(model),
        "config_sha256": digest(config_raw),
        "script_sha256": digest(Path(__file__).read_bytes()),
        "codecs": {str(p): digest(p.read_bytes()) for p in codec_paths.values()},
        "files": [],
        "bf16_cases": [],
        "fp8_samples": [],
        "dtype_bytes": {},
        "limitations": [
            "Checkpoint byte counts, not measured retained GPU allocations or throughput.",
            "BF16 projection coding is a CPU feasibility format, without a fused linear adapter.",
            "256-column selector codebooks require GPU and torch.compile qualification.",
            "FP8 zlib samples are a generic CPU storage screen, not an inference codec.",
            "Target-owned shared head and embedding are not counted again for DFlash.",
        ],
    }
    for file in sorted(model.glob("*.safetensors")):
        with file.open("rb") as handle:
            header_bytes = struct.unpack("<Q", handle.read(8))[0]
            if not 0 < header_bytes <= 32 * 1024**2:
                raise ValueError("Invalid bounded DFlash safetensors header")
            header_raw = handle.read(header_bytes)
            header = json.loads(header_raw)
            report["files"].append(
                {"file": file.name, "header_sha256": digest(header_raw), "bytes": file.stat().st_size}
            )
            for name, spec in header.items():
                if name == "__metadata__":
                    continue
                start, end = spec["data_offsets"]
                size = end - start
                dtype = spec["dtype"]
                report["dtype_bytes"][dtype] = report["dtype_bytes"].get(dtype, 0) + size
                handle.seek(8 + header_bytes + start)
                if dtype == "BF16":
                    source_hash = hashlib.sha256()
                    case: dict[str, Any] = {"name": name, "shape": spec["shape"], "raw_bytes": size}
                    if name.endswith(("predecessor_codebook", "successor_codebook")):
                        raw = handle.read(size)
                        if len(raw) != size:
                            raise ValueError("Truncated DFlash selector codebook")
                        source_hash.update(raw)
                        words = np.frombuffer(raw, dtype="<u2").reshape(spec["shape"])
                        if words.shape[1] != 256:
                            raise ValueError("Expected the deployed 256-wide selector codebook")
                        packed_rows = codecs["rows"]["pack_rows"](
                            words, block_size=256, chunk_rows=512, source_sha256=digest(raw)
                        )
                        restored_hash = hashlib.sha256()
                        for first in range(0, len(words), 512):
                            last = min(first + 512, len(words))
                            restored = packed_rows.row_words(np.arange(first, last))
                            if not np.array_equal(restored, words[first:last]):
                                raise ValueError("Selector row codec changed BF16 bits")
                            restored_hash.update(restored.tobytes())
                        if restored_hash.digest() != source_hash.digest():
                            raise ValueError("Selector full reconstruction hash differs")
                        case.update(
                            codec="row-addressable-bf16-block256",
                            retained_bytes=packed_rows.retained_bytes,
                            exact_all_rows=True,
                        )
                        del packed_rows, words, raw
                    else:
                        retained = 0
                        for offset in range(0, size, 2 * 1024**2):
                            count = min(size - offset, 2 * 1024**2)
                            raw = handle.read(count)
                            if len(raw) != count:
                                raise ValueError("Truncated DFlash BF16 tensor")
                            source_hash.update(raw)
                            packed = codecs["bf16"]["pack_bf16_bytes"](raw, block_size=512)
                            if codecs["bf16"]["unpack_bf16_bytes"](packed) != raw:
                                raise ValueError("BF16 chunk codec changed bits")
                            retained += min(len(raw), packed.storage_bytes)
                        case.update(codec="chunked-bf16-exponent-or-raw", retained_bytes=retained)
                    case.update(
                        source_sha256=source_hash.hexdigest(),
                        byte_exact=True,
                        saving_fraction=1 - case["retained_bytes"] / size,
                    )
                    report["bf16_cases"].append(case)
                    print(f"BF16 codec exact: {name}", flush=True)
                elif dtype == "F8_E4M3" and any(
                    prefix in name for prefix in ("layers.0.", "layers.4.")
                ):
                    count = min(size, 1024**2)
                    for relative in (0, (size - count) // 2, size - count):
                        handle.seek(8 + header_bytes + start + relative)
                        raw = handle.read(count)
                        if len(raw) != count:
                            raise ValueError("Truncated FP8 screen slice")
                        coded = zlib.compress(raw, level=1)
                        if zlib.decompress(coded) != raw:
                            raise ValueError("Generic FP8 storage screen changed bytes")
                        report["fp8_samples"].append(
                            {
                                "name": name,
                                "relative_offset": relative,
                                "bytes": count,
                                "sha256": digest(raw),
                                "zlib1_bytes": len(coded),
                                "saving_fraction": 1 - len(coded) / count,
                            }
                        )
    cases = report["bf16_cases"]
    raw_bytes = sum(c["raw_bytes"] for c in cases)
    retained_bytes = sum(c["retained_bytes"] for c in cases)
    total_bytes = sum(report["dtype_bytes"].values())
    report.update(
        status="complete",
        checkpoint_payload_bytes=total_bytes,
        bf16_raw_bytes=raw_bytes,
        bf16_retained_cpu_bytes=retained_bytes,
        bf16_saved_bytes=raw_bytes - retained_bytes,
        bf16_saving_fraction=1 - retained_bytes / raw_bytes,
        checkpoint_saving_fraction=(raw_bytes - retained_bytes) / total_bytes,
        seconds=time.monotonic() - started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    )
    write_json(output, report)
    print(
        json.dumps(
            {key: report[key] for key in ("status", "bf16_saved_bytes", "checkpoint_saving_fraction")}
        )
    )


def main() -> None:
    """Run a bounded, read-only feasibility probe against the deployed draft weights."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--codec-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    measure(args.model, args.codec_root, args.output)


if __name__ == "__main__":
    main()
