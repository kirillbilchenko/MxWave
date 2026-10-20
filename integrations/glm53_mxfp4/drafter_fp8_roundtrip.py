"""CPU-only, scale-preserving FP8/BF16 checkpoint round-trip; no model execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import time
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

BLOCK = 128
SOURCE_SHA256 = "6cba1cff8ab4fcd657b89575f5db3debe1f5a39c78bda422953f553dc2357f8b"


def digest(path: Path) -> str:
    """Hash a file without retaining its contents."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def raw(tensor: torch.Tensor) -> bytes:
    """Return exact contiguous tensor bytes, including FP8 and BF16 encodings."""
    return tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()


def read_header(path: Path) -> tuple[bytes, dict[str, Any]]:
    """Read the original serialized header as well as its tensor descriptors."""
    with path.open("rb") as stream:
        prefix = stream.read(8)
        data = stream.read(struct.unpack("<Q", prefix)[0])
    return prefix + data, json.loads(data)


def convert_rows(
    values: torch.Tensor, scales: torch.Tensor, *, encode: bool,
) -> torch.Tensor:
    """Decode/encode whole 128-column blocks using the retained FP32 scales."""
    rows, columns = values.shape
    if rows % BLOCK or columns % BLOCK or scales.shape != (rows // BLOCK, columns // BLOCK):
        raise ValueError("Expected complete 128x128 weight blocks and matching scale grid")
    if not torch.isfinite(scales).all() or not (scales > 0).all():
        raise ValueError("Weight scales must be finite and positive")
    blocks = values.float().reshape(rows // BLOCK, BLOCK, columns // BLOCK, BLOCK)
    expanded = scales.float()[:, None, :, None]
    result = blocks / expanded if encode else blocks * expanded
    if not torch.isfinite(result).all():
        raise ValueError("Nonfinite converted weight")
    # BF16 rounding may put a recovered 448 code slightly above 448.
    # 464 is the midpoint toward the next (nonfinite) E4M3FN encoding.
    if encode and result.abs().max().item() >= 464:
        raise ValueError("FP8 overflow; refusing silent clipping or scale recalibration")
    dtype = torch.float8_e4m3fn if encode else torch.bfloat16
    converted = result.reshape(rows, columns).to(dtype)
    if not torch.isfinite(converted.float()).all():
        raise ValueError("FP8 overflow; refusing silent clipping or scale recalibration")
    return converted


def run(source: Path, output: Path, *, expected_sha256: str = SOURCE_SHA256) -> dict[str, Any]:
    """Write BF16 weights, reconstruct FP8 from them, and verify every payload."""
    started = time.monotonic()
    source_hash = digest(source / "model.safetensors")
    if source_hash != expected_sha256:
        raise ValueError("Source checkpoint hash differs from the frozen baseline")
    output.mkdir(parents=True, exist_ok=False)
    original_file = source / "model.safetensors"
    prefix, header = read_header(original_file)
    names = sorted(
        (name for name in header if name != "__metadata__"),
        key=lambda name: header[name]["data_offsets"][0],
    )
    fp8_names = [name for name in names if header[name]["dtype"] == "F8_E4M3"]
    scale_names = {name.removesuffix(".weight") + ".weight_scale_inv" for name in fp8_names}
    if any(name not in header or header[name]["dtype"] != "F32" for name in scale_names):
        raise ValueError("Missing FP32 block scales")
    dense_header: dict[str, Any] = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name in names:
        if name in scale_names:
            continue
        desc = dict(header[name])
        size = desc["data_offsets"][1] - desc["data_offsets"][0]
        if name in fp8_names:
            desc["dtype"] = "BF16"
            size *= 2
        desc["data_offsets"] = [offset, offset + size]
        dense_header[name] = desc
        offset += size
    encoded_header = json.dumps(dense_header, separators=(",", ":")).encode()
    encoded_header += b" " * (-len(encoded_header) % 8)
    trainable = output / "trainable"
    exported = output / "exported"
    trainable.mkdir()
    exported.mkdir()
    dense_file = trainable / "model.safetensors"
    with safe_open(original_file, framework="pt", device="cpu") as src, dense_file.open("xb") as dst:
        dst.write(struct.pack("<Q", len(encoded_header)))
        dst.write(encoded_header)
        for name in names:
            if name in scale_names:
                continue
            if name not in fp8_names:
                dst.write(raw(src.get_tensor(name)))
                continue
            scales = src.get_tensor(name.removesuffix(".weight") + ".weight_scale_inv")
            for start in range(0, header[name]["shape"][0], BLOCK):
                values = src.get_slice(name)[start:start + BLOCK, :]
                dst.write(raw(convert_rows(values, scales[start // BLOCK:start // BLOCK + 1],
                                           encode=False)))
    config_bytes = (source / "config.json").read_bytes()
    config = json.loads(config_bytes)
    config.pop("quantization_config")
    (trainable / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (exported / "config.json").write_bytes(config_bytes)
    rows: list[dict[str, Any]] = []
    exported_file = exported / "model.safetensors"
    with (
        safe_open(original_file, framework="pt", device="cpu") as src,
        safe_open(dense_file, framework="pt", device="cpu") as dense,
        exported_file.open("xb") as dst,
    ):
        # Preserve serialization, so file-byte equality can also be checked.
        # FP8 payloads below are always recomputed from the BF16 file.
        dst.write(prefix)
        for name in names:
            original_digest = hashlib.sha256()
            restored_digest = hashlib.sha256()
            mismatches = 0
            if name in fp8_names:
                scales = src.get_tensor(name.removesuffix(".weight") + ".weight_scale_inv")
                for start in range(0, header[name]["shape"][0], BLOCK):
                    values = dense.get_slice(name)[start:start + BLOCK, :]
                    recovered = convert_rows(
                        values, scales[start // BLOCK:start // BLOCK + 1], encode=True,
                    )
                    original = src.get_slice(name)[start:start + BLOCK, :]
                    mismatches += int((recovered.view(torch.uint8)
                                       != original.view(torch.uint8)).sum().item())
                    original_digest.update(raw(original))
                    payload = raw(recovered)
                    restored_digest.update(payload)
                    dst.write(payload)
            else:
                original = raw(src.get_tensor(name))
                payload = original if name in scale_names else raw(dense.get_tensor(name))
                original_digest.update(original)
                restored_digest.update(payload)
                mismatches = 0 if original == payload else sum(
                    a != b for a, b in zip(original, payload, strict=True)
                )
                dst.write(payload)
            rows.append({
                "name": name, "dtype": header[name]["dtype"], "shape": header[name]["shape"],
                "original_sha256": original_digest.hexdigest(),
                "exported_sha256": restored_digest.hexdigest(),
                "different_bytes": mismatches,
                "operation": "fp8_to_bf16_to_fp8" if name in fp8_names else (
                    "original_scale_retained" if name in scale_names else "bf16_copy"),
            })
    exported_hash = digest(exported_file)
    report = {
        "schema": "mxwave-drafter-fp8-roundtrip-v1", "cpu_only": True,
        "source_sha256": source_hash, "exported_sha256": exported_hash,
        "file_bytes_identical": source_hash == exported_hash,
        "config_bytes_identical": digest(source / "config.json") == digest(exported / "config.json"),
        "trainable_sha256": digest(dense_file), "trainable_bytes": dense_file.stat().st_size,
        "source_bytes": original_file.stat().st_size, "fp8_tensor_count": len(fp8_names),
        "fp8_element_count": sum(
            header[name]["shape"][0] * header[name]["shape"][1] for name in fp8_names
        ),
        "retained_scale_count": len(scale_names), "tensor_count": len(rows),
        "different_tensor_bytes": sum(row["different_bytes"] for row in rows),
        "elapsed_seconds": time.monotonic() - started,
        "torch_version": str(torch.__version__), "source_code_sha256": digest(Path(__file__)),
        "scale_policy": "Preserve original block scales; never estimate new scales or clip.",
        "limits": [
            "No parameters trained and no native inference executed.",
            "Recalibrating scales after updating backbone weights is not tested.",
            "BF16 training arithmetic is not proved equivalent to FP8 Marlin serving arithmetic.",
            "Original FP32 scales must accompany any export to this FP8 format.",
        ],
        "tensors": rows,
    }
    (output / "review.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    """Run the frozen no-training checkpoint conversion check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = run(args.source, args.output)
    print(json.dumps({key: value for key, value in report.items() if key != "tensors"}, indent=2))
    if not report["file_bytes_identical"] or not report["config_bytes_identical"]:
        raise SystemExit("Round-trip failed; review retained, no checkpoint promoted")


if __name__ == "__main__":
    main()
