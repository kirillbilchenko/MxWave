"""Independently invert small triple-bitmap fixtures on CPU to locate packing errors."""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Any


def unpack_words(buffers: tuple[bytes, ...], rows: int, width: int, exponent: int,
                 *, wrap_exponent: bool = False) -> bytes:
    """Read each integer code and raw exception without floating-point conversion."""
    if sys.byteorder != "little" or len(buffers) != 7 or rows % 64 or width % 64:
        raise ValueError("Require little-endian aligned fixture storage")
    common, rare, *rest = buffers
    bitmaps = rest[:3]
    medium, global_offsets = rest[3:]
    output = array.array("H", [0]) * (rows * width)
    for tile_row in range(rows // 64):
        for tile_column in range(width // 64):
            tile = tile_row * (width // 64) + tile_column
            high_base, low_base = struct.unpack_from("<ii", global_offsets, tile * 8)
            for warp in range(4):
                high, low = struct.unpack_from("<ii", medium, (tile * 4 + warp) * 8)
                high += high_base
                low += low_base
                for small in range(16):
                    bitmap_index = tile * 64 + warp * 16 + small
                    words = [struct.unpack_from("<Q", b, bitmap_index * 8)[0] for b in bitmaps]
                    for position in range(64):
                        code = sum(((word >> position) & 1) << bit
                                   for bit, word in enumerate(words))
                        if code:
                            packed = common[high]
                            value = (packed & 127) | ((packed & 128) << 8)
                            restored_exponent = exponent + code
                            if wrap_exponent:
                                restored_exponent &= 255
                            value |= restored_exponent << 7
                            high += 1
                        else:
                            value = struct.unpack_from("<H", rare, low * 2)[0]
                            low += 1
                        row = tile_row * 64 + warp * 16 + (small % 2) * 8 + position // 8
                        column = tile_column * 64 + (small // 2) * 8 + position % 8
                        output[row * width + column] = value
    return output.tobytes()


def synthetic_matrices() -> list[Any]:
    """Cover the original raw words inside a fixture with a representable common exponent base."""
    import torch

    index = torch.arange(256 * 256, dtype=torch.int64)
    words = (index & 127) | (((index % 7) + 120) << 7) | (((index // 7) % 2) << 15)
    words[-16384:] = (torch.arange(16384, dtype=torch.int64) * 37) & 65535
    return [torch.zeros((64, 64), dtype=torch.bfloat16),
            words.to(torch.uint16).view(torch.bfloat16).reshape(256, 256),
            torch.randn((64, 256)).to(torch.bfloat16)]


def main() -> None:
    """Test the unchanged CPU packer without initializing CUDA or loading any model."""
    import torch
    from mxwave_zipserv import runtime

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("Preserve the first CPU packing diagnosis")
    torch.set_num_threads(1)
    torch.manual_seed(9417)
    underflow = ((torch.arange(128 * 128, dtype=torch.int64) * 37) & 65535).to(
        torch.uint16).view(torch.bfloat16).reshape(128, 128)
    matrices = [underflow, *synthetic_matrices()]
    report: dict[str, Any] = {"status": "running", "cuda_initialized": False,
        "model_weights_loaded": False, "cases": [],
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for index, cpu in enumerate(matrices):
        packed = runtime.compress(cpu)
        raw = tuple(b.numpy().tobytes() for b in packed.buffers)
        decoded = unpack_words(raw, packed.m, packed.k, packed.start_exp)
        wrapped = unpack_words(raw, packed.m, packed.k, packed.start_exp, wrap_exponent=True)
        expected = cpu.view(torch.uint8).numpy().tobytes()
        expected_words = struct.unpack("<" + "H" * cpu.numel(), expected)
        actual_words = struct.unpack("<" + "H" * cpu.numel(), decoded)
        bad = [i for i, (a, b) in enumerate(zip(expected_words, actual_words, strict=True))
               if a != b]
        report["cases"].append({"index": index, "shape": list(cpu.shape),
            "source_sha256": hashlib.sha256(expected).hexdigest(),
            "decoded_sha256": hashlib.sha256(decoded).hexdigest(), "mismatched_words": len(bad),
            "role": "retained_underflow_regression" if index == 0 else "gpu_smoke_fixture",
            "wrapped_decode_exact": wrapped == expected,
            "metadata": [packed.max_common, packed.max_rare, packed.start_exp],
            "examples": [{"index": i, "source": expected_words[i], "decoded": actual_words[i]}
                         for i in bad[:16]]})
    assert not torch.cuda.is_initialized()
    report.update(status="complete", cuda_initialized=False,
        all_source_words_exact=all(not row["mismatched_words"] for row in report["cases"]),
        all_wrapped_source_words_exact=all(row["wrapped_decode_exact"] for row in report["cases"]),
        supported_fixtures_exact=all(not row["mismatched_words"] and 0 <= row["metadata"][2] <= 248
                                    for row in report["cases"][1:]))
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
