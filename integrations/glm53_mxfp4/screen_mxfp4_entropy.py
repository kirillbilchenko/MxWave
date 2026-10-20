"""Read-only MXFP4 symbol-entropy and CPU rANS feasibility screen, not a GPU runtime codec."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any

SCALE_BITS = 12
TOTAL = 1 << SCALE_BITS
LOWER = 1 << 23


def frequencies(counts: list[int]) -> list[int]:
    """Normalize observed symbols to a positive integer frequency table summing to 4096."""
    if not any(counts) or any(c < 0 for c in counts) or len(counts) > TOTAL:
        raise ValueError("Require a nonempty nonnegative bounded alphabet")
    size = sum(counts)
    result = [max(1, c * TOTAL // size) if c else 0 for c in counts]
    order = sorted(range(len(counts)), key=lambda i: (counts[i] * TOTAL % size, counts[i]),
                   reverse=True)
    while sum(result) < TOTAL:
        for i in order:
            if counts[i] and sum(result) < TOTAL:
                result[i] += 1
    while sum(result) > TOTAL:
        i = max(range(len(result)), key=lambda j: result[j])
        if result[i] <= 1:
            raise ValueError("Alphabet does not fit normalization")
        result[i] -= 1
    return result


def encode(symbols: bytes, freq: list[int]) -> bytes:
    """Encode one independently initialized byte-renormalized rANS stream from its equations."""
    if sum(freq) != TOTAL or any(f < 0 for f in freq):
        raise ValueError("Invalid rANS table")
    cumulative, cursor = [], 0
    for f in freq:
        cumulative.append(cursor)
        cursor += f
    state, emitted = LOWER, bytearray()
    for symbol in reversed(symbols):
        f = freq[symbol]
        if not f:
            raise ValueError("Symbol has no probability")
        threshold = ((LOWER >> SCALE_BITS) << 8) * f
        while state >= threshold:
            emitted.append(state & 255)
            state >>= 8
        state = ((state // f) << SCALE_BITS) + state % f + cumulative[symbol]
    return struct.pack("<I", state) + bytes(reversed(emitted))


def decode(payload: bytes, size: int, freq: list[int]) -> bytes:
    """Independently invert rANS and require the exact terminal state and stream length."""
    table, cumulative, cursor = [], [], 0
    for symbol, f in enumerate(freq):
        cumulative.append(cursor)
        table.extend([symbol] * f)
        cursor += f
    if len(payload) < 4 or len(table) != TOTAL:
        raise ValueError("Invalid rANS stream/table")
    state, position = struct.unpack_from("<I", payload)[0], 4
    output = bytearray()
    for _ in range(size):
        slot = state & (TOTAL - 1)
        symbol = table[slot]
        output.append(symbol)
        state = freq[symbol] * (state >> SCALE_BITS) + slot - cumulative[symbol]
        while state < LOWER:
            if position >= len(payload):
                raise ValueError("Truncated rANS stream")
            state = (state << 8) | payload[position]
            position += 1
    if state != LOWER or position != len(payload):
        raise ValueError("rANS terminal state or payload length mismatch")
    return bytes(output)


def entropy(counts: list[int]) -> float:
    """Return zero-order entropy in bits per symbol, excluding representation overhead."""
    total = sum(counts)
    return -sum(c / total * math.log2(c / total) for c in counts if c)


def screen(raw: bytes, *, nibble: bool) -> dict[str, Any]:
    """Measure independent-tile streams with explicit state, index, table and raw-fallback costs."""
    if not raw or len(raw) % 2048:
        raise ValueError("Require complete 2048-byte sample tiles")
    symbols = bytes(s for byte in raw for s in (byte & 15, byte >> 4)) if nibble else raw
    counts = [symbols.count(i) for i in range(16 if nibble else 256)]
    freq = frequencies(counts)
    tile_symbols = 4096 if nibble else 2048
    result: dict[str, Any] = {"raw_bytes": len(raw), "symbols": len(symbols),
        "symbol_counts": counts, "entropy_bits_per_symbol": entropy(counts),
        "sample_sha256": hashlib.sha256(raw).hexdigest(), "stream_layouts": []}
    for lanes in (1, 32):
        retained, encoded_total, raw_tiles, tiles = len(freq) * 2, 0, 0, 0
        for first in range(0, len(symbols), tile_symbols):
            tile = symbols[first:first + tile_symbols]
            encoded = 0
            for lane in range(lanes):
                values = tile[lane::lanes]
                payload = encode(values, freq)
                if decode(payload, len(values), freq) != values:
                    raise RuntimeError("CPU rANS did not restore original codes/scales")
                encoded += len(payload)
            encoded_total += encoded
            # Four-byte tile offset and one raw/encoded tag per tile, plus
            # two-byte lane lengths for the warp-interleaved case.
            lane_index = 2 * lanes if lanes > 1 else 0
            compressed = encoded + lane_index
            raw_tiles += int(compressed >= 2048)
            retained += min(2048, compressed) + 5
            tiles += 1
        retained += 4  # final tile-offset sentinel
        result["stream_layouts"].append({"lanes": lanes, "tiles": tiles,
            "encoded_stream_bytes": encoded_total, "retained_bytes": retained,
            "saving_fraction": 1 - retained / len(raw), "raw_fallback_tiles": raw_tiles,
            "exact_roundtrip": True})
    return result


def read_sample(source: Path, index: dict[str, str], name: str) -> tuple[bytes, dict[str, Any]]:
    """Read four evenly spaced 16-KiB windows with validated shard-header offsets."""
    path = source / index[name]
    if path.name != index[name]:
        raise ValueError("Unsafe checkpoint shard path")
    with path.open("rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        if size > 32 * 1024**2:
            raise ValueError("Oversized safetensors header")
        header = f.read(size)
        spec = json.loads(header)[name]
        start, stop = spec["data_offsets"]
        if spec["dtype"] != "U8" or stop - start != math.prod(spec["shape"]):
            raise ValueError("Require declared U8 payload with consistent shape")
        if stop + 8 + size > path.stat().st_size:
            raise ValueError("Tensor interval exceeds the shard")
        window = min(16384, (stop - start) // 2048 * 2048)
        if not window:
            raise ValueError("Tensor too small for aligned tile sampling")
        positions = sorted({((stop - start - window) * i // 3) // 2048 * 2048 for i in range(4)})
        samples = []
        for position in positions:
            f.seek(8 + size + start + position)
            chunk = f.read(window)
            if len(chunk) != window:
                raise ValueError("Truncated checkpoint payload")
            samples.append(chunk)
    return b"".join(samples), {"tensor": name, "shard": path.name,
        "shard_header_sha256": hashlib.sha256(header).hexdigest(), "shape": spec["shape"],
        "window_bytes": window, "window_offsets": positions, "tensor_bytes": stop - start}


def main() -> None:
    """Screen representative main-model experts without CUDA, checkpoint changes or extrapolation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("Preserve prior compression measurements")
    config_bytes = (args.source / "config.json").read_bytes()
    index_bytes = (args.source / "model.safetensors.index.json").read_bytes()
    config, index = json.loads(config_bytes), json.loads(index_bytes)["weight_map"]
    quant = config["quantization_config"]
    groups = [g for g in quant["config_groups"].values()
              if g.get("format") == "mxfp4-pack-quantized"]
    if quant.get("quant_method") != "compressed-tensors" or len(groups) != 1:
        raise ValueError("Require a declared compressed-tensors MXFP4 group")
    weights = groups[0]["weights"]
    if (weights.get("num_bits"), weights.get("group_size"), weights.get("type")) != (4, 32, "float"):
        raise ValueError("Unexpected MXFP4 scheme")
    targets = set(groups[0]["targets"])
    reports = []
    for layer in (3, 22, 44):
        for expert in (0, 143, 287):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                module = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}"
                if module not in targets:
                    raise ValueError("Sampled expert is not explicitly configured MXFP4")
                for suffix, nibble in (("weight_packed", True), ("weight_scale", False)):
                    raw, metadata = read_sample(args.source, index, module + "." + suffix)
                    reports.append({**metadata, "kind": "codes" if nibble else "scales",
                                    **screen(raw, nibble=nibble)})
    result = {"status": "passed", "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_metadata": {"config.json": hashlib.sha256(config_bytes).hexdigest(),
                            "model.safetensors.index.json": hashlib.sha256(index_bytes).hexdigest()},
        "cases": reports, "limits": ["Stratified sample, not all experts or a checkpoint-size projection.",
            "Contiguous checkpoint windows, not the runtime Marlin tile layout.",
            "Per-tensor sample codebooks; GPU LUT, allocation alignment and decoder scratch excluded.",
            "One- and 32-stream rANS sizes do not predict GPU decode speed or serving throughput.",
            "Zero-order symbol entropy is not a bound on codecs exploiting longer correlations."]}
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "cases": len(reports)}))


if __name__ == "__main__":
    main()
