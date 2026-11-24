"""Bounded CPU audit of every downloaded base-expert scale and both TP2 partitions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np

PATTERN = re.compile(
    r"model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.weight_scale"
)


def main() -> None:
    """Read only verified files, one small scale at a time; never initialize CUDA."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    destination = root / "full-scale-audit.json"
    if destination.exists():
        raise FileExistsError("Preserve earlier audit evidence")
    groups: dict[tuple[int, int, str], dict[str, Any]] = {}
    seen: set[str] = set()
    state: dict[str, Any] = {"status": "waiting_for_verified_shards", "shards": [],
                             "tensors": 0, "scale_bytes": 0, "started_epoch": time.time()}
    digest = hashlib.sha256()

    def save(status: str) -> None:
        state.update(status=status, updated_epoch=time.time())
        state["groups"] = list(groups.values())
        temporary = destination.with_suffix(".incomplete")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(destination)

    try:
        deadline = time.monotonic() + 14400
        while time.monotonic() < deadline:
            download = json.loads((root / "download-state.json").read_bytes())
            for item in download["verified_shards"]:
                if item["name"] in seen:
                    continue
                path = root / "source" / item["name"]
                if path.stat().st_size != item["bytes"]:
                    raise ValueError("A verified source shard changed size")
                with path.open("rb") as stream:
                    length = struct.unpack("<Q", stream.read(8))[0]
                    if length > 32 * 1024**2:
                        raise ValueError("Unexpected safetensors header size")
                    header_bytes = stream.read(length)
                    header = json.loads(header_bytes)
                    count = 0
                    for name, spec in header.items():
                        match = PATTERN.fullmatch(name)
                        if match is None:
                            continue
                        layer, expert = int(match[1]), int(match[2])
                        projection = match[3]
                        if layer == 45:  # MTP has a separate qualification gate.
                            continue
                        if layer not in range(3, 45) or expert not in range(288):
                            raise ValueError("Unexpected base-expert coverage")
                        shape = [4096, 64] if projection == "down_proj" else [2048, 128]
                        if spec["dtype"] != "U8" or spec["shape"] != shape:
                            raise ValueError("Source expert scale is not canonical E8M0")
                        lo, hi = spec["data_offsets"]
                        if hi - lo != math.prod(shape):
                            raise ValueError("Scale tensor offsets and shape disagree")
                        stream.seek(8 + length + lo)
                        payload = stream.read(hi - lo)
                        if len(payload) != hi - lo:
                            raise ValueError("Truncated verified scale payload")
                        values = np.frombuffer(payload, dtype=np.uint8).reshape(shape)
                        digest.update(name.encode() + b"\0" + payload)
                        for rank in range(2):
                            part = (values[:, rank * 32:(rank + 1) * 32]
                                    if projection == "down_proj"
                                    else values[rank * 1024:(rank + 1) * 1024])
                            family = "w2" if projection == "down_proj" else "w13"
                            key = (layer, rank, family)
                            group = groups.setdefault(key, {"layer": layer, "rank": rank,
                                "family": family, "minimum": 255, "maximum": 0,
                                "matrices": 0, "raw_bytes": 0})
                            group["minimum"] = min(group["minimum"], int(part.min()))
                            group["maximum"] = max(group["maximum"], int(part.max()))
                            group["matrices"] += 1
                            group["raw_bytes"] += part.size
                            group["aligned_16_bucket"] = (
                                group["minimum"] // 16 == group["maximum"] // 16
                                and group["maximum"] != 255)
                        state["tensors"] += 1
                        state["scale_bytes"] += len(payload)
                        count += 1
                seen.add(item["name"])
                state["shards"].append({**item, "base_scale_tensors": count,
                    "header_sha256": hashlib.sha256(header_bytes).hexdigest()})
                save("auditing_verified_source")
            if len(seen) == 120:
                if state["tensors"] != 42 * 288 * 3 or len(groups) != 42 * 2 * 2:
                    raise ValueError("Full base-expert coverage is incomplete")
                for group in groups.values():
                    expected = 288 if group["family"] == "w2" else 576
                    if group["matrices"] != expected:
                        raise ValueError("Fused tensor coverage differs from the installed loader")
                state["scale_payload_sha256"] = digest.hexdigest()
                state["projected_scale_saving_bytes"] = sum(
                    g["raw_bytes"] // 2 - 16 for g in groups.values()
                    if g["aligned_16_bucket"])
                state["limitations"] = [("CPU source ranges only; layout and actual ownership "
                                         "are verified again by the GPU plugin."),
                                         "MTP, head, embedding and MXFP8 linears excluded."]
                save("complete")
                return
            if download["status"] == "failed":
                raise RuntimeError("Source download failed")
            save("waiting_for_verified_shards")
            time.sleep(10)
        raise TimeoutError("Full scale audit source deadline elapsed")
    except BaseException as exc:
        state["error"] = repr(exc)
        save("failed")
        raise


if __name__ == "__main__":
    main()
