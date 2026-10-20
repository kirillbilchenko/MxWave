"""Read the two known BF16 production vocabulary matrices and compare TP2 source bits."""

from __future__ import annotations

import hashlib
import json
import os
import struct
import time
from pathlib import Path


def main():
    os.nice(10)
    model = Path("/home/kirya/local-spark/models/glm-5.3-flash/derived/glm-quant-mix/lossless8")
    reference = Path(
        "/home/kirya/local-spark/experiments/glm53-mxfp4-audit-2026-10-15/"
        "full-model-v1/vocab-v2/prepared.json"
    )
    expected = json.loads(reference.read_bytes())["cases"]
    index = model / "model.safetensors.index.json"
    mapping = json.loads(index.read_bytes())["weight_map"]
    rows = []
    started = time.time()
    for kind, key in (("head", "lm_head.weight"),
                      ("embedding", "model.language_model.embed_tokens.weight")):
        path = model / mapping[key]
        before = path.stat()
        with path.open("rb") as handle:
            size = struct.unpack("<Q", handle.read(8))[0]
            header_bytes = handle.read(size)
            header = json.loads(header_bytes)[key]
            assert header["dtype"] == "BF16" and header["shape"] == [154880, 4096]
            begin, end = header["data_offsets"]
            half = 77440 * 4096 * 2
            assert end - begin == 2 * half
            handle.seek(8 + size + begin)
            for rank in (0, 1):
                digest = hashlib.sha256()
                remaining = half
                while remaining:
                    chunk = handle.read(min(16 * 1024**2, remaining))
                    assert chunk
                    digest.update(chunk)
                    remaining -= len(chunk)
                ref = next(r for r in expected if r["kind"] == kind and r["rank"] == rank)
                rows.append({
                    "kind": kind, "rank": rank, "source_key": key,
                    "path": str(path), "dtype": header["dtype"], "shape": [77440, 4096],
                    "source_sha256": digest.hexdigest(),
                    "ct_source_sha256": ref["source_sha256"],
                    "identical": digest.hexdigest() == ref["source_sha256"],
                    "header_sha256": hashlib.sha256(header_bytes).hexdigest(),
                })
        after = path.stat()
        assert (before.st_size, before.st_mtime_ns, before.st_ino) == (
            after.st_size, after.st_mtime_ns, after.st_ino
        )
    print(json.dumps({
        "schema": "mxwave-vocab-source-identity-v1",
        "scope": "Current checkpoint bytes, not a retroactive runtime tensor assertion",
        "started_epoch": started, "finished_epoch": time.time(),
        "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
        "index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": rows, "all_equal": all(x["identical"] for x in rows),
    }, indent=2))


if __name__ == "__main__":
    main()
