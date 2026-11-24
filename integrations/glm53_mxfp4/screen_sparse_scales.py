"""Qualify a sparse exception index on previously hash-bound complete Marlin scale groups."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scale_tiles import ScaleTiles, pack_sparse, unpack, unpack_sparse


def main() -> None:
    """Reuse decoded scale artifacts without reading weights or initializing CUDA."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "screen.json").exists():
        raise FileExistsError("Preserve previous sparse evidence")
    prior = json.loads(args.prior.read_bytes())
    if prior["status"] not in ("running", "passed"):
        raise ValueError("Require recorded CPU fixture qualifications")
    cases = []
    for row in prior["cases"]:
        if row["layer"] != 23:
            continue
        arm = next(f for f in row["formats"] if f["mode"] == "palette64")
        if not arm["cpu_roundtrip_exact"]:
            raise ValueError("Prior complete-layer fixture did not pass CPU correctness")
        path = args.prior.parent / arm["artifact"]
        if path.name != arm["artifact"] or hashlib.sha256(path.read_bytes()).hexdigest() != arm["artifact_sha256"]:
            raise ValueError("Prior complete scale fixture changed")
        with np.load(path, allow_pickle=False) as data:
            tiled = ScaleTiles(int(data["count"]), "palette64", tuple(int(v) for v in data["palette"]),
                               data["selectors"], data["offsets"], data["exceptions"])
        raw = unpack(tiled)
        if hashlib.sha256(raw).hexdigest() != row["raw_sha256"]:
            raise ValueError("Prior source reconstruction differs")
        packed = pack_sparse(raw)
        if not np.array_equal(unpack_sparse(packed), raw):
            raise RuntimeError("Sparse index changed source bytes")
        output = args.output / f"l23-r{row['rank']}-{row['family']}-sparse.npz"
        np.savez(output, selectors=packed.selectors,
                 exceptions=packed.exceptions, palette=np.array(packed.palette, dtype=np.uint8),
                 count=np.array(packed.count, dtype=np.int64))
        cases.append({**{k: row[k] for k in ("layer", "rank", "family", "shape", "raw_sha256",
            "raw_bytes", "aligned_nibble_retained_bytes", "aligned_nibble_allocated_estimate")},
            "palette": packed.palette, "affine_palette": bool(packed.descriptor & (1 << 24)),
            "selectors_bytes": packed.selectors.nbytes,
            "exception_bytes": packed.exceptions.nbytes, "exception_vectors": len(packed.exceptions),
            "retained_bytes": packed.retained_bytes,
            "allocated_estimate": packed.allocated_bytes(),
            "incremental_saving": 1 - packed.retained_bytes / row["aligned_nibble_retained_bytes"],
            "cpu_roundtrip_exact": True, "artifact": output.name,
            "artifact_sha256": hashlib.sha256(output.read_bytes()).hexdigest()})
    if len(cases) != 4:
        raise ValueError("Require both families and TP shards for the complete source layer")
    report = {"status": "passed", "format": "mxwave-palette16-v2", "cases": cases,
        "prior_status": prior["status"],
        "prior_sha256": hashlib.sha256(args.prior.read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "codec_sha256": hashlib.sha256(Path(pack_sparse.__code__.co_filename).read_bytes()).hexdigest(),
        "limits": ["Complete 288-expert layer; no whole-model memory or speed claim.",
                   "CPU-only reconstruction; GPU source-permutation equality remains required."]}
    (args.output / "screen.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "passed", "cases": len(cases), "incremental_savings":
                     [c["incremental_saving"] for c in cases]}))


if __name__ == "__main__":
    main()
