"""Independently verify and compare retained activation bytes without GPU or Torch."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def read_tensor(root: Path, item: dict[str, Any]) -> tuple[bytes, np.ndarray]:
    """Verify exact bytes and decode only supported diagnostic activation dtypes."""
    name = item["file"]
    if Path(name).name != name:
        raise ValueError("Snapshot file must be local to its manifest")
    raw = (root / name).read_bytes()
    if len(raw) != item["nbytes"] or hashlib.sha256(raw).hexdigest() != item["sha256"]:
        raise ValueError("Snapshot bytes changed")
    if item["dtype"] == "torch.bfloat16":
        values = (np.frombuffer(raw, dtype="<u2").astype("<u4") << 16).view("<f4")
    else:
        dtypes = {"torch.float32": "<f4", "torch.float16": "<f2", "torch.int32": "<i4",
                  "torch.int64": "<i8", "torch.uint8": "u1", "torch.bool": "?"}
        if item["dtype"] not in dtypes:
            raise ValueError("Unsupported snapshot dtype")
        values = np.frombuffer(raw, dtype=dtypes[item["dtype"]])
    if len(values) != math.prod(item["shape"]):
        raise ValueError("Snapshot shape disagrees with its byte length")
    return raw, values


def compare_snapshots(left: Path, right: Path) -> dict[str, Any]:
    """Report the first changed boundary and numerical magnitude, checking every artifact hash."""
    reports = [json.loads((p / "snapshots.json").read_bytes()) for p in (left, right)]
    if any(r["status"] != "complete" for r in reports):
        raise ValueError("Cannot compare incomplete snapshots")
    events = []
    for a, b in zip(reports[0]["events"], reports[1]["events"], strict=True):
        if a["name"] != b["name"]:
            raise ValueError("Captured operation ordering changed")
        changed = []
        for x, y in zip(a["tensors"], b["tensors"], strict=True):
            if any(x[k] != y[k] for k in ("path", "shape", "dtype")):
                raise ValueError("Captured tensor identity changed")
            raw_a, va = read_tensor(left, x)
            raw_b, vb = read_tensor(right, y)
            if raw_a == raw_b:
                continue
            delta = va.astype(np.float64) - vb.astype(np.float64)
            changed.append({
                "path": x["path"], "shape": x["shape"], "dtype": x["dtype"],
                "differing_bytes": int(np.count_nonzero(
                    np.frombuffer(raw_a, dtype="u1") != np.frombuffer(raw_b, dtype="u1"))),
                "different_values": int(np.count_nonzero(va != vb)),
                "elements": len(va), "all_finite": bool(np.isfinite(va).all()
                                                      and np.isfinite(vb).all()),
                "max_abs": float(np.max(np.abs(delta))) if len(delta) else 0.0,
                "rmse": float(np.sqrt(np.mean(delta**2))) if len(delta) else 0.0,
            })
        events.append({"name": a["name"], "bits_exact": not changed, "changed": changed})
    first = next((row for row in events if not row["bits_exact"]), None)
    return {"status": "complete", "first_changed_boundary": first, "events": events,
            "limits": ["Snapshots add GPU clones and change scheduling.",
                       "First differing bytes locate an observed boundary, not a proven root cause."]}


def main() -> None:
    """Analyze only fresh, bounded fixtures belonging to this rank and diagnostic revision."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", required=True, type=Path)
    parser.add_argument("--rank", required=True, choices=(0, 1), type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve prior diagnostic analysis")
    report = {"status": "complete", "rank": args.rank, "comparisons": [
        compare_snapshots(args.traces / f"repeat-default-0-rank{args.rank}",
                          args.traces / f"repeat-default-{i}-rank{args.rank}")
        for i in (1, 2)
    ]}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"rank": args.rank, "first_changed": [
        r["first_changed_boundary"] for r in report["comparisons"]]}))


if __name__ == "__main__":
    main()
