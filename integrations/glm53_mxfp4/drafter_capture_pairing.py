"""Pair independent rank exports before releasing verified temporary staging."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def compare_ranks(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    """Require bit-identical independently audited exports before authorizing staging release."""
    if (first["status"] != "rank_verified" or second["status"] != "rank_verified"
            or first["rank"] != 0 or second["rank"] != 1
            or any(first[k] != second[k] for k in ("start", "stop", "plan_sha256", "source_sha256"))
            or len(first["rows"]) != first["stop"] - first["start"]):
        raise ValueError("Rank chunk receipts disagree")
    def comparable(row: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in row.items() if k != "raw_frames"}

    if [comparable(r) for r in first["rows"]] != [comparable(r) for r in second["rows"]]:
        raise ValueError("Rank accepted feature payloads or histories differ")
    return {
        "status": "paired_verified", "start": first["start"], "stop": first["stop"],
        "plan_sha256": first["plan_sha256"], "rows": [comparable(r) for r in first["rows"]],
        "all_rank_features_bitwise_equal": True, "sealed_test_opened": False,
    }


def file_digest(path: Path) -> str:
    """Hash payloads using only the host standard library."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pair_chunk(root: Path, start: int, stop: int) -> dict[str, Any]:
    """Bind paired receipts and the frozen plan without touching raw feature files."""
    paths = [root / f"export-{start:04d}-{stop:04d}-r{rank}.json" for rank in (0, 1)]
    report = compare_ranks(*(json.loads(p.read_bytes()) for p in paths))
    if report["plan_sha256"] != file_digest(root / "plan.json"):
        raise ValueError("Rank receipts refer to another capture plan")
    report["rank_receipts"] = {p.name: file_digest(p) for p in paths}
    path = root / f"paired-{start:04d}-{stop:04d}.json"
    with path.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    return report


def release_staging(root: Path, rank: int, start: int, stop: int) -> dict[str, Any]:
    """Delete only checked raw payloads after equal audited exports exist on both ranks."""
    plan = json.loads((root / "plan.json").read_bytes())
    if rank not in (0, 1) or not plan.get("release_verified_raw_staging", False):
        raise ValueError("This plan does not authorize temporary frame release")
    path = root / f"paired-{start:04d}-{stop:04d}.json"
    paired = json.loads(path.read_bytes())
    ranks = []
    for other in (0, 1):
        receipt = root / f"export-{start:04d}-{stop:04d}-r{other}.json"
        if file_digest(receipt) != paired["rank_receipts"][receipt.name]:
            raise ValueError("Rank receipt changed before staging release")
        ranks.append(json.loads(receipt.read_bytes()))
    expected = compare_ranks(*ranks)
    if (any(paired[key] != value for key, value in expected.items())
            or paired["plan_sha256"] != file_digest(root / "plan.json")):
        raise ValueError("Paired export or capture plan changed")
    destination = root / f"released-{start:04d}-{stop:04d}-r{rank}.json"
    if destination.exists():
        raise FileExistsError(destination)
    cases = plan["calibration_cases"][start:stop]
    if [row["id"] for row in ranks[rank]["rows"]] != [case["id"] for case in cases]:
        raise ValueError("Release range differs from frozen capture order")
    pending = []
    for row in ranks[rank]["rows"]:
        exported = root / "accepted" / row["path"]
        if (Path(row["path"]).name != row["path"] or file_digest(exported) != row["sha256"]):
            raise ValueError("Accepted export changed; preserve raw staging")
        for index, frame in enumerate(row["raw_frames"]):
            relative = f"ct-r{rank}/context-{row['id']}-{index:04d}-r{rank}.safetensors"
            if frame["path"] != relative or not (root / relative).resolve().is_relative_to(root.resolve()):
                raise ValueError("Raw frame path is outside the owned sequential capture")
            raw = root / relative
            if (raw.is_symlink() or raw.stat().st_size != frame["bytes"]
                    or file_digest(raw) != frame["sha256"]
                    or file_digest(raw.with_suffix(".json")) != frame["metadata_sha256"]):
                raise ValueError("Raw frame changed; preserve staging")
            pending.append(raw)
    # Validate the entire chunk before the first unlink. Metadata and audits remain.
    released_bytes = sum(p.stat().st_size for p in pending)
    for raw in pending:
        raw.unlink()
    report = {"status": "released", "rank": rank, "start": start, "stop": stop,
              "paired_sha256": file_digest(path), "payloads": len(pending),
              "bytes": released_bytes, "accepted_exports_preserved": True}
    destination.write_text(json.dumps(report, indent=2) + "\n")
    return report

