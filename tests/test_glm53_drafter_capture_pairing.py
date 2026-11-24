"""Raw staging release must preserve exports and fail before deletion on broken evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from integrations.glm53_mxfp4.drafter_capture_pairing import (
    file_digest,
    pair_chunk,
    release_staging,
)


def prepared(tmp_path: Path) -> list[Path]:
    """Create two independently bound tiny rank receipts and a retained export."""
    (tmp_path / "plan.json").write_text(json.dumps({
        "release_verified_raw_staging": True, "calibration_cases": [{"id": "one"}],
    }))
    (tmp_path / "accepted").mkdir()
    exported = tmp_path / "accepted/train-one.safetensors"
    exported.write_bytes(b"accepted")
    raw_paths = []
    for rank in (0, 1):
        (tmp_path / f"ct-r{rank}").mkdir()
        path = tmp_path / f"ct-r{rank}/context-one-0000-r{rank}.safetensors"
        path.write_bytes(b"raw features")
        path.with_suffix(".json").write_text("{}")
        raw_paths.append(path)
        row = {"id": "one", "path": exported.name, "sha256": file_digest(exported),
               "raw_frames": [{"path": str(path.relative_to(tmp_path)),
                               "bytes": path.stat().st_size, "sha256": file_digest(path),
                               "metadata_sha256": file_digest(path.with_suffix(".json"))}]}
        report = {"status": "rank_verified", "rank": rank, "start": 0, "stop": 1,
                  "plan_sha256": file_digest(tmp_path / "plan.json"), "source_sha256": "code",
                  "rows": [row]}
        (tmp_path / f"export-0000-0001-r{rank}.json").write_text(json.dumps(report))
    pair_chunk(tmp_path, 0, 1)
    return raw_paths


def test_release_keeps_accepted_payloads_and_raw_metadata(tmp_path: Path) -> None:
    paths = prepared(tmp_path)
    result = release_staging(tmp_path, 0, 0, 1)
    assert result["payloads"] == 1 and result["accepted_exports_preserved"]
    assert not paths[0].exists() and paths[1].exists()
    assert paths[0].with_suffix(".json").exists()
    assert (tmp_path / "accepted/train-one.safetensors").read_bytes() == b"accepted"


@pytest.mark.parametrize("fault", ["export", "raw", "metadata", "receipt", "plan"])
def test_changed_evidence_preserves_every_raw_payload(tmp_path: Path, fault: str) -> None:
    raw_paths = prepared(tmp_path)
    path = {
        "export": tmp_path / "accepted/train-one.safetensors", "raw": raw_paths[0],
        "metadata": raw_paths[0].with_suffix(".json"),
        "receipt": tmp_path / "export-0000-0001-r1.json", "plan": tmp_path / "plan.json",
    }[fault]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        release_staging(tmp_path, 0, 0, 1)
    assert all(path.exists() for path in raw_paths)
