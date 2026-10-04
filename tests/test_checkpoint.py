"""Regression tests for checkpoint concurrency and durable metadata."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mxwave import checkpoint


def test_output_lock_is_exclusive_across_processes_and_released_after_exception(
    tmp_path: Path,
) -> None:
    output = tmp_path / "out"
    command = [
        sys.executable,
        "-c",
        (
            "from pathlib import Path; from mxwave.checkpoint import locked_output_directory; "
            "import sys\n"
            "try:\n"
            " with locked_output_directory(Path(sys.argv[1]), source_dir=Path(sys.argv[2])): pass\n"
            "except FileExistsError: sys.exit(7)\n"
        ),
        str(output),
        str(tmp_path / "source"),
    ]
    with (
        pytest.raises(RuntimeError, match="interrupted"),
        checkpoint.locked_output_directory(output, source_dir=tmp_path / "source"),
    ):
        result = subprocess.run(command, timeout=15, capture_output=True, check=False)
        assert result.returncode == 7, result.stderr.decode()
        raise RuntimeError("interrupted")
    with checkpoint.locked_output_directory(output, source_dir=tmp_path / "source"):
        assert not checkpoint._directory_has_artifacts(output)


def test_metadata_is_synced_before_and_after_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "marker.json"
    events = []
    actual_fsync = os.fsync
    actual_directory_sync = checkpoint._fsync_directory

    def sync_file(descriptor: int) -> None:
        events.append(("fsync", destination.exists()))
        actual_fsync(descriptor)

    def sync_directory(directory: Path) -> None:
        events.append(("directory", destination.exists()))
        actual_directory_sync(directory)

    monkeypatch.setattr(checkpoint.os, "fsync", sync_file)
    monkeypatch.setattr(checkpoint, "_fsync_directory", sync_directory)
    checkpoint._atomic_write_text(destination, '{"complete": true}\n')
    assert events == [("fsync", False), ("directory", True), ("fsync", True)]
    assert not list(tmp_path.glob("*.incomplete"))


@pytest.mark.parametrize("phase", ["before-rename", "after-rename", "after-ledger"])
def test_resume_recovers_each_shard_transaction_crash_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    name = "model.safetensors"
    temporary = tmp_path / f".{name}.123.incomplete"
    final = tmp_path / name
    temporary.write_bytes(b"verified, synced payload")
    identity = "a" * 64
    ledger, records = checkpoint._prepare_integrity_records(
        tmp_path, run_identity_sha256=identity, expected_filenames={name}, resume=False
    )
    expected = checkpoint._file_integrity(temporary)
    replace = Path.replace
    write_ledger = checkpoint._atomic_write_integrity_ledger
    unlink = Path.unlink

    def interrupted_replace(path: Path, target: Path) -> Path:
        if phase == "before-rename" and target == final:
            raise RuntimeError("interrupted transaction")
        return replace(path, target)

    def interrupted_ledger(path: Path, **kwargs: object) -> None:
        if phase == "after-rename":
            raise RuntimeError("interrupted transaction")
        write_ledger(path, **kwargs)

    def interrupted_unlink(path: Path, missing_ok: bool = False) -> None:
        if phase == "after-ledger" and path.name == checkpoint._PENDING_SHARD_FILENAME:
            raise RuntimeError("interrupted transaction")
        unlink(path, missing_ok=missing_ok)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", interrupted_replace)
        patch.setattr(Path, "unlink", interrupted_unlink)
        patch.setattr(checkpoint, "_atomic_write_integrity_ledger", interrupted_ledger)
        with pytest.raises(RuntimeError, match="interrupted transaction"):
            checkpoint._commit_shard(
                temporary,
                final,
                integrity_path=ledger,
                run_identity_sha256=identity,
                records=records,
            )
    assert (tmp_path / checkpoint._PENDING_SHARD_FILENAME).exists()
    checkpoint._cleanup_stale_shard_temporaries(tmp_path, {name})
    _, recovered = checkpoint._prepare_integrity_records(
        tmp_path, run_identity_sha256=identity, expected_filenames={name}, resume=True
    )
    assert not (tmp_path / checkpoint._PENDING_SHARD_FILENAME).exists()
    if phase == "before-rename":
        assert recovered == {}
        assert not final.exists()
    else:
        assert recovered == {name: expected}
        assert checkpoint._file_integrity(final) == expected


@pytest.mark.parametrize("mutation", ["payload", "identity", "filename"])
def test_pending_shard_recovery_rejects_untrusted_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    name = "model.safetensors"
    final = tmp_path / name
    temporary = tmp_path / f".{name}.123.incomplete"
    temporary.write_bytes(b"verified, synced payload")
    identity = "a" * 64
    ledger, records = checkpoint._prepare_integrity_records(
        tmp_path, run_identity_sha256=identity, expected_filenames={name}, resume=False
    )

    def interrupted(*args: object, **kwargs: object) -> None:
        raise RuntimeError("crash after rename")

    with monkeypatch.context() as patch:
        patch.setattr(checkpoint, "_atomic_write_integrity_ledger", interrupted)
        with pytest.raises(RuntimeError, match="crash after rename"):
            checkpoint._commit_shard(
                temporary,
                final,
                integrity_path=ledger,
                run_identity_sha256=identity,
                records=records,
            )
    pending = tmp_path / checkpoint._PENDING_SHARD_FILENAME
    if mutation == "payload":
        final.write_bytes(b"corrupted,synced payload")
    else:
        document = json.loads(pending.read_text())
        if mutation == "identity":
            document["run_identity_sha256"] = "b" * 64
        else:
            document["filename"] = "../outside.safetensors"
        pending.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="Cannot safely resume"):
        checkpoint._prepare_integrity_records(
            tmp_path, run_identity_sha256=identity, expected_filenames={name}, resume=True
        )
    assert pending.exists()
