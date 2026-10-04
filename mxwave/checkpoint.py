"""Shared bounded quantization, locking, and durable checkpoint bookkeeping."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

DEFAULT_TENSOR_CHUNK_MAX_ELEMENTS = 1_048_576
OUTPUT_LOCK_FILENAME = ".mxwave-output.lock"
_INTEGRITY_SIDECAR_FILENAME = "mxwave-shard-integrity.json"
_INTEGRITY_SCHEMA_VERSION = 1
_PENDING_SHARD_FILENAME = "mxwave-shard-pending.json"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


def _directory_has_artifacts(path: Path) -> bool:
    return any(entry.name != OUTPUT_LOCK_FILENAME for entry in path.iterdir())


def _validate_source_output(model_dir: Path, output_dir: Path) -> None:
    source = model_dir.resolve()
    output = output_dir.resolve()
    if source == output or output.is_relative_to(source):
        raise ValueError("output_dir must not equal or be nested inside model_dir")


@contextmanager
def locked_output_directory(output_dir: Path, *, source_dir: Path) -> Iterator[None]:
    """Hold an exclusive writer lock until conversion finishes or raises.

    The lock file remains in place so another process cannot lock a different
    inode during release. The operating system releases the lock after a crash.
    """
    _validate_source_output(source_dir, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(output_dir / OUTPUT_LOCK_FILENAME, flags, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise FileExistsError(f"Another writer holds the output lock: {output_dir}") from exc
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"{os.getpid()}\n".encode())
        yield
    finally:
        os.close(descriptor)


def _atomic_write_text(path: Path, value: str) -> None:
    """Replace a text artifact after syncing its bytes and directory entry."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.incomplete")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_save_shard(tensors: dict[str, torch.Tensor], path: Path) -> None:
    """Replace a materialized safetensors shard after syncing it to disk."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.incomplete")
    try:
        save_file(tensors, str(temporary))
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        temporary.replace(path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def tensor_chunk_rows(columns: int, max_rows: int, max_elements: int) -> int:
    """Bound rows by both the configured row limit and the matrix width."""
    if columns <= 0 or max_rows <= 0 or max_elements <= 0:
        raise ValueError("Chunk dimensions and limits must be positive")
    if columns > max_elements:
        raise ValueError(
            f"One row has {columns} elements, above tensor_chunk_max_elements={max_elements}"
        )
    return min(max_rows, max_elements // columns)


def iter_quantized_row_chunks(
    read_rows: Callable[[int, int], torch.Tensor],
    quantize: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    *,
    rows: int,
    columns: int,
    max_rows: int,
    max_elements: int,
    name: str,
) -> Iterator[tuple[int, int, torch.Tensor, torch.Tensor]]:
    """Read, validate, and quantize one element-bounded row slice at a time."""
    chunk_rows = tensor_chunk_rows(columns, max_rows, max_elements)
    for start in range(0, rows, chunk_rows):
        stop = min(start + chunk_rows, rows)
        weight = read_rows(start, stop)
        if tuple(weight.shape) != (stop - start, columns):
            raise ValueError(f"Row slice has an unexpected shape for {name!r}")
        if not torch.isfinite(weight).all():
            raise ValueError(f"Tensor contains NaN or infinity in rows [{start}, {stop}): {name}")
        packed, scales = quantize(weight)
        del weight
        for tensor, width in ((packed, columns // 2), (scales, columns // 32)):
            if tuple(tensor.shape) != (stop - start, width) or tensor.dtype != torch.uint8:
                raise ValueError(f"Quantized row slice has an unexpected shape or dtype: {name}")
        del tensor
        yield start, stop, packed, scales
        del packed, scales


@dataclass(frozen=True)
class _ShardIntegrityRecord:
    """Cryptographic identity of one complete output shard file."""

    bytes: int
    sha256: str

    def as_json(self) -> dict[str, int | str]:
        """Return the canonical JSON representation persisted in the ledger."""
        return {"bytes": self.bytes, "sha256": self.sha256}


def _fsync_directory(path: Path) -> None:
    """Persist a rename or unlink in one directory before returning."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _stale_shard_temporary_names(expected_filenames: set[str]) -> tuple[re.Pattern[str], ...]:
    """Return exact patterns for temporary files created by the shard writer."""
    return tuple(
        re.compile(rf"\.{re.escape(name)}\.[0-9]+\.incomplete\Z")
        for name in sorted(expected_filenames)
    )


def _cleanup_stale_shard_temporaries(
    output_dir: Path,
    expected_filenames: set[str],
) -> None:
    """Remove only stale temporary files for shards in the current model plan."""
    if not output_dir.is_dir():
        return
    patterns = _stale_shard_temporary_names(expected_filenames)
    removed = False
    for candidate in output_dir.iterdir():
        if not any(pattern.fullmatch(candidate.name) for pattern in patterns):
            continue
        if candidate.is_dir() and not candidate.is_symlink():
            raise ValueError(f"Refusing to remove stale shard temporary directory: {candidate}")
        candidate.unlink()
        removed = True
    if removed:
        _fsync_directory(output_dir)


def _validate_output_path(
    model_dir: Path,
    output_dir: Path,
    resume: bool,
    expected_filenames: set[str],
) -> None:
    _validate_source_output(model_dir, output_dir)
    _cleanup_stale_shard_temporaries(output_dir, expected_filenames)
    if output_dir.exists() and _directory_has_artifacts(output_dir) and not resume:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}; use --resume for a partial run"
        )


def _canonical_sha256(value: object) -> str:
    """Hash one JSON-compatible value using a stable canonical encoding."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _file_integrity(path: Path) -> _ShardIntegrityRecord:
    """Hash one shard in bounded chunks and return its exact file size."""
    before = path.stat()
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    after = path.stat()
    if size != after.st_size or (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise OSError(
            f"Shard {path.name} changed while its integrity hash was computed: "
            f"read={size}, stat={after.st_size}"
        )
    return _ShardIntegrityRecord(bytes=size, sha256=digest.hexdigest())


def _source_checkpoint_identity(
    model_dir: Path,
    shard_paths: list[Path],
    *,
    repository: str | None,
    revision: str | None,
) -> dict[str, Any]:
    """Bind a conversion to source contents independently of file timestamps."""
    identity: dict[str, Any] = {
        "repository": repository,
        "revision": revision,
        "config_sha256": (
            _file_integrity(model_dir / "config.json").sha256
            if (model_dir / "config.json").is_file()
            else None
        ),
        "shards": {
            path.name: _file_integrity(path).as_json()
            for path in sorted(shard_paths, key=lambda value: value.name)
        },
    }
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        identity["weight_index_sha256"] = _file_integrity(index_path).sha256
    return identity


def _commit_shard(
    temporary: Path,
    destination: Path,
    *,
    integrity_path: Path,
    run_identity_sha256: str,
    records: dict[str, _ShardIntegrityRecord],
) -> None:
    """Journal a verified, synced shard before renaming and recording it.

    The durable intent allows resume to verify and adopt the complete file if
    the process crashes between its rename and the integrity-ledger write.
    Callers must hold the output lock and verify the temporary shard first.
    """
    pending_path = destination.parent / _PENDING_SHARD_FILENAME
    if pending_path.exists():
        raise FileExistsError("An unfinished shard transaction requires --resume")
    if destination.exists():
        raise FileExistsError(f"Refusing to replace existing shard: {destination}")
    record = _file_integrity(temporary)
    document = {
        "schema_version": 1,
        "run_identity_sha256": run_identity_sha256,
        "filename": destination.name,
        "integrity": record.as_json(),
    }
    _atomic_write_text(pending_path, json.dumps(document, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)
    _fsync_directory(destination.parent)
    records[destination.name] = record
    _atomic_write_integrity_ledger(
        integrity_path, run_identity_sha256=run_identity_sha256, records=records
    )
    pending_path.unlink()
    _fsync_directory(destination.parent)


def _recover_pending_shard(
    output_dir: Path,
    *,
    integrity_path: Path,
    run_identity_sha256: str,
    expected_filenames: set[str],
    records: dict[str, _ShardIntegrityRecord],
) -> None:
    """Finish a run-bound, hash-verified transaction or discard an unrenamed intent."""
    pending_path = output_dir / _PENDING_SHARD_FILENAME
    if not pending_path.exists():
        return
    try:
        raw = json.loads(pending_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError("Cannot safely resume: invalid pending shard transaction JSON") from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("Cannot safely resume: invalid pending shard transaction schema")
    if raw.get("run_identity_sha256") != run_identity_sha256:
        raise ValueError("Cannot safely resume: pending shard transaction has a different identity")
    name = raw.get("filename")
    if not isinstance(name, str) or name not in expected_filenames:
        raise ValueError("Cannot safely resume: unexpected pending shard filename")
    record = _parse_integrity_record(name, raw.get("integrity"))
    if name in records and records[name] != record:
        raise ValueError("Cannot safely resume: pending shard disagrees with integrity ledger")
    path = output_dir / name
    if path.exists():
        _validate_resumed_shard_integrity(path, {name: record})
        records[name] = record
        _atomic_write_integrity_ledger(
            integrity_path, run_identity_sha256=run_identity_sha256, records=records
        )
    # No renamed file means the interrupted shard will be regenerated. Its
    # stale temporary payload has already been removed under the output lock.
    pending_path.unlink()
    _fsync_directory(output_dir)


def _atomic_write_integrity_ledger(
    path: Path,
    *,
    run_identity_sha256: str,
    records: Mapping[str, _ShardIntegrityRecord],
) -> None:
    """Atomically and durably persist the complete per-shard integrity ledger."""
    document = {
        "schema_version": _INTEGRITY_SCHEMA_VERSION,
        "algorithm": "sha256",
        "run_identity_sha256": run_identity_sha256,
        "role": "resume-only metadata; final evidence is copied into mxwave-manifest.json",
        "shards": {name: record.as_json() for name, record in sorted(records.items())},
    }
    _atomic_write_text(path, json.dumps(document, indent=2, sort_keys=True) + "\n")


def _parse_integrity_records(
    path: Path,
    *,
    run_identity_sha256: str,
    expected_filenames: set[str],
) -> dict[str, _ShardIntegrityRecord]:
    """Load and strictly validate one run-bound shard-integrity ledger."""
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError("Cannot safely resume: invalid shard-integrity ledger JSON") from exc
    if not isinstance(raw, dict):
        raise TypeError("Cannot safely resume: shard-integrity ledger must be an object")
    if raw.get("schema_version") != _INTEGRITY_SCHEMA_VERSION:
        raise ValueError("Cannot safely resume: unsupported shard-integrity ledger schema")
    if raw.get("algorithm") != "sha256":
        raise ValueError("Cannot safely resume: shard-integrity algorithm changed")
    if raw.get("run_identity_sha256") != run_identity_sha256:
        raise ValueError("Cannot safely resume: shard-integrity ledger does not match run identity")
    raw_records = raw.get("shards")
    if not isinstance(raw_records, dict):
        raise TypeError("Cannot safely resume: shard-integrity ledger has no shards object")

    records: dict[str, _ShardIntegrityRecord] = {}
    for name, raw_record in raw_records.items():
        if not isinstance(name, str) or name not in expected_filenames:
            raise ValueError(f"Cannot safely resume: unexpected integrity shard {name!r}")
        records[name] = _parse_integrity_record(name, raw_record)
    return records


def _parse_integrity_record(name: str, raw: object) -> _ShardIntegrityRecord:
    if not isinstance(raw, dict) or set(raw) != {"bytes", "sha256"}:
        raise TypeError(f"Cannot safely resume: invalid integrity record for {name!r}")
    byte_count = raw.get("bytes")
    digest = raw.get("sha256")
    if (
        not isinstance(byte_count, int)
        or isinstance(byte_count, bool)
        or byte_count <= 0
        or not isinstance(digest, str)
        or _SHA256_PATTERN.fullmatch(digest) is None
    ):
        raise ValueError(f"Cannot safely resume: invalid size or SHA-256 for {name!r}")
    return _ShardIntegrityRecord(bytes=byte_count, sha256=digest)


def _prepare_integrity_records(
    output_dir: Path,
    *,
    run_identity_sha256: str,
    expected_filenames: set[str],
    resume: bool,
) -> tuple[Path, dict[str, _ShardIntegrityRecord]]:
    """Create or load the run-bound ledger without trusting existing shards."""
    ledger_path = output_dir / _INTEGRITY_SIDECAR_FILENAME
    existing_shards = {
        path.name for path in output_dir.glob("*.safetensors") if path.name in expected_filenames
    }
    if resume:
        if ledger_path.is_file():
            records = _parse_integrity_records(
                ledger_path,
                run_identity_sha256=run_identity_sha256,
                expected_filenames=expected_filenames,
            )
            _recover_pending_shard(
                output_dir,
                integrity_path=ledger_path,
                run_identity_sha256=run_identity_sha256,
                expected_filenames=expected_filenames,
                records=records,
            )
            orphan_shards = sorted(existing_shards.difference(records))
            if orphan_shards:
                raise ValueError(
                    "Cannot safely resume: existing shard has no integrity record; this is an "
                    f"untrusted crash window: {orphan_shards[:3]}"
                )
            return ledger_path, records
        if existing_shards:
            raise ValueError(
                "Cannot safely resume: shard-integrity ledger is missing while output shards "
                f"exist ({sorted(existing_shards)[:3]}). This is an untrusted crash window."
            )
    records = {}
    _atomic_write_integrity_ledger(
        ledger_path,
        run_identity_sha256=run_identity_sha256,
        records=records,
    )
    return ledger_path, records


def _validate_resumed_shard_integrity(
    path: Path,
    records: Mapping[str, _ShardIntegrityRecord],
) -> None:
    """Require a matching recorded size and full-file SHA-256 before reuse."""
    recorded = records.get(path.name)
    if recorded is None:
        raise ValueError(
            f"Cannot safely resume: existing shard {path.name!r} has no integrity record. "
            "This is an untrusted crash window."
        )
    actual_size = path.stat().st_size
    if actual_size != recorded.bytes:
        raise ValueError(
            f"Cannot safely resume: shard {path.name!r} size mismatch "
            f"(recorded={recorded.bytes}, actual={actual_size})"
        )
    actual = _file_integrity(path)
    if actual.sha256 != recorded.sha256:
        raise ValueError(f"Cannot safely resume: shard {path.name!r} SHA-256 mismatch")


def _prepare_run_marker(output_dir: Path, identity: dict[str, Any], resume: bool) -> None:
    """Create or validate the run marker before any output shard is reused."""
    marker_path = output_dir / "mxwave-run.json"
    if resume and _directory_has_artifacts(output_dir):
        if not marker_path.is_file():
            raise ValueError(
                "Cannot safely resume: mxwave-run.json is missing from the non-empty output"
            )
        try:
            recorded = json.loads(marker_path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError("Cannot safely resume: mxwave-run.json is invalid") from exc
        if recorded != identity:
            raise ValueError(
                "Cannot safely resume: quantization settings, calibration, targets, or source changed"
            )
        return

    _atomic_write_text(marker_path, json.dumps(identity, indent=2, sort_keys=True) + "\n")
