"""Independent byte-exact conformance and artifact integrity for the reused row codec."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from mxwave.lossless_bf16_rows import pack_rows, read_from, write_to


def test_all_bf16_patterns_roundtrip_and_chunking() -> None:
    words = np.arange(65536, dtype="<u2").reshape(128, 512)
    packed = pack_rows(words, chunk_rows=17)
    assert np.array_equal(packed.row_words(np.arange(128)), words)
    assert np.array_equal(packed.row_words(np.array([127, 0, 127, 64])), words[[127, 0, 127, 64]])
    assert np.array_equal(np.concatenate([chunk.row_words(np.arange(chunk.rows))
        for chunk in packed.iter_chunks(19)]), words)


def test_hash_bound_artifact_and_every_serialized_byte(tmp_path: Path) -> None:
    words = np.random.default_rng(91).integers(0, 65536, size=(3, 512), dtype=np.uint16)
    digest = hashlib.sha256(words.tobytes()).hexdigest()
    packed = pack_rows(words, source_sha256=digest)
    path = tmp_path / "rows.mxb"
    write_to(packed, path)
    assert path.stat().st_size == packed.retained_bytes
    restored = read_from(path, verify_hashes=True)
    assert restored.source_sha256 == digest
    assert np.array_equal(restored.row_words(np.arange(3)), words)
    with path.open("r+b") as handle:
        handle.seek(-1, 2)
        byte = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([byte[0] ^ 1]))
    with pytest.raises(ValueError):
        read_from(path, verify_hashes=True)


def test_invalid_ids_and_storage_contract_fail() -> None:
    words = np.zeros((2, 512), dtype="<u2")
    packed = pack_rows(words)
    for ids in (np.array([-1]), np.array([2])):
        with pytest.raises(ValueError):
            packed.row_words(ids)
    with pytest.raises(ValueError):
        pack_rows(words[:, ::2])
