"""The streaming loader must verify bytes before use and bound temporary disk storage."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from integrations.glm53_mxfp4 import checkpoint_stream as stream


def fixture(tmp_path: Path) -> tuple[dict, dict[str, bytes]]:
    """Build two independently hashed shards with distinct named tensors."""
    records, data = [], {}
    for i in range(2):
        name = f"model-{i + 1:05d}-of-00002.safetensors"
        path = tmp_path / name
        save_file({f"tensor_{i}": torch.arange(12, dtype=torch.int16) + i}, str(path))
        raw = path.read_bytes()
        data[name] = raw
        records.append({"name": name, "bytes": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest()})
    return {"shards": records, "source_shards": 2,
            "total_safetensors_file_bytes": sum(x["bytes"] for x in records)}, data


def test_local_tensors_exact_and_sources_preserved(tmp_path: Path) -> None:
    inventory, original = fixture(tmp_path)
    result = dict(stream.iter_checkpoint(inventory, tmp_path, tmp_path / "scratch"))
    for i in range(2):
        assert torch.equal(result[f"tensor_{i}"], torch.arange(12, dtype=torch.int16) + i)
    assert all((tmp_path / name).read_bytes() == raw for name, raw in original.items())
    assert not list((tmp_path / "scratch").iterdir())


def test_remote_has_only_one_temporary_shard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inventory, data = fixture(tmp_path)
    seen, progress = [], []

    def fetch(request, timeout):
        name = request.full_url.rsplit("/", 1)[-1]
        seen.append(name)
        return io.BytesIO(data[name])

    monkeypatch.setattr(stream.urllib.request, "urlopen", fetch)
    scratch = tmp_path / "scratch"
    iterator = stream.iter_checkpoint(inventory, tmp_path / "absent", scratch,
                                      remote_base="http://fixture.invalid", progress=progress.append)
    for i in range(2):
        name, tensor = next(iterator)
        assert name == f"tensor_{i}"
        assert torch.equal(tensor, torch.arange(12, dtype=torch.int16) + i)
        assert len(list(scratch.rglob("*.safetensors"))) == 1
    with pytest.raises(StopIteration):
        next(iterator)
    assert len(seen) == len(progress) == 2
    assert not list(scratch.iterdir())


@pytest.mark.parametrize("corruption", ["hash", "truncated", "oversized"])
def test_corruption_rejected_before_first_tensor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corruption: str,
) -> None:
    inventory, data = fixture(tmp_path)
    raw = next(iter(data.values()))
    bad = raw[:-1] if corruption == "truncated" else (
        raw + b"extra" if corruption == "oversized" else raw[:-1] + bytes([raw[-1] ^ 1]))
    monkeypatch.setattr(stream.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(bad))
    scratch = tmp_path / "scratch"
    with pytest.raises(ValueError):
        next(stream.iter_checkpoint(inventory, tmp_path / "absent", scratch,
                                    remote_base="http://fixture.invalid"))
    assert not list(scratch.iterdir())


def test_unsafe_and_duplicate_names_rejected(tmp_path: Path) -> None:
    inventory, _ = fixture(tmp_path)
    inventory["shards"][0]["name"] = "../model-00001-of-00002.safetensors"
    with pytest.raises(ValueError, match="Invalid"):
        stream.checked_shards(inventory)
    inventory["shards"][0]["name"] = inventory["shards"][1]["name"]
    with pytest.raises(ValueError, match="duplicate"):
        stream.checked_shards(inventory)
