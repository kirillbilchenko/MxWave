from __future__ import annotations

import gc
import weakref
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from integrations.glm53_mxfp4.drafter_feature_store import VerifiedFeatureStore
from integrations.glm53_mxfp4.drafter_fp8_roundtrip import digest


def write_sample(path: Path, offset: int = 0) -> dict[str, torch.Tensor]:
    values = {"input_ids": torch.arange(12) + offset,
              "target_features": (torch.arange(44).reshape(11, 4) + offset).to(torch.bfloat16)}
    save_file(values, path)
    return values


def test_exact_payload_and_single_sample_residency(tmp_path: Path) -> None:
    first = write_sample(tmp_path / "a.safetensors")
    second = write_sample(tmp_path / "b.safetensors", 1)
    a_sha, b_sha = (digest(tmp_path / n) for n in ("a.safetensors", "b.safetensors"))
    store = VerifiedFeatureStore(tmp_path)
    assert torch.equal(store.input_ids("a.safetensors", a_sha), first["input_ids"])
    loaded = store.get("a.safetensors", a_sha)
    assert all(torch.equal(loaded[k], v) for k, v in first.items())
    assert loaded is store.get("a.safetensors", a_sha)
    reference = weakref.ref(loaded["target_features"])
    del loaded
    other = store.get("b.safetensors", b_sha)
    gc.collect()
    assert reference() is None
    assert all(torch.equal(other[k], v) for k, v in second.items())
    assert all(torch.equal(store.get("a.safetensors", a_sha)[k], v) for k, v in first.items())


def test_replaced_cached_file_and_wrong_hash_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "a.safetensors"
    write_sample(path)
    sha = digest(path)
    store = VerifiedFeatureStore(tmp_path)
    store.get(path.name, sha)
    replacement = tmp_path / "replacement.safetensors"
    write_sample(replacement, 2)
    replacement.replace(path)
    with pytest.raises(ValueError, match="checksum"):
        store.get(path.name, sha)
    with pytest.raises(ValueError, match="checksum"):
        store.input_ids(path.name, sha)


@pytest.mark.parametrize("relative", ["../outside.safetensors", "/tmp/outside.safetensors"])
def test_dataset_boundary_is_enforced(tmp_path: Path, relative: str) -> None:
    with pytest.raises(ValueError, match="inside"):
        VerifiedFeatureStore(tmp_path).get(relative, "unused")


def test_eviction_does_not_break_live_autograd_inputs(tmp_path: Path) -> None:
    write_sample(tmp_path / "a.safetensors")
    write_sample(tmp_path / "b.safetensors", 1)
    store = VerifiedFeatureStore(tmp_path)
    values = store.get("a.safetensors", digest(tmp_path / "a.safetensors"))
    master = torch.ones(4, 2, requires_grad=True)
    x = values["target_features"].float()
    expected = x.T @ torch.ones(11, 2)
    loss = (x @ master).sum()
    del values, x
    store.get("b.safetensors", digest(tmp_path / "b.safetensors"))
    loss.backward()
    assert torch.equal(master.grad, expected)
