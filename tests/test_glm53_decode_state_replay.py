"""Exact storage restoration, aliasing and non-selection contracts for decode replay."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

import pytest
import torch


@pytest.fixture(params=["gpu_images", "disk_images"])
def replay(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest,
           tmp_path: Path) -> ModuleType:
    """Load the device-agnostic snapshot helper without importing a vLLM engine."""
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))
    module = importlib.import_module("decode_state_replay")
    if request.param == "disk_images":
        disk = importlib.import_module("decode_state_disk")
        owned = []

        def snapshots(tensors: list[torch.Tensor], maximum_bytes: int) -> object:
            result = disk.DiskByteSnapshots(tensors, maximum_bytes, tmp_path)
            owned.append(result)
            return result

        monkeypatch.setattr(module, "ByteSnapshots", snapshots)

        def cleanup() -> None:
            for result in owned:
                result.close()
            assert not list(tmp_path.iterdir())

        request.addfinalizer(cleanup)
    return module


def test_aliases_strides_and_nan_payloads_restore_exact_bytes(replay: ModuleType) -> None:
    """Snapshot the underlying bytes once, including padding outside a selected view."""
    bits = torch.tensor([0x7FC1, 0x8000, 0x3F80, 0x7FC2, 0, 1, 2, 3], dtype=torch.uint16)
    value = bits.view(torch.bfloat16)
    original = bits.clone()
    snapshot = replay.ByteSnapshots([value[1::2], value[2:]], maximum_bytes=bits.numel() * 2)
    assert len(snapshot.images) == 1 and snapshot.payload_bytes == 16
    snapshot.capture_before()
    bits.fill_(123)
    snapshot.restore_before()
    assert snapshot.before_exact() and torch.equal(bits, original)


def test_final_served_state_advances_once_not_five_times(replay: ModuleType) -> None:
    """Every replay starts from identical bytes, while the final real result is retained."""
    cache = torch.tensor([10], dtype=torch.int32)
    snapshot = replay.ByteSnapshots([cache], maximum_bytes=4)
    calls = []

    def call() -> torch.Tensor:
        calls.append(int(cache[0]))
        cache.add_(1)
        return cache.clone()

    output, rows = replay.replay_identical_state(call, snapshot, 5, [("cache", cache)])
    assert calls == [10] * 5 and int(output[0]) == int(cache[0]) == 11
    assert len(rows) == 5 and all(r["initial_bytes_exact"] for r in rows)
    assert all(r["outputs"] == rows[0]["outputs"] for r in rows)
    assert all(r["changed_storage_count"] == 0 for r in rows)


def test_output_and_cache_nondeterminism_are_retained(replay: ModuleType) -> None:
    """Keep all unfavorable outputs and identify the changed logical cache view."""
    cache = torch.tensor([0, 0], dtype=torch.int32)
    snapshot = replay.ByteSnapshots([cache], maximum_bytes=8)
    counter = 0

    def call() -> torch.Tensor:
        nonlocal counter
        counter += 1
        cache[1] = counter
        return cache.clone()

    output, rows = replay.replay_identical_state(call, snapshot, 5, [("cache", cache[1:])])
    assert int(output[1]) == 5 and int(cache[1]) == 5
    assert len({r["outputs"][0]["sha256"] for r in rows}) == 5
    assert all(r["changed_cache_views"] == ["cache"] for r in rows[1:])


def test_reference_view_keeps_storage_offset_and_strides(replay: ModuleType) -> None:
    """Reconstruct a typed view from the raw reference allocation without numeric conversion."""
    base = torch.arange(24, dtype=torch.int16).reshape(4, 6)
    view = base[1:, 1::2].transpose(0, 1)
    snapshot = replay.ByteSnapshots([view], maximum_bytes=48)
    snapshot.capture_reference_after()
    reference = snapshot.reference_view(view)
    assert reference.stride() == view.stride() and reference.storage_offset() == view.storage_offset()
    assert replay.logical_bytes_equal(view, reference)
    base[2, 3] = -1
    assert not replay.logical_bytes_equal(view, reference)


def test_exception_restores_initial_state(replay: ModuleType) -> None:
    """An aborted probe leaves no partially advanced cache behind."""
    cache = torch.tensor([7], dtype=torch.int32)
    snapshot = replay.ByteSnapshots([cache], maximum_bytes=4)

    def call() -> torch.Tensor:
        cache.add_(10)
        raise RuntimeError("fixture failure")

    with pytest.raises(RuntimeError, match="fixture failure"):
        replay.replay_identical_state(call, snapshot, 5, [("cache", cache)])
    assert int(cache[0]) == 7


def test_limits_fail_before_executing_a_graph(replay: ModuleType) -> None:
    """Reject an oversized/empty snapshot and any changed fixed replay count."""
    with pytest.raises(ValueError, match="byte budget"):
        replay.ByteSnapshots([torch.zeros(8)], maximum_bytes=4)
    with pytest.raises(ValueError, match="Empty"):
        replay.ByteSnapshots([], maximum_bytes=4)
    snapshot = replay.ByteSnapshots([torch.zeros(1)], maximum_bytes=4)
    with pytest.raises(ValueError, match="five"):
        replay.replay_identical_state(lambda: pytest.fail("must not run"), snapshot, 4, [])


def test_byte_comparison_preserves_nan_and_signed_zero(replay: ModuleType) -> None:
    """Ordinary float equality would mishandle both NaNs and negative-zero distinctions."""
    bits = torch.tensor([0x7FC1, 0x8000], dtype=torch.uint16)
    left, right = bits.view(torch.bfloat16), bits.clone().view(torch.bfloat16)
    assert replay.logical_bytes_equal(left, right)
    right.view(torch.uint16)[1] = 0
    assert not replay.logical_bytes_equal(left, right)


def test_disk_images_cover_chunk_boundary_and_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Exercise multi-chunk I/O and resource transfer beyond the GPU-memory failure mode."""
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))
    disk = importlib.import_module("decode_state_disk")
    base = torch.arange(disk.CHUNK_BYTES + 19, dtype=torch.uint8)
    extra = torch.tensor([10], dtype=torch.int32)
    snapshot = disk.DiskByteSnapshots([base], base.numel(), tmp_path)
    additional = disk.DiskByteSnapshots([extra], 4, tmp_path)
    snapshot.absorb(additional)
    additional.close()
    snapshot.capture_before()
    snapshot.capture_reference_after()
    base[disk.CHUNK_BYTES:] = 42
    extra.add_(2)
    assert len(snapshot.changed_storages()) == 2
    snapshot.restore_before()
    assert snapshot.before_exact() and int(extra[0]) == 10
    assert int(base[-1]) == (base.numel() - 1) % 256
    snapshot.close()
    assert not list(tmp_path.iterdir())
