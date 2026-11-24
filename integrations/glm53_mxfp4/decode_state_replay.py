"""Bounded byte snapshots for an intrusive same-input, same-state decode diagnostic."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import torch


def tensor_values(value: Any) -> list[torch.Tensor]:
    """Flatten only tensor/list/tuple payloads, without traversing model parameters."""
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        return [item for child in value for item in tensor_values(child)]
    return []


def tensor_digest(value: torch.Tensor) -> dict[str, Any]:
    """Fingerprint a bounded logical tensor as bytes, preserving BF16/FP8/NaN payloads."""
    raw = value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
    return {"shape": list(value.shape), "dtype": str(value.dtype), "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()}


@dataclass
class StorageImage:
    """Retain exact before/reference-after bytes for one unique underlying allocation."""

    key: tuple[int, int]
    live: torch.Tensor
    before: torch.Tensor
    reference_after: torch.Tensor


class ByteSnapshots:
    """Snapshot aliased and strided tensors once per storage, including padding bytes."""

    def __init__(self, tensors: Iterable[torch.Tensor], maximum_bytes: int) -> None:
        """Reserve two copies with a fail-closed byte budget before running any replay."""
        unique: dict[tuple[int, int], torch.Tensor] = {}
        device: torch.device | None = None
        for tensor in tensors:
            if device is not None and tensor.device != device:
                raise ValueError("All snapshot tensors must use one device")
            device = tensor.device
            storage = tensor.untyped_storage()
            key = (storage.data_ptr(), storage.nbytes())
            if key[1] and key not in unique:
                unique[key] = torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(
                    storage, 0, (storage.nbytes(),), (1,))
        self.payload_bytes = sum(key[1] for key in unique)
        if not unique or self.payload_bytes > maximum_bytes:
            raise ValueError("Empty snapshot or storage byte budget exceeded")
        if device is not None and device.type == "cuda":
            free, _ = torch.cuda.mem_get_info(device)
            if free < 2 * self.payload_bytes + 512 * 1024**2:
                raise MemoryError("Insufficient free memory for both bounded snapshot copies")
        self.images = {key: StorageImage(key, live, torch.empty_like(live), torch.empty_like(live))
                       for key, live in unique.items()}

    def capture_before(self) -> None:
        """Copy the current complete byte state into its preallocated before image."""
        for image in self.images.values():
            image.before.copy_(image.live)

    def restore_before(self) -> None:
        """Restore initial bytes without dtype conversion, including aliased padding."""
        for image in self.images.values():
            image.live.copy_(image.before)

    def before_exact(self) -> bool:
        """Check restoration using byte equality, including NaNs and signed zero."""
        return all(torch.equal(image.live, image.before) for image in self.images.values())

    def capture_reference_after(self) -> None:
        """Retain the first replay's complete resulting byte state."""
        for image in self.images.values():
            image.reference_after.copy_(image.live)

    def changed_storages(self) -> list[tuple[int, int]]:
        """Identify differing allocations without materializing a full boolean copy."""
        return [key for key, image in self.images.items()
                if not torch.equal(image.live, image.reference_after)]

    def reference_view(self, value: torch.Tensor) -> torch.Tensor:
        """Reinterpret a reference byte allocation using the original tensor's exact view."""
        storage = value.untyped_storage()
        image = self.images[(storage.data_ptr(), storage.nbytes())]
        return torch.empty(0, dtype=value.dtype, device=value.device).set_(
            image.reference_after.untyped_storage(), value.storage_offset(),
            value.size(), value.stride())


def logical_bytes_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Compare logical bytes in bounded chunks even for noncontiguous cache views."""
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if left.ndim == 0:
        left, right = left.reshape(1), right.reshape(1)
    row_bytes = max(1, left[0:1].numel() * left.element_size())
    if row_bytes > 8 * 1024**2:
        raise ValueError("One cache row exceeds the bounded comparison chunk")
    rows = max(1, 8 * 1024**2 // row_bytes)
    for start in range(0, left.shape[0], rows):
        a = left[start:start + rows].contiguous().reshape(-1).view(torch.uint8)
        b = right[start:start + rows].contiguous().reshape(-1).view(torch.uint8)
        if a.device != b.device:
            a, b = a.cpu(), b.cpu()
        if not torch.equal(a, b):
            return False
    return True


def replay_identical_state(
    call: Callable[[], Any], snapshots: ByteSnapshots, iterations: int,
    bindings: list[tuple[str, torch.Tensor]],
    observer: Callable[[int, Any], dict[str, Any]] | None = None,
) -> tuple[Any, list[dict[str, Any]]]:
    """Replay fixed initial bytes; serve the final attempt, never a selected favorable output."""
    if iterations != 5:
        raise ValueError("The diagnostic retains exactly five fixed replays")
    snapshots.capture_before()
    rows = []
    result: Any = None
    try:
        for index in range(iterations):
            snapshots.restore_before()
            if not snapshots.before_exact():
                raise RuntimeError("Snapshot restore failed its exact-byte check")
            result = call()
            row: dict[str, Any] = {"index": index, "initial_bytes_exact": True,
                                   "outputs": [tensor_digest(t) for t in tensor_values(result)]}
            if not row["outputs"]:
                raise RuntimeError("The captured decode graph returned no tensor")
            if index == 0:
                snapshots.capture_reference_after()
                row.update(changed_storage_count=0, changed_cache_views=[])
            else:
                changed = set(snapshots.changed_storages())
                names = []
                for name, value in bindings:
                    storage = value.untyped_storage()
                    if ((storage.data_ptr(), storage.nbytes()) in changed
                            and not logical_bytes_equal(value, snapshots.reference_view(value))):
                        names.append(name)
                row.update(changed_storage_count=len(changed), changed_cache_views=names)
            if observer is not None:
                row["observation"] = observer(index, result)
            rows.append(row)
        return result, rows
    except BaseException:
        snapshots.restore_before()
        raise
