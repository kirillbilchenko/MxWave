"""SSD-backed exact snapshots for memory-bounded, intrusive graph diagnostics."""

from __future__ import annotations

import mmap
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch
from decode_state_replay import ByteSnapshots, StorageImage

CHUNK_BYTES = 8 * 1024**2


class DiskByteSnapshots(ByteSnapshots):
    """Keep both byte images on SSD; transfer and compare at most eight MiB at once."""

    def __init__(self, tensors: list[torch.Tensor], maximum_bytes: int, directory: Path) -> None:
        """Reserve owned files before admission without retaining additional GPU cache copies."""
        unique: dict[tuple[int, int], torch.Tensor] = {}
        device = None
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
        self.images: dict[tuple[int, int], StorageImage] = {}
        self._mappings: list[mmap.mmap] = []
        self._mapping_by_pointer: dict[int, mmap.mmap] = {}
        self._directories: list[Any] = []
        if not unique or self.payload_bytes > maximum_bytes:
            raise ValueError("Empty snapshot or storage byte budget exceeded")
        if shutil.disk_usage(directory).free < 2 * self.payload_bytes + 512 * 1024**2:
            raise MemoryError("Insufficient SSD capacity for both exact byte images")
        if device is not None and device.type == "cuda":
            free, _ = torch.cuda.mem_get_info(device)
            if free < 128 * 1024**2:
                raise MemoryError("Require 128 MiB for bounded diagnostic work buffers")
        temporary = tempfile.TemporaryDirectory(prefix="mxwave-decode-images-", dir=directory)
        self._directories.append(temporary)
        try:
            for index, (key, live) in enumerate(unique.items()):
                copies = []
                for phase in ("before", "after"):
                    path = Path(temporary.name) / f"{index}-{phase}.bin"
                    with path.open("w+b") as output:
                        if hasattr(os, "posix_fallocate"):
                            os.posix_fallocate(output.fileno(), 0, key[1])
                        else:
                            output.truncate(key[1])
                        mapping = mmap.mmap(output.fileno(), key[1], access=mmap.ACCESS_WRITE)
                    self._mappings.append(mapping)
                    view = torch.frombuffer(mapping, dtype=torch.uint8)
                    copies.append(view)
                    self._mapping_by_pointer[view.data_ptr()] = mapping
                    del view
                self.images[key] = StorageImage(key, live, copies[0], copies[1])
        except BaseException:
            copies.clear()
            self.images.clear()
            self.close()
            raise

    def _release_pages(self, flush: bool = False) -> None:
        """Flush written images and let the OS reclaim their file-backed resident pages."""
        for mapping in self._mappings:
            if flush:
                mapping.flush()
            if hasattr(mapping, "madvise") and hasattr(mmap, "MADV_DONTNEED"):
                mapping.madvise(mmap.MADV_DONTNEED)

    def _copy(self, target: torch.Tensor, source: torch.Tensor) -> None:
        """Copy complete storage in fixed chunks, preserving its raw bit representation."""
        for start in range(0, source.numel(), CHUNK_BYTES):
            target[start:start + CHUNK_BYTES].copy_(source[start:start + CHUNK_BYTES])
            for value, written in ((target, True), (source, False)):
                mapping = self._mapping_by_pointer.get(value.data_ptr())
                if mapping is not None:
                    count = min(CHUNK_BYTES, source.numel() - start)
                    if written:
                        mapping.flush(start, count)
                    if hasattr(mapping, "madvise") and hasattr(mmap, "MADV_DONTNEED"):
                        mapping.madvise(mmap.MADV_DONTNEED, start, count)

    def _equal(self, left: torch.Tensor, right: torch.Tensor) -> bool:
        """Compare both byte arrays on CPU using bounded transfer buffers."""
        for start in range(0, left.numel(), CHUNK_BYTES):
            equal = torch.equal(left[start:start + CHUNK_BYTES].cpu(),
                                right[start:start + CHUNK_BYTES].cpu())
            for value in (left, right):
                mapping = self._mapping_by_pointer.get(value.data_ptr())
                if (mapping is not None and hasattr(mapping, "madvise")
                        and hasattr(mmap, "MADV_DONTNEED")):
                    mapping.madvise(mmap.MADV_DONTNEED, start,
                                    min(CHUNK_BYTES, left.numel() - start))
            if not equal:
                return False
        return True

    def capture_before(self) -> None:
        """Persist all original bytes before running any replay."""
        for image in self.images.values():
            self._copy(image.before, image.live)
        self._release_pages(flush=True)

    def restore_before(self) -> None:
        """Restore every tracked byte from its disk image without changing graph addresses."""
        for image in self.images.values():
            self._copy(image.live, image.before)
        self._release_pages()

    def before_exact(self) -> bool:
        """Check restored live bytes against the complete original image, not only a digest."""
        result = all(self._equal(image.live, image.before) for image in self.images.values())
        self._release_pages()
        return result

    def capture_reference_after(self) -> None:
        """Persist the first replay's complete state as the exact comparison reference."""
        for image in self.images.values():
            self._copy(image.reference_after, image.live)
        self._release_pages(flush=True)

    def changed_storages(self) -> list[tuple[int, int]]:
        """Identify any complete allocation whose result differs from the first replay."""
        result = [key for key, image in self.images.items()
                  if not self._equal(image.live, image.reference_after)]
        self._release_pages()
        return result

    def reference_view(self, value: torch.Tensor) -> torch.Tensor:
        """Reconstruct the original typed view over the SSD-backed CPU reference image."""
        storage = value.untyped_storage()
        image = self.images[(storage.data_ptr(), storage.nbytes())]
        return torch.empty(0, dtype=value.dtype, device="cpu").set_(
            image.reference_after.untyped_storage(), value.storage_offset(),
            value.size(), value.stride())

    def absorb(self, other: DiskByteSnapshots) -> None:
        """Transfer disjoint bounded metadata images and their resource ownership."""
        if self.images.keys() & other.images.keys():
            raise ValueError("Metadata images must not duplicate tracked allocations")
        self.images.update(other.images)
        self.payload_bytes += other.payload_bytes
        self._mappings.extend(other._mappings)
        self._mapping_by_pointer.update(other._mapping_by_pointer)
        self._directories.extend(other._directories)
        other.images.clear()
        other._mappings.clear()
        other._mapping_by_pointer.clear()
        other._directories.clear()

    def close(self) -> None:
        """Remove only owned temporary files and release their mappings after probe completion."""
        self.images.clear()
        for mapping in self._mappings:
            mapping.close()
        self._mappings.clear()
        self._mapping_by_pointer.clear()
        for owned in self._directories:
            owned.cleanup()
        self._directories.clear()
