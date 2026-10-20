"""Hash-verified, single-sample feature residency for a separately frozen full run."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from .drafter_fp8_roundtrip import digest


class VerifiedFeatureStore:
    """Retain at most one sample; autograd may retain its inputs until backward finishes."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self._key: tuple[Path, str, tuple[int, ...]] | None = None
        self._cached: dict[str, torch.Tensor] | None = None

    def _path(self, relative: str) -> Path:
        path = (self.root / relative).resolve()
        if Path(relative).is_absolute() or not path.is_relative_to(self.root):
            raise ValueError("Feature path must stay inside its dataset")
        return path

    @staticmethod
    def _identity(path: Path) -> tuple[int, ...]:
        value = path.stat()
        return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns

    def input_ids(self, relative: str, sha256: str) -> torch.Tensor:
        """Read only token IDs for anchor selection, verifying the entire immutable payload."""
        path = self._path(relative)
        before = self._identity(path)
        if digest(path) != sha256:
            raise ValueError("Feature checksum differs from its manifest")
        with safe_open(path, framework="pt", device="cpu") as tensors:
            ids = tensors.get_tensor("input_ids").clone()
        if self._identity(path) != before:
            raise ValueError("Feature file changed while reading token IDs")
        return ids

    def get(self, relative: str, sha256: str) -> dict[str, torch.Tensor]:
        """Verify every new sample before loading, evicting the previous cached sample."""
        path = self._path(relative)
        key = (path, sha256, self._identity(path))
        if key == self._key:
            assert self._cached is not None
            return self._cached
        # Drop the previous store-owned reference before reading another payload.
        self._key, self._cached = None, None
        if digest(path) != sha256:
            raise ValueError("Feature checksum differs from its manifest")
        tensors = load_file(path, device="cpu")
        if self._identity(path) != key[2]:
            raise ValueError("Feature file changed while loading")
        self._key, self._cached = key, tensors
        return tensors
