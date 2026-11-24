"""Shared byte contracts without importing the old BF16 KDA experiment."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch

from ..bundle import PINNED_ASSETS, digest


def tensor_digest(value: torch.Tensor) -> str:
    """Hash exact contiguous tensor storage without a numerical cast."""
    return hashlib.sha256(value.detach().contiguous().view(torch.uint8).cpu().numpy()).hexdigest()


def validate_runtime() -> None:
    """Require the verified head library and its original SM121 device contract."""
    library = Path(os.environ["MXWAVE_ZIPSERV_LIBRARY"])
    if (digest(library) != PINNED_ASSETS["native/libmxwave_kda_zipserv.so"]
            or torch.cuda.get_device_capability() != (12, 1)):
        raise ValueError("Require the verified ZipServ ABI on an SM121 GPU")
