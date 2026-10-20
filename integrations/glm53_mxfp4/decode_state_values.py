"""Bounded, byte-preserving evidence and CPU metrics for reference decode drift."""

from __future__ import annotations

import gzip
import hashlib
import math
from pathlib import Path
from typing import Any

import torch

MAX_TENSOR_BYTES = 16 * 1024**2
METRIC_CHUNK_ELEMENTS = 128 * 1024


def value_metrics(current: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    """Compare CPU tensor bits and finite numerical errors using bounded temporary chunks."""
    if (current.device.type != "cpu" or reference.device.type != "cpu"
            or current.shape != reference.shape or current.dtype != reference.dtype):
        raise ValueError("Require matching CPU tensor shapes and dtypes")
    if current.numel() * current.element_size() > MAX_TENSOR_BYTES:
        raise ValueError("One logical tensor exceeds the bounded evidence budget")
    a, b = current.contiguous().reshape(-1), reference.contiguous().reshape(-1)
    changed_elements = changed_bytes = finite_pairs = 0
    error_energy = reference_energy = maximum = 0.0
    for start in range(0, a.numel(), METRIC_CHUNK_ELEMENTS):
        left, right = a[start:start + METRIC_CHUNK_ELEMENTS], b[start:start + METRIC_CHUNK_ELEMENTS]
        different = left.view(torch.uint8) != right.view(torch.uint8)
        changed_bytes += int(different.sum())
        changed_elements += int(different.reshape(-1, a.element_size()).any(dim=1).sum())
        x, y = left.to(torch.float64), right.to(torch.float64)
        finite = torch.isfinite(x) & torch.isfinite(y)
        finite_pairs += int(finite.sum())
        error = x[finite] - y[finite]
        if error.numel():
            maximum = max(maximum, float(error.abs().max()))
            error_energy += float(error.square().sum())
            reference_energy += float(y[finite].square().sum())
    return {"elements": a.numel(), "changed_elements_by_bits": changed_elements,
        "changed_bytes": changed_bytes, "bytes_exact": changed_bytes == 0,
        "finite_pairs": finite_pairs, "nonfinite_pairs": a.numel() - finite_pairs,
        "max_absolute_error_finite": maximum if finite_pairs else None,
        "rms_error_finite": math.sqrt(error_energy / finite_pairs) if finite_pairs else None,
        "reference_rms_finite": math.sqrt(reference_energy / finite_pairs) if finite_pairs else None,
        "relative_l2_finite": math.sqrt(error_energy / reference_energy)
            if reference_energy else (0.0 if not error_energy else None),
        "sqnr_db_finite": 10 * math.log10(reference_energy / error_energy)
            if error_energy and reference_energy else None}


def save_tensor(directory: Path, name: str, value: torch.Tensor) -> dict[str, Any]:
    """Retain a bounded CPU tensor as exact compressed bytes with a portable manifest."""
    if value.device.type != "cpu" or value.numel() * value.element_size() > MAX_TENSOR_BYTES:
        raise ValueError("Require a bounded CPU tensor for evidence recording")
    if Path(name).name != name or not name.endswith(".bin.gz"):
        raise ValueError("Require an owned artifact basename")
    directory.mkdir(parents=True, exist_ok=True)
    raw = value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    path = directory / name
    with path.open("xb") as target, gzip.GzipFile(fileobj=target, mode="wb", compresslevel=1,
                                                mtime=0) as output:
        output.write(raw)
    return {"path": name, "shape": list(value.shape), "dtype": str(value.dtype),
            "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
            "compressed_bytes": path.stat().st_size,
            "compressed_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
