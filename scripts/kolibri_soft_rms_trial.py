"""Reconstruct and verify a compact softened-RMS checkpoint during CPU loading."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def composed_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    specification: Path,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Apply changed blocks and verify every effective payload before yielding it."""
    spec_raw = specification.read_bytes()
    spec = json.loads(spec_raw)
    if spec["format"] != "mxwave-kolibri-soft-rms-patch-v1":
        raise ValueError("Unknown compact-patch format")
    if _sha(Path(spec["baseline"]) / "mxwave-manifest.json") != spec["baseline_manifest_sha256"]:
        raise ValueError("Soft RMS parent manifest changed")
    if (
        _sha(Path(spec["counts_file"])) != spec["counts_sha256"]
        or _sha(Path(spec["statistics_file"])) != spec["calibration_file_sha256"]
    ):
        raise ValueError("Soft RMS calibration or counts changed")
    expected, patches = spec["effective_payload_sha256"], spec["patches"]
    payload_targets = {
        target.removesuffix(".weight") + suffix: (target, kind)
        for target in patches
        for suffix, kind in ((".weight_packed", "packed"), (".weight_scale", "scales"))
    }
    if not set(payload_targets).issubset(expected):
        raise ValueError("Foreign patch target")
    seen, validated = set(), set()
    with ExitStack() as stack:
        readers = {}
        for filename, integrity in spec["patch_files"].items():
            path = Path(filename)
            if path.stat().st_size != integrity["bytes"] or _sha(path) != integrity["sha256"]:
                raise ValueError("Compact patch file changed")
            readers[filename] = stack.enter_context(safe_open(filename, framework="pt"))
        for name, tensor in weights:
            if name in seen or name not in expected:
                raise ValueError(f"Duplicate or foreign trial tensor: {name}")
            if tensor.device.type != "cpu" or not tensor.is_contiguous():
                raise ValueError("Patch verification requires contiguous CPU payloads")
            if name in payload_targets:
                target, kind = payload_targets[name]
                row = patches[target]
                reader = readers[row["file"]]
                indices = reader.get_tensor(row["prefix"] + "indices")
                packed = reader.get_tensor(row["prefix"] + "packed")
                scales = reader.get_tensor(row["prefix"] + "scales")
                if target not in validated:
                    total = row["rows"] * (row["columns"] // 32)
                    if (
                        indices.dtype != torch.int64
                        or indices.ndim != 1
                        or len(indices) != row["changed_blocks"]
                        or not len(indices)
                        or int(indices[0]) < 0
                        or int(indices[-1]) >= total
                        or bool((indices[1:] <= indices[:-1]).any())
                        or packed.shape != (len(indices), 16)
                        or packed.dtype != torch.uint8
                        or scales.shape != (len(indices),)
                        or scales.dtype != torch.uint8
                    ):
                        raise ValueError(f"Invalid patch block geometry: {target}")
                    validated.add(target)
                shape = (row["rows"], row["columns"] // (2 if kind == "packed" else 32))
                if tuple(tensor.shape) != shape or tensor.dtype != torch.uint8:
                    raise ValueError(f"Parent patch geometry changed: {name}")
                tensor = tensor.clone()
                if kind == "packed":
                    tensor.reshape(-1, 16)[indices] = packed
                else:
                    tensor.reshape(-1)[indices] = scales
            digest = hashlib.sha256(
                tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            ).hexdigest()
            if digest != expected[name]:
                raise ValueError(f"Effective trial payload hash mismatch: {name}")
            seen.add(name)
            yield name, tensor
    if seen != set(expected) or validated != set(patches) or specification.read_bytes() != spec_raw:
        raise ValueError("Incomplete or changed compact-patch trial")
    audit = {
        "status": "passed",
        "specification_sha256": hashlib.sha256(spec_raw).hexdigest(),
        "verified_payloads": len(seen),
        "patched_targets": len(validated),
    }
    specification.with_name("load-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
