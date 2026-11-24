"""Opt-in scale packing for the isolated full-model palette comparison."""

from __future__ import annotations

import hashlib
import os
from typing import Any

import model_plugin
import torch
from combined_worker import CombinedWorker

_original_pack = model_plugin.pack_scales
_source_scales: list[dict[str, Any]] = []
_packed_sources: dict[int, str] = {}


def _palette_pack(
    scales: torch.Tensor, *args: Any, encoding: str = "records", **kwargs: Any
) -> Any:
    """Hash every actual source scale group and validate policy before the bare dispatcher."""
    if encoding != "aligned_nibbles":
        return _original_pack(scales, *args, encoding=encoding, **kwargs)
    source_sha = hashlib.sha256(scales.view(torch.uint8).cpu().numpy()).hexdigest()
    _source_scales.append({"shape": list(scales.shape), "sha256": source_sha})
    native = model_plugin.native(1)
    if os.environ.get("MXWAVE_PALETTE_MODEL") == "1":
        if native.encoding != "sparse_palette":
            raise RuntimeError("Palette model was given the wrong native consumer")
        encoding = "sparse_palette"
    packed = _original_pack(scales, *args, encoding=encoding, **kwargs)
    if native.affine_only and not packed.affine_palette:
        raise ValueError("Nonconsecutive palette requires the exact raw fallback")
    _packed_sources[id(packed)] = source_sha
    return packed


model_plugin.pack_scales = _palette_pack


def install() -> None:
    """Install the proof hook before the qualified model plugin prepares any weight."""
    model_plugin.install()


class PaletteModelWorker(CombinedWorker):
    """Keep all existing source checks and expose complete prepared-scale coverage."""

    def mxwave_palette_report(self) -> dict[str, Any]:
        """Require every new packed tensor to have passed its full GPU source comparison."""
        from vllm.distributed import get_tensor_model_parallel_rank

        report = self.mxwave_scale_report()
        scales = [s for layer in report["layers"] for s in layer["scales"]]
        packed = [s for s in model_plugin._STORAGE.values() if s["kind"] == 1]
        candidate = os.environ.get("MXWAVE_PALETTE_MODEL") == "1"
        expected = 84 if candidate else 83
        if (
            report["layers_seen"] != 42
            or len(scales) != 84
            or len(packed) != expected
            or any(not s.get("gpu_bytes_exact") for s in scales if s["compressed"])
        ):
            raise RuntimeError("Prepared full-model GPU scale coverage is incomplete")
        encoding = "sparse_palette" if candidate else "aligned_nibbles"
        if any(s["packed"].encoding != encoding for s in packed):
            raise RuntimeError("A loaded scale tensor used the wrong representation")
        if candidate and any(not s["packed"].affine_palette for s in packed):
            raise RuntimeError("Loaded affine consumer has an unsupported palette")
        if len(_source_scales) != 84:
            raise RuntimeError("Full-model source hashing omitted a scale group")
        for state in packed:
            value = state["packed"]
            decoded = model_plugin.native(1).decode(value).cpu().numpy()
            if hashlib.sha256(decoded).hexdigest() != _packed_sources[id(value)]:
                raise RuntimeError("Actual full-model GPU reconstruction differs from source bytes")
        return {
            "status": "passed",
            "candidate": candidate,
            "encoding": encoding,
            "tp_rank": get_tensor_model_parallel_rank(),
            "expert_layers": 42,
            "scale_groups": 84,
            "compressed_groups": len(packed),
            "compressed_groups_gpu_source_bytes_exact": True,
            "source_scale_hashes": _source_scales,
            "source_scale_bytes": sum(s["raw_bytes"] for s in scales),
            "retained_scale_bytes": sum(s["retained_bytes"] for s in scales),
            "allocation": report,
        }
