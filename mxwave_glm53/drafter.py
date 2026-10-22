"""Preserve the pinned mixed-precision Modal checkpoint under native vLLM 0.32."""

from __future__ import annotations

import errno
import json
import os
import shutil
import struct
import tempfile
from pathlib import Path
from typing import Any

from .bundle import DRAFTER_FILES, digest, local_file, verify_files

PREPARED_DRAFTER_FILES = {
    **DRAFTER_FILES,
    "config.json": "28df83230b06fb5460eed2105357e81f09560124b0b4d65366d7afcd70dfa089",
}


def verify_drafter(root: Path) -> None:
    """Reject the original incompatible metadata and require the GPU-tested correction."""
    if digest(local_file(root, "config.json")) == DRAFTER_FILES["config.json"]:
        raise ValueError(
            "Original Modal config misloads BF16 convolutions as FP8 on vLLM 0.32; "
            "run mxwave-glm53 prepare-drafter --source ORIGINAL --output PREPARED "
            "and serve with --drafter PREPARED")
    verify_files(root, PREPARED_DRAFTER_FILES)


def corrected_config(config: dict[str, Any], header: dict[str, Any]) -> dict[str, Any]:
    """Exclude twelve existing BF16 projections under local and global layer names."""
    result: dict[str, Any] = json.loads(json.dumps(config))
    quant = result["quantization_config"]
    if quant["quant_method"] != "fp8" or result["num_hidden_layers"] != 6:
        raise ValueError("Require the pinned six-layer Modal FP8 DFlash2 config")
    for index in range(6):
        for side in ("attention_conv", "mlp_conv"):
            name = f"layers.{index}.{side}.kernel_projection"
            tensor = header[name + ".weight"]
            if tensor["dtype"] != "BF16" or tensor["shape"] != [1024, 4096]:
                raise ValueError(f"Expected BF16 [1024,4096] projection: {name}")
            if any(key.startswith(name + ".") and "scale" in key for key in header):
                raise ValueError(f"Unexpected quantization scales: {name}")
            for path in ("model." + name,
                         f"model.layers.{index + 45}.{side}.kernel_projection"):
                if path not in quant["ignored_layers"]:
                    quant["ignored_layers"].append(path)
    return result


def prepare_drafter(source: Path, output: Path) -> dict[str, Any]:
    """Atomically prepare config-only compatibility; keep weight bytes and source intact."""
    if output.exists():
        raise FileExistsError(f"Preserve existing drafter directory: {output}")
    verify_files(source, DRAFTER_FILES)
    weight = local_file(source, "model.safetensors")
    with weight.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        if not 0 < size <= min(16 * 1024**2, weight.stat().st_size - 8):
            raise ValueError("Invalid safetensors header length")
        header = json.loads(stream.read(size))
    config = json.loads(local_file(source, "config.json").read_bytes())
    corrected = corrected_config(config, header)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".mxwave-drafter-", dir=output.parent) as tmp:
        staged = Path(tmp) / "drafter"
        staged.mkdir()
        (staged / "config.json").write_text(json.dumps(corrected, indent=2) + "\n")
        verify_files(staged, {"config.json": PREPARED_DRAFTER_FILES["config.json"]})
        transfers = {}
        for path in sorted(source.iterdir()):
            if not path.is_file() or path.name in ("config.json", "mxwave-drafter.json"):
                continue
            src = local_file(source, path.name)
            try:
                os.link(src, staged / path.name)
                transfers[path.name] = "hardlink"
            except OSError as exc:
                if exc.errno not in (errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP):
                    raise
                shutil.copyfile(src, staged / path.name)
                transfers[path.name] = "copy"
        verify_drafter(staged)
        receipt = {
            "schema": 1, "profile": "modal-dflash2-glm53-vllm032-bf16-conv-v1",
            "source_files": DRAFTER_FILES, "prepared_files": PREPARED_DRAFTER_FILES,
            "transfers": transfers,
            "change": "Quantization metadata only; twelve existing BF16 conv projections",
            "qualification": "GPU deployment screen complete; quality/repeatability gates not passed",
        }
        (staged / "mxwave-drafter.json").write_text(json.dumps(receipt, indent=2) + "\n")
        if output.exists():
            raise FileExistsError(f"Preserve existing drafter directory: {output}")
        staged.rename(output)
    return receipt
