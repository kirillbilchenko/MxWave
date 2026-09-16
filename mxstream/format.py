"""Input format detection from ``config.json``, not tensor-name sniffing.

The FP8/FP4 ecosystem has multiple incompatible packings:
  - block-FP8 with ``.weight_scale_inv`` (DeepSeek/Qwen)
  - per-channel ``float-quantized`` FP8 with ``.weight_scale`` (compressed-tensors)
  - native MXFP8 with uint8 e8m0 ``.weight_scale_inv``
  - plain FP16/BF16

Detecting from ``config.json`` ``quantization_config`` (quant_method / format /
strategy / scale granularity) is more robust than suffix-sniffing, which
misclassifies non-block packings and quantizes raw FP8 bytes without scales.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InputFormat:
    """Normalized description of the source checkpoint's quantization format."""

    # Known values include: fp16, fp8_block, fp8_per_channel, mxfp8, mxfp4,
    # nvfp4, and unknown_quantized.
    kind: str
    quant_method: str | None = None
    scale_suffix: str | None = None
    block_size: int | None = None


def load_config_json(model_dir: str | Path) -> dict[str, object]:
    """Load the model's ``config.json`` as a dict (empty if absent)."""
    path = Path(model_dir) / "config.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def detect_input_format(model_dir: str | Path) -> InputFormat:
    """Detect input quantization format from config.json (defaulting to fp16)."""
    cfg = load_config_json(model_dir)
    raw_qcfg = cfg.get("quantization_config")
    qcfg: dict[str, object] = raw_qcfg if isinstance(raw_qcfg, dict) else {}

    quant_method = str(qcfg.get("quant_method", "")).lower()
    fmt = str(qcfg.get("format", "")).lower()
    strategy = str(qcfg.get("strategy", "")).lower()
    quant_algo = str(qcfg.get("quant_algo", "")).lower()
    # Native MXFP8 / MXFP4 (e8m0 uint8 scales)
    if "mxfp" in fmt or "mxfp" in quant_method:
        if "fp8" in fmt or "fp8" in quant_method:
            return InputFormat(kind="mxfp8", quant_method=quant_method, block_size=32)
        return InputFormat(kind="mxfp4", quant_method=quant_method, block_size=32)

    if "nvfp4" in fmt or "nvfp4" in quant_method or "nvfp4" in quant_algo:
        return InputFormat(kind="nvfp4", quant_method=quant_method, block_size=16)

    # compressed-tensors float-quantized (per-channel or per-tensor FP8)
    if "float" in fmt:
        granularity = strategy  # "channel" or "tensor"
        return InputFormat(
            kind="fp8_per_channel" if "channel" in granularity else "fp8_block",
            quant_method=quant_method,
            scale_suffix=".weight_scale",
            block_size=1 if "channel" in granularity else None,
        )

    # Block-FP8 with weight_scale_inv (DeepSeek / Qwen native)
    if "fp8" in fmt or "fp8" in quant_method:
        return InputFormat(
            kind="fp8_block",
            quant_method=quant_method,
            scale_suffix=".weight_scale_inv",
            block_size=128,
        )

    # A non-empty but unfamiliar quantization_config is never safe to treat as
    # dense floats. This catches AWQ/GPTQ/ModelOpt variants without relying on
    # tensor suffixes.
    if qcfg:
        return InputFormat(kind="unknown_quantized", quant_method=quant_method)

    return InputFormat(kind="fp16")


def scale_suffix_for(format: InputFormat) -> str:
    """Return the scale tensor suffix for a detected input format."""
    if format.scale_suffix is not None:
        return format.scale_suffix
    # Defaults by kind
    if format.kind in ("fp8_block", "mxfp8"):
        return ".weight_scale_inv"
    return ""
