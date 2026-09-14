"""mxstream: GPU-streaming, calibration-aware, quality-oriented MXFP4 quantization.

Built for models larger than a single machine, targeting vLLM's
``compressed-tensors`` ``mxfp4-pack-quantized`` format on Blackwell-class GPUs.

The engine streams transformer layers / shards through GPU VRAM, quantizes
on-device with calibration-aware scale selection, and emits a standard,
verified checkpoint.
"""

__version__ = "0.1.0"

from .core import BLOCK_SIZE, quantize_mxfp4
from .engine import QuantizeConfig, quantize_model
from .format import detect_input_format

__all__ = [
    "BLOCK_SIZE",
    "QuantizeConfig",
    "__version__",
    "detect_input_format",
    "quantize_model",
    "quantize_mxfp4",
]
