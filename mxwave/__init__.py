"""MxWave: calibration-aware, bounded-memory MXFP4 quantization.

The engine targets vLLM's ``compressed-tensors``
``mxfp4-pack-quantized`` format. It processes source tensors in bounded chunks,
supports real activation statistics for scale selection, and verifies the
emitted checkpoint structure and reconstruction quality.
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
