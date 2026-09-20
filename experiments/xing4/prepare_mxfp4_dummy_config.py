"""Create a metadata-only Xing4 config that exercises vLLM's MXFP4 paths.

This is a runtime compatibility probe, not a quantized model artifact.  The
broad ``Linear`` target is intentional: with ``--load-format dummy`` it makes
vLLM instantiate both dense and fused-MoE MXFP4 methods without requiring a
converted checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _mxfp4_scheme() -> dict[str, Any]:
    """Return the compressed-tensors MXFP4 scheme used by MxWave."""
    tensor_scheme: dict[str, Any] = {
        "num_bits": 4,
        "type": "float",
        "symmetric": True,
        "group_size": 32,
        "strategy": "group",
        "block_structure": None,
        "actorder": None,
        "scale_dtype": "torch.uint8",
        "zp_dtype": None,
        "observer_kwargs": {},
    }
    weights = {
        **tensor_scheme,
        "dynamic": False,
        "observer": "memoryless_minmax",
    }
    activations = {
        **tensor_scheme,
        "dynamic": True,
        "observer": None,
    }
    return {
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": weights,
                "input_activations": activations,
                "output_activations": None,
            }
        },
        "quant_method": "compressed-tensors",
        "kv_cache_scheme": None,
        "format": "mxfp4-pack-quantized",
        "quantization_status": "compressed",
        "global_compression_ratio": None,
        "ignore": [],
    }


def main() -> int:
    """Write a dummy-probe config derived from a real Xing4 config."""
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    raw = json.loads(args.source.read_text())
    if not isinstance(raw, dict):
        raise TypeError("Source config must be a JSON object")
    if raw.get("model_type") != "xing4_0":
        raise ValueError("Source config is not Xing4")
    raw["quantization_config"] = _mxfp4_scheme()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
