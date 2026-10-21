"""Describe the pinned Modal drafter's existing BF16 weights accurately for vLLM 0.32."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from pathlib import Path
from typing import Any


def corrected_config(config: dict[str, Any], header: dict[str, Any],
                     target_layers: int) -> tuple[dict[str, Any], list[str]]:
    """Exclude existing BF16 conv projections under local and native global layer names."""
    result = json.loads(json.dumps(config))
    quant = result["quantization_config"]
    if quant["quant_method"] != "fp8" or result["num_hidden_layers"] != 6 or target_layers != 45:
        raise ValueError("This correction is pinned to Modal GLM-5.3 DFlash2 with 45 target layers")
    paths = []
    for index in range(result["num_hidden_layers"]):
        for side in ("attention_conv", "mlp_conv"):
            name = f"layers.{index}.{side}.kernel_projection"
            if header[name + ".weight"]["dtype"] != "BF16":
                raise ValueError(f"Expected existing BF16 weight: {name}")
            if any(k.startswith(name + ".") and "scale" in k for k in header):
                raise ValueError(f"Unexpected quantization scales: {name}")
            paths += ["model." + name,
                      f"model.layers.{index + target_layers}.{side}.kernel_projection"]
    ignored = quant["ignored_layers"]
    for path in paths:
        if path not in ignored:
            ignored.append(path)
    return result, paths


def main() -> None:
    """Create a separate config with byte-identical hard-linked weights, never edit the source."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source, args.output
    if hashlib.sha256((source / "config.json").read_bytes()).hexdigest() != (
            "c54c63c667baca52604519c5ac0d0d053210e3a3231dfb626d4758b1b6f8655e"):
        raise ValueError("Unrecognized original drafter configuration")
    config = json.loads((source / "config.json").read_bytes())
    weight = source / "model.safetensors"
    with weight.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(size))
    fixed, exclusions = corrected_config(config, header, 45)
    with weight.open("rb") as stream:
        weight_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    if weight_hash != "45f34ef126aa258d8e16bd605a3a05c36eb3fef3ed0f57aef20191db49f1ef46":
        raise ValueError("Unrecognized drafter weights; do not launch")
    if output.exists():
        existing = json.loads((output / "compatibility.json").read_bytes())
        if (json.loads((output / "config.json").read_bytes()) != fixed
                or not os.path.samefile(weight, output / weight.name)
                or existing["weights_sha256"] != weight_hash):
            raise ValueError("Existing corrected drafter does not match; preserve it")
        print(json.dumps(existing))
        return
    output.mkdir(exist_ok=False)
    for path in source.iterdir():
        if path.is_file() and path.name != "config.json":
            os.link(path, output / path.name)
    (output / "config.json").write_text(json.dumps(fixed, indent=2) + "\n")
    receipt = {
        "source": str(source), "output": str(output), "weights_sha256": weight_hash,
        "weights_same_inode": os.path.samefile(weight, output / weight.name),
        "original_config_sha256": hashlib.sha256((source / "config.json").read_bytes()).hexdigest(),
        "corrected_config_sha256": hashlib.sha256((output / "config.json").read_bytes()).hexdigest(),
        "added_exclusions": exclusions,
        "changed": "Quantization metadata only; preserve 12 existing BF16 conv projections",
        "gpu_qualified": False,
    }
    (output / "compatibility.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
