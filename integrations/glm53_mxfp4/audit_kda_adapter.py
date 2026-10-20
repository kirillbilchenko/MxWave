"""CPU-only audit of all selected KDA source headers and actual vLLM method selection."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import struct
from pathlib import Path
from typing import Any

from kda_policy import KDA_PARTS, load_policy


def main() -> None:
    """Require all 68 selected paths to be BF16 and dispatch to the opt-in adapter."""
    import torch
    from vllm import platforms
    from vllm.platforms.cpu import CpuPlatform

    if platforms.current_platform.is_unspecified():
        platforms.current_platform = CpuPlatform()
    from attention_mapping import install as fix_mapping
    from kda_adapter import KdaLinearMethod, install
    from vllm.model_executor.layers.linear import RowParallelLinear
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )
    from vllm.model_executor.model_loader.utils import configure_quant_config
    from vllm.models.glm5next.nvidia.model import Glm5NextForConditionalGeneration

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("Preserve prior CPU method-selection evidence")
    targets = load_policy(args.source)
    config = json.loads((args.source / "config.json").read_bytes())
    index = json.loads((args.source / "model.safetensors.index.json").read_bytes())["weight_map"]
    headers, header_hashes = {}, {}
    for file in sorted(set(index.values())):
        with (args.source / file).open("rb") as source:
            count = struct.unpack("<Q", source.read(8))[0]
            if count > 64 * 1024**2:
                raise ValueError("Unbounded safetensors header")
            raw = source.read(count)
        header_hashes[file] = hashlib.sha256(raw).hexdigest()
        headers.update({name: value for name, value in json.loads(raw).items()
                        if name != "__metadata__" and index.get(name) == file})
    fix_mapping()
    install(args.source)
    quant = CompressedTensorsConfig.from_config(copy.deepcopy(config["quantization_config"]))
    configure_quant_config(quant, Glm5NextForConditionalGeneration)
    dummy = object.__new__(RowParallelLinear)
    torch.nn.Module.__init__(dummy)
    rows = []
    for name, target in targets.items():
        selected = quant.get_quant_method(dummy, name)
        if not isinstance(selected, KdaLinearMethod):
            raise TypeError("A selected KDA path missed the actual vLLM method hook")
        prefix = f"model.language_model.layers.{target.layer}.self_attn."
        pieces = KDA_PARTS if target.kind == "input" else ("o_proj",)
        shapes = ([[8192, 4096]] * 3 + [[64, 4096], [128, 4096], [128, 4096]]
                  if target.kind == "input" else [[4096, 8192]])
        for part, shape in zip(pieces, shapes, strict=True):
            key = prefix + part + ".weight"
            if headers[key]["dtype"] != "BF16" or headers[key]["shape"] != shape:
                raise ValueError("A selected KDA source has an unexpected dtype or shape")
        rows.append({"prefix": name, "method": type(selected).__name__,
                     "source_headers_verified": True, "logical_shape": [target.rows, target.width]})
    controls: list[dict[str, Any]] = []
    for name in ("language_model.model.layers.3.self_attn.fused_qkv_a_proj",
                 "language_model.model.layers.3.self_attn.o_proj"):
        method = quant.get_quant_method(dummy, name)
        if isinstance(method, KdaLinearMethod):
            raise TypeError("An MLA control was selected")
        controls.append({"prefix": name, "method": type(method).__name__})
    result = {"status": "passed", "passed": True, "targets": rows, "controls": controls,
              "safetensors_header_sha256": header_hashes,
              "code_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()
                              ).hexdigest() for name in ("kda_policy.py", "kda_adapter.py",
                                                        "audit_kda_adapter.py")},
              "limits": ["CPU headers and method selection; real tensor execution is a separate gate."]}
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "selected": len(rows), "controls": len(controls)}))


if __name__ == "__main__":
    main()
