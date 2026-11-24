"""CPU-only audit of real GLM fused-attention selectors against checkpoint headers."""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import struct
from collections import Counter
from pathlib import Path
from typing import Any

from attention_mapping import FUSED_ATTENTION_COMPONENTS, register_model_mapping


def main() -> None:
    """Compare installed selectors before/after the opt-in repair, without a model load."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("Preserve previous selector evidence")

    from vllm import platforms
    from vllm.platforms.cpu import CpuPlatform

    if platforms.current_platform.is_unspecified():
        platforms.current_platform = CpuPlatform()
    from torch import nn
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )
    from vllm.model_executor.model_loader.utils import configure_quant_config
    from vllm.models.glm5next.nvidia.model import Glm5NextForConditionalGeneration as Model

    config = json.loads((args.source / "config.json").read_bytes())
    index = json.loads((args.source / "model.safetensors.index.json").read_bytes())["weight_map"]
    headers: dict[str, Any] = {}
    header_hashes = {}
    for shard in sorted(set(index.values())):
        with (args.source / shard).open("rb") as handle:
            length = struct.unpack("<Q", handle.read(8))[0]
            if length > 64 * 1024**2:
                raise ValueError("Unexpectedly large safetensors header")
            raw = handle.read(length)
        header_hashes[shard] = hashlib.sha256(raw).hexdigest()
        for name, entry in json.loads(raw).items():
            if name != "__metadata__" and index.get(name) == shard:
                headers[name] = entry
    if set(headers) != set(index):
        raise ValueError("Missing active indexed tensors")

    before = CompressedTensorsConfig.from_config(copy.deepcopy(config["quantization_config"]))
    configure_quant_config(before, Model)
    original_mapping = copy.deepcopy(before.packed_modules_mapping)
    register_model_mapping(Model)
    after = CompressedTensorsConfig.from_config(copy.deepcopy(config["quantization_config"]))
    configure_quant_config(after, Model)
    dummy = nn.Linear(2, 2, bias=False)

    def scheme(quant: Any, name: str) -> str | None:
        selected = quant.get_scheme_dict(dummy, name)
        return str(selected["format"]) if selected else None

    records = []
    for layer in range(config["text_config"]["num_hidden_layers"]):
        for fused, parts in FUSED_ATTENTION_COMPONENTS.items():
            prefix = f"model.language_model.layers.{layer}.self_attn."
            keys = [prefix + part + ".weight" for part in parts]
            if not all(key in headers for key in keys):
                continue
            name = Model.hf_to_vllm_mapper._map_name(prefix + fused)
            source = [{"name": key, "dtype": headers[key]["dtype"],
                       "shape": headers[key]["shape"],
                       "has_scale": any(key.removesuffix(".weight") + suffix in index
                                        for suffix in (".weight_scale", ".weight_scale_inv"))}
                      for key in keys]
            records.append({"name": name, "fused": fused, "source": source,
                "selector_before": scheme(before, name), "selector_after": scheme(after, name),
                "constructor_forces_unquantized": fused == "wk_weights_proj"})

    controls = []
    for layer, suffix in ((0, "mlp.gate_up_proj"), (0, "mlp.down_proj"),
                          (3, "mlp.shared_experts.gate_up_proj"),
                          (3, "mlp.shared_experts.down_proj"),
                          (3, "mlp.experts.0.down_proj"),
                          (0, "self_attn.o_proj"), (0, "self_attn.f_b_proj"),
                          (3, "self_attn.q_b_proj")):
        name = f"language_model.model.layers.{layer}.{suffix}"
        controls.append({"name": name, "before": scheme(before, name), "after": scheme(after, name)})

    passed = bool(records and all(
        row["selector_after"] is None
        and all(item["dtype"] == "BF16" and not item["has_scale"] for item in row["source"])
        for row in records) and all(row["before"] == row["after"] for row in controls))
    source_files = {"model": inspect.getfile(Model),
                    "selector": inspect.getfile(CompressedTensorsConfig),
                    "configure_quant_config": inspect.getfile(configure_quant_config),
                    "audit": str(Path(__file__)),
                    "mapping": str(Path(__file__).with_name("attention_mapping.py"))}
    result = {"status": "passed" if passed else "failed", "passed": passed,
        "scope": "CPU selector and checkpoint-header audit; no loaded-model or quality claim",
        "source_sha256": {name: hashlib.sha256((args.source / name).read_bytes()).hexdigest()
                          for name in ("config.json", "model.safetensors.index.json")},
        "runtime_source_sha256": {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                  for name, path in source_files.items()},
        "safetensors_header_sha256": header_hashes,
        "mapping_before": original_mapping, "mapping_after": after.packed_modules_mapping,
        "fused_counts": dict(Counter(row["fused"] for row in records)),
        "unexpected_quantized_selectors_before": sum(row["selector_before"] is not None
                                                     for row in records),
        "unexpected_quantized_selectors_after": sum(row["selector_after"] is not None
                                                    for row in records),
        "records": records, "unchanged_controls": controls}
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("records", "safetensors_header_sha256")}))
    if not passed:
        raise RuntimeError("Fused-attention CPU audit failed")


if __name__ == "__main__":
    main()
