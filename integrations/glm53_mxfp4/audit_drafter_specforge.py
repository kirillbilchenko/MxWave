"""Audit a pinned upstream DFlash2 class against the CPU-converted checkpoint."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from .drafter_fp8_roundtrip import digest


def audit(checkout: Path, checkpoint: Path, output: Path) -> dict[str, Any]:
    """Instantiate the upstream class on meta and strictly load all BF16 parameters."""
    import transformers
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    revision = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True,
    ).strip()
    if revision != "03f3682ea852b73488d62dab4cd539033cc60e3e":
        raise ValueError("SpecForge revision differs from the reviewed source")
    if subprocess.check_output(
        ["git", "-C", str(checkout), "status", "--porcelain"], text=True,
    ).strip():
        raise ValueError("Upstream audit checkout must remain unmodified")
    sys.path.insert(0, str(checkout.resolve()))
    # Avoid unrelated Eagle/MTP aggregate imports and accelerator dependencies.
    # The actual DFlash2, DFlash, kernels and registry modules load unmodified.
    namespace = types.ModuleType("specforge.modeling.draft")
    namespace.__path__ = [str((checkout / "specforge/modeling/draft").resolve())]
    sys.modules["specforge.modeling.draft"] = namespace
    upstream = importlib.import_module("specforge.modeling.draft.dflash2")
    config_dict = json.loads((checkpoint / "config.json").read_text())
    if "quantization_config" in config_dict:
        raise ValueError("Audit expects the dense training checkpoint, not raw FP8 codes")
    config = Qwen3Config(**config_dict)
    config._attn_implementation = "eager"
    with torch.device("meta"):
        model = upstream.DFlash2DraftModel(config)
    expected = {name: list(value.shape) for name, value in model.state_dict().items()}
    weights = load_file(str(checkpoint / "model.safetensors"), device="cpu")
    result = model.load_state_dict(weights, strict=True, assign=True)
    rows = []
    for name, value in model.state_dict().items():
        if value.is_meta or value.device.type != "cpu":
            raise ValueError(f"Unmaterialized or non-CPU tensor: {name}")
        if value.data_ptr() != weights[name].data_ptr():
            raise ValueError(f"Strict assigned tensor differs from source: {name}")
        rows.append({
            "checkpoint_key": name, "specforge_key": name,
            "shape": expected[name], "dtype": str(value.dtype),
            "train_context_first": name == "fc.weight",
        })
    source_paths = [
        "specforge/modeling/draft/dflash2.py", "specforge/modeling/draft/dflash.py",
        "specforge/modeling/draft/dflash_kernels.py",
        "specforge/training/model_loading.py", "specforge/export/to_hf.py",
        "pyproject.toml", "LICENSE",
    ]
    report = {
        "schema": "mxwave-drafter-specforge-compatibility-v1",
        "specforge_revision": revision, "torch_version": str(torch.__version__),
        "transformers_version": transformers.__version__,
        "pinned_specforge_torch_requirement": "2.13.0",
        "checkpoint_sha256": digest(checkpoint / "model.safetensors"),
        "source_code_sha256": digest(Path(__file__)),
        "source_hashes": {name: digest(checkout / name) for name in source_paths},
        "strict_load_pass": True, "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys), "parameter_tensors": len(rows),
        "parameters": sum(value.numel() for value in model.parameters()),
        "context_projection_parameters": weights["fc.weight"].numel(),
        "layer_count": config.num_hidden_layers,
        "target_layers": config_dict["dflash_config"]["target_layer_ids"],
        "layer_types": config_dict["layer_types"], "block_size": model.block_size,
        "keys_and_shapes": rows,
        "limits": [
            "Architecture/strict CPU weight loading only; no forward, backward, optimizer or native GPU execution.",
            "Package aggregate imports intentionally bypassed; full SpecForge CLI environment unvalidated.",
            "Local Torch 2.14.0 differs from SpecForge's pinned 2.13.0 training requirement.",
            "35 FP8 scale tensors live in the export sidecar/original checkpoint, not the dense training model.",
            "Native FP8 Marlin weights/activation arithmetic and draft KV/head adapters require a serving parity check.",
            "Target features, target head and embeddings must come from the fixed CT target; not owned by this drafter checkpoint.",
            "Existing selector-only capture omits full context feature sequences required for context-projection training.",
        ],
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    """Run the isolated CPU architecture/load audit and summarize its coverage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = audit(args.checkout, args.checkpoint, args.output)
    print(json.dumps({k: v for k, v in report.items() if k != "keys_and_shapes"}, indent=2))


if __name__ == "__main__":
    main()
