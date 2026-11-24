"""CPU-only preparation of small real GLM fixtures for the existing ZipServ kernel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import time
from pathlib import Path
from typing import Any

import torch
from diagnose_mxfp8 import CONFIG_SHA, INDEX_SHA
from safetensors import safe_open
from safetensors.torch import save_file

LIBRARY_SHA = "6d51630cd075a8c0dcbb957594341fb21e42d4b4634bb1c92735275d81a70f62"
RUNTIME_SHA = "2a296929bf8363ec6bd9358c8eadbfcca47b77522bf030c8096ab22bd6441c35"


def tensor_digest(value: torch.Tensor) -> str:
    """Hash exact contiguous BF16 storage without float conversion or a byte copy."""
    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy()).hexdigest()


def rebuild_weight(source: Path, index: dict[str, str], case: dict[str, Any]) -> torch.Tensor:
    """Reassemble recorded TP slices and explicit NoPE zeros on CPU."""
    parts = []
    for item in case["source_tensors"]:
        with safe_open(source / index[item["name"]], framework="pt", device="cpu") as handle:
            view = handle.get_slice(item["name"])
            first, last = item["range"]
            value = view[first:last] if item["axis"] == 0 else view[:, first:last]
            if value.dtype != torch.bfloat16:
                raise ValueError("BF16 component preparation rejects implicit dtype casts")
            parts.append(value.contiguous())
    weight = torch.cat(parts)
    if case.get("nope_rows"):
        weight = torch.cat([weight, torch.zeros((case["nope_rows"], weight.shape[1]),
                                               dtype=torch.bfloat16)])
    return weight.contiguous()


def main() -> None:
    """Pack representative KDA input/output and MLA input fixtures without a GPU."""
    from mxwave_zipserv import runtime

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True)
    report_path = args.output / "prepared.json"
    if report_path.exists():
        raise FileExistsError("Preserve prepared attention fixtures")
    library = Path(os.environ["MXWAVE_ZIPSERV_LIBRARY"])
    if hashlib.sha256(library.read_bytes()).hexdigest() != LIBRARY_SHA:
        raise ValueError("Use the previously qualified ZipServ native library")
    if hashlib.sha256(Path(runtime.__file__).read_bytes()).hexdigest() != RUNTIME_SHA:
        raise ValueError("The existing ZipServ runtime source changed")
    for file, expected in (("config.json", CONFIG_SHA),
                           ("model.safetensors.index.json", INDEX_SHA)):
        if hashlib.sha256((args.source / file).read_bytes()).hexdigest() != expected:
            raise ValueError("The audited imported source metadata changed")
    index = json.loads((args.source / "model.safetensors.index.json").read_bytes())["weight_map"]
    config = json.loads((args.source / "config.json").read_bytes())["text_config"]
    if config["qk_rope_head_dim"] != 0:
        raise ValueError("These fixtures require the audited zero-RoPE attention dimensions")
    torch.set_num_threads(1)
    started = time.monotonic()

    def item(layer: int, projection: str, axis: int, first: int, last: int) -> dict[str, Any]:
        return {"name": f"model.language_model.layers.{layer}.self_attn.{projection}.weight",
                "axis": axis, "range": [first, last]}

    cases = [
        {"name": "kda-input-l0-r0", "rank": 0, "source_tensors": [
            item(0, name, 0, 0, count) for name, count in (
                ("q_proj", 4096), ("k_proj", 4096), ("v_proj", 4096),
                ("b_proj", 32), ("f_a_proj", 128), ("g_a_proj", 128))]},
        {"name": "mla-input-l3-replicated", "rank": "replicated", "source_tensors": [
            item(3, "q_a_proj", 0, 0, 1536), item(3, "kv_a_proj_with_mqa", 0, 0, 512)],
         "nope_rows": 0},
        {"name": "kda-output-l0-r0", "rank": 0, "source_tensors": [
            item(0, "o_proj", 1, 0, 4096)]},
    ]
    result: dict[str, Any] = {"status": "packing", "device": "cpu", "cases": [],
        "library_sha256": LIBRARY_SHA, "runtime_sha256": RUNTIME_SHA,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_config_sha256": CONFIG_SHA, "source_index_sha256": INDEX_SHA,
        "scope": "Real full projection fixtures; native decode/GEMM correctness pending GPU gate"}
    for case in cases:
        weight = rebuild_weight(args.source, index, case)
        rows, width = weight.shape
        padded_rows = (rows + 63) // 64 * 64
        padded = torch.nn.functional.pad(weight, (0, 0, 0, padded_rows - rows))
        packed = runtime.compress(padded)
        path = args.output / (case["name"] + ".safetensors")
        save_file(dict(zip(runtime.BUFFER_NAMES, packed.buffers, strict=True)), path)
        record = {**case, "logical_shape": [rows, width], "native_shape": [padded_rows, width],
            "logical_weight_sha256": tensor_digest(weight),
            "padded_weight_sha256": tensor_digest(padded), "raw_bytes": weight.numel() * 2,
            "native_padding_rows": padded_rows - rows, "retained_bytes": packed.storage_bytes,
            "saving_fraction": 1 - packed.storage_bytes / (weight.numel() * 2),
            "native_metadata": [packed.m, packed.k, packed.max_common,
                                packed.max_rare, packed.start_exp],
            "artifact": path.name, "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        result["cases"].append(record)
        report_path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({key: record[key] for key in (
            "name", "logical_shape", "native_shape", "raw_bytes", "retained_bytes",
            "saving_fraction")}), flush=True)
        del weight, padded, packed
    result.update(status="packed_gpu_verification_pending", seconds=time.monotonic() - started,
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    report_path.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
