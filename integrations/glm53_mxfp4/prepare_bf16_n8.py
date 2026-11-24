"""Prepare source-bound N8 fixtures for both TP shards at KDA layers 0, 22 and 44."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
from pathlib import Path
from typing import Any

import torch
from benchmark_bf16_kernels import digest, fixtures
from kda_policy import CONFIG_SHA, INDEX_SHA
from prepare_bf16_attention import LIBRARY_SHA, RUNTIME_SHA, rebuild_weight, tensor_digest
from safetensors.torch import save_file


def main() -> None:
    """Pack only twelve small KDA projections on CPU, reusing both existing head artifacts."""
    from mxwave_zipserv import runtime

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/audit"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    if (digest(Path(os.environ["MXWAVE_ZIPSERV_LIBRARY"])) != LIBRARY_SHA
            or digest(Path(runtime.__file__)) != RUNTIME_SHA):
        raise ValueError("Require the qualified CPU packer/ABI")
    source = args.root / "source"
    for name, expected in (("config.json", CONFIG_SHA), ("model.safetensors.index.json", INDEX_SHA)):
        if digest(source / name) != expected:
            raise ValueError("Pinned source metadata changed")
    args.output.mkdir()
    path = args.output / "prepared.json"
    torch.set_num_threads(2)
    index = json.loads((source / "model.safetensors.index.json").read_bytes())["weight_map"]
    originals = fixtures(args.root)
    cases = []
    report: dict[str, Any] = {"status": "packing", "cases": cases,
        "source_metadata": {"config.json": CONFIG_SHA, "model.safetensors.index.json": INDEX_SHA},
        "library_sha256": LIBRARY_SHA, "runtime_sha256": RUNTIME_SHA,
        "code_sha256": digest(Path(__file__)), "device": "cpu"}
    for rank in (0, 1):
        for layer in (0, 22, 44):
            for base in originals[:2]:
                case = copy.deepcopy(base)
                kind = "input" if case["name"].startswith("kda-input") else "output"
                case["name"] = f"kda-{kind}-l{layer}-r{rank}"
                case["rank"], case["layer"], case["kind"] = rank, layer, kind
                for i, part in enumerate(case["source_tensors"]):
                    part["name"] = part["name"].replace(".layers.0.", f".layers.{layer}.")
                    if kind == "output" or i < 4:
                        first, last = part["range"]
                        part["range"] = [first + rank * (last - first), last + rank * (last - first)]
                cpu = rebuild_weight(source, index, case)
                case["logical_weight_sha256"] = tensor_digest(cpu)
                padded = torch.nn.functional.pad(cpu, (0, 0, 0,
                                                        case["native_shape"][0] - cpu.shape[0]))
                packed = runtime.compress(padded)
                artifact = case["name"] + ".safetensors"
                save_file(dict(zip(runtime.BUFFER_NAMES, packed.buffers, strict=True)),
                          args.output / artifact)
                case.update(padded_weight_sha256=tensor_digest(padded),
                    artifact_path=str(args.artifact_root / artifact),
                    artifact_sha256=digest(args.output / artifact),
                    native_metadata=[packed.m, packed.k, packed.max_common,
                                     packed.max_rare, packed.start_exp],
                    retained_bytes=packed.storage_bytes,
                    saving_fraction=1 - packed.storage_bytes / case["raw_bytes"])
                case.pop("directory", None)
                cases.append(case)
                path.write_text(json.dumps(report, indent=2) + "\n")
                del cpu, padded, packed
                gc.collect()
    heads = json.loads((args.root / "vocab-v2/prepared.json").read_bytes())
    for head in heads["cases"]:
        if head["kind"] != "head":
            continue
        cases.append({**head, "name": f"head-r{head['rank']}", "layer": None,
            "logical_shape": head["shape"], "native_shape": head["shape"],
            "logical_weight_sha256": head["source_sha256"],
            "padded_weight_sha256": head["source_sha256"],
            "native_metadata": head["metadata"],
            "artifact_path": str(args.root / "vocab-v2" / head["artifact"]),
            "source_tensors": [{"name": head["source_key"], "axis": 0,
                                "range": [head["rank"] * head["shape"][0],
                                          (head["rank"] + 1) * head["shape"][0]]}]})
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU preparation unexpectedly initialized CUDA")
    report["status"] = "prepared_pending_gpu_qualification"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "cases": len(cases)}))


if __name__ == "__main__":
    main()
