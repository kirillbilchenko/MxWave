"""Download a bounded, resumable full-shape base-expert fixture from pinned weights."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
from typing import Any

from audit_checkpoint import DEFAULT_REPOSITORY, DEFAULT_REVISION, digest, fetch, write_json

EXPERT_IDS = (0, 1, 2, 3, 47, 63, 79, 95, 111, 127, 143, 159, 191, 223, 255, 287)
INDEX_SHA256 = "6d1364a27eef78675390422b1b5230601817fb7460ce816951b2cd515934c85d"


def prepare(root: Path, *, full_layer: bool = False) -> None:
    """Fetch the requested layer-23 expert set, keeping exact codes and scale bytes."""
    index_raw = (root / "metadata/model.safetensors.index.json").read_bytes()
    if digest(index_raw) != INDEX_SHA256:
        raise ValueError("Expert fixture requires the audited checkpoint index")
    index = json.loads(index_raw)["weight_map"]
    config_raw = (root / "metadata/config.json").read_bytes()
    config = json.loads(config_raw)["quantization_config"]
    if config["quant_method"] != "compressed-tensors":
        raise ValueError("Fixture requires the audited compressed-tensors checkpoint")
    if config["config_groups"]["group_0"]["format"] != "mxfp4-pack-quantized":
        raise ValueError("Expert fixture source is not config-declared packed MXFP4")
    experts = tuple(range(288)) if full_layer else EXPERT_IDS
    namespace = "grouped-moe-full-layer-v1" if full_layer else "grouped-moe-v1"
    directory = root / namespace / "fixture"
    directory.mkdir(parents=True, exist_ok=True)
    tensors: list[dict[str, Any]] = []

    def prepare_expert(expert: int) -> list[dict[str, Any]]:
        records = []
        for projection in ("gate_proj", "up_proj", "down_proj"):
            n, k = (4096, 2048) if projection == "down_proj" else (2048, 4096)
            for suffix, shape in (("weight_packed", [n, k // 2]), ("weight_scale", [n, k // 32])):
                name = f"model.language_model.layers.23.mlp.experts.{expert}.{projection}.{suffix}"
                shard = index[name]
                saved = json.loads((root / "headers" / (shard + ".json")).read_bytes())
                spec = saved["header"][name]
                start, end = spec["data_offsets"]
                if spec["dtype"] != "U8" or spec["shape"] != shape or end - start != n * shape[1]:
                    raise ValueError("Expert fixture tensor does not match audited MXFP4 shapes")
                start += 8 + saved["header_size"]
                end += 8 + saved["header_size"]
                path = directory / (name + ".bin")
                sidecar = path.with_suffix(".json")
                if path.exists() and sidecar.exists():
                    record = json.loads(sidecar.read_bytes())
                    if (digest(path.read_bytes()) != record["sha256"]
                            or record["name"] != name or record["shape"] != shape):
                        raise ValueError("Cached expert fixture payload or selection changed")
                else:
                    url = f"https://huggingface.co/{DEFAULT_REPOSITORY}/resolve/{DEFAULT_REVISION}"
                    raw = fetch(url + "/" + shard, start, end)
                    temporary = path.with_suffix(".incomplete")
                    temporary.write_bytes(raw)
                    temporary.replace(path)
                    record = {
                        "name": name, "expert": expert, "projection": projection,
                        "suffix": suffix, "shape": shape, "dtype": "U8", "shard": shard,
                        "bytes": len(raw), "sha256": digest(raw), "path": path.name,
                        "header_sha256": saved["raw_header_sha256"], "range": [start, end],
                    }
                    write_json(sidecar, record)
                records.append(record)
        return records

    # Independent HTTP ranges run concurrently; only this coordinator writes progress/manifest.
    with concurrent.futures.ThreadPoolExecutor(max_workers=4 if full_layer else 1) as pool:
        for position, ready in enumerate(pool.map(prepare_expert, experts)):
            tensors.extend(ready)
            write_json(directory / "state.json", {"status": "preparing", "tensors": len(tensors)})
            print(f"Complete expert fixture: {position + 1}/{len(experts)}", flush=True)
    write_json(directory / "manifest.json", {
        "repository": DEFAULT_REPOSITORY, "revision": DEFAULT_REVISION,
        "config_sha256": digest(config_raw), "index_sha256": INDEX_SHA256,
        "layer": 23, "expert_ids": experts, "tensors": tensors,
        "hidden_size": 4096, "intermediate_size": 2048, "top_k": 8,
        "payload_bytes": sum(t["bytes"] for t in tensors),
        "scope": ("Complete 288-expert layer; not a whole model" if full_layer else
                  "Reduced 16-expert component; not a full layer or whole model"),
    })
    write_json(directory / "state.json", {
        "status": "complete", "tensors": len(tensors),
        "payload_bytes": sum(t["bytes"] for t in tensors),
    })


def main() -> None:
    """Prepare only the fixed base-expert component, without initializing CUDA."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--full-layer", action="store_true")
    args = parser.parse_args()
    try:
        prepare(args.root, full_layer=args.full_layer)
    except BaseException as error:
        namespace = "grouped-moe-full-layer-v1" if args.full_layer else "grouped-moe-v1"
        write_json(args.root / namespace / "fixture/state.json", {
            "status": "failed", "error": str(error),
        })
        raise


if __name__ == "__main__":
    main()
