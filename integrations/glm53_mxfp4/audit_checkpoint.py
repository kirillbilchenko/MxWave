"""Audit a pinned GLM mixed MXFP4/MXFP8 checkpoint using headers only.

No tensor payloads, CUDA libraries, or running model processes are accessed.
The generic Linear selector is checked against physically quantized matrices;
headers alone cannot prove which module class an inference engine constructs.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import json
import math
import os
import re
import struct
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_REPOSITORY = "INCModel3/GLM-5.3-Flash-MXFP4-Mixed-CT-AutoRound"
DEFAULT_REVISION = "35d1b0d5cb4a0db075fd2749c897e970dc15bfa5"
METADATA_LIMIT = 32 * 1024**2
DTYPE_BYTES = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "F64": 8,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "BOOL": 1,
    "U16": 2,
    "U32": 4,
    "U64": 8,
}
EXPERT = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)$"
)


def digest(data: bytes) -> str:
    """Hash exact metadata bytes for reproducibility."""
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: object) -> None:
    """Replace a private audit record atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def fetch(url: str, start: int | None = None, stop: int | None = None) -> bytes:
    """Read bounded metadata and fail if a server ignores an exact byte range."""
    headers = {"User-Agent": "MxWave-GLM-header-audit", "Accept-Encoding": "identity"}
    limit = METADATA_LIMIT
    if start is not None:
        if stop is None or start < 0 or stop <= start:
            raise ValueError("Invalid exclusive byte range")
        limit = stop - start
        if limit > METADATA_LIMIT:
            raise ValueError("Requested metadata exceeds the read bound")
        headers["Range"] = f"bytes={start}-{stop - 1}"
        url += f"?mxwave_range={start}-{stop - 1}"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, headers=headers), timeout=45
            ) as response:
                if start is not None:
                    expected = f"bytes {start}-{stop - 1}/"  # type: ignore[operator]
                    if response.status != 206 or not response.headers.get(
                        "Content-Range", ""
                    ).startswith(expected):
                        raise ValueError("Server did not honor the exact metadata range")
                data = response.read(limit + 1)
                if len(data) > limit or (start is not None and len(data) != limit):
                    raise ValueError("Remote metadata exceeds its bound or is truncated")
                return data
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 3:
                raise RuntimeError(f"Metadata HTTP status {exc.code}") from None
        except (OSError, TimeoutError):
            if attempt == 3:
                raise RuntimeError("Metadata request timed out or connection failed") from None
        time.sleep(1 + attempt)
    raise AssertionError("Unreachable retry state")


def validate_header(
    header: dict[str, Any],
    header_size: int,
    file_size: int,
) -> dict[str, dict[str, Any]]:
    """Validate tensor shapes, byte intervals and complete shard payload accounting."""
    specs: dict[str, dict[str, Any]] = {}
    intervals: list[tuple[int, int]] = []
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        dtype = spec["dtype"]
        shape, offsets = spec["shape"], spec["data_offsets"]
        if dtype not in DTYPE_BYTES or any(type(x) is not int or x < 0 for x in shape):
            raise ValueError(f"Invalid tensor dtype or shape for {name}")
        start, stop = offsets
        if (
            type(start) is not int
            or type(stop) is not int
            or start < 0
            or stop < start
            or stop - start != math.prod(shape) * DTYPE_BYTES[dtype]
        ):
            raise ValueError(f"Tensor byte interval disagrees with its shape: {name}")
        intervals.append((start, stop))
        specs[name] = spec
    cursor = 0
    for start, stop in sorted(intervals):
        if start != cursor:
            raise ValueError("Shard tensor payloads overlap or contain unaccounted gaps")
        cursor = stop
    if 8 + header_size + cursor != file_size:
        raise ValueError("Shard header and tensor intervals do not account for its file size")
    return specs


def classify_family(module: str) -> str:
    """Name architecture families independently of their observed storage precision."""
    if ".layers.45." in module:
        return "mtp/" + classify_family(module.replace(".layers.45.", ".layers.3."))
    if ".mlp.experts." in module:
        return "routed_experts"
    if ".mlp.shared_experts." in module:
        return "shared_experts"
    if re.search(r"\.layers\.[012]\.mlp\.(gate_proj|up_proj|down_proj)$", module):
        return "leading_dense_mlp"
    if ".self_attn.indexer." in module:
        return "attention_indexer"
    if ".self_attn." in module:
        return "attention"
    if module.startswith("model.visual."):
        return "vision"
    if module.endswith(".mlp.gate"):
        return "routers"
    if module == "lm_head":
        return "output_head"
    if module.endswith(".embed_tokens"):
        return "embedding"
    if "norm" in module:
        return "norms"
    return "other"


def analyze(
    config: dict[str, Any],
    weight_map: dict[str, str],
    shards: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Check every expert, scale and weighted module in the complete indexed checkpoint."""
    quant = config["quantization_config"]
    if quant.get("quant_method") != "compressed-tensors":
        raise ValueError("This audit requires declared compressed-tensors metadata")
    groups = quant["config_groups"]
    four = [g for g in groups.values() if g.get("format") == "mxfp4-pack-quantized"]
    eight = [g for g in groups.values() if g.get("format") == "mxfp8-quantized"]
    if len(four) != 1 or len(eight) != 1 or len(groups) != 2:
        raise ValueError("Expected exactly one MXFP4 and one MXFP8 group")
    for bits, group in ((4, four[0]), (8, eight[0])):
        weight = group["weights"]
        if (
            weight.get("num_bits") != bits
            or weight.get("group_size") != 32
            or weight.get("strategy") != "group"
            or weight.get("type") != "float"
            or weight.get("scale_dtype") != "torch.uint8"
            or weight.get("dynamic") is not False
        ):
            raise ValueError("Declared weight scheme is not canonical static MX block scaling")
    if eight[0].get("targets") != ["Linear"]:
        raise ValueError("This audit expects the MXFP8 fallback group to target Linear")
    targets = four[0]["targets"]
    if len(targets) != len(set(targets)):
        raise ValueError("Explicit expert selectors are duplicated")
    text = config["text_config"]
    if (
        config.get("model_type") != "glm5_next"
        or config.get("architectures") != ["Glm5NextForConditionalGeneration"]
        or text.get("model_type") != "glm5_next_text"
    ):
        raise ValueError("The checkpoint does not declare the expected GLM architecture")
    layers, experts, first = (
        text[k] for k in ("num_hidden_layers", "n_routed_experts", "first_k_dense_replace")
    )
    mtp = text["num_nextn_predict_layers"]
    hidden, intermediate = text["hidden_size"], text["moe_intermediate_size"]
    expected = {
        f"model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}"
        for layer in range(first, layers + mtp)
        for expert in range(experts)
        for projection in ("gate_proj", "up_proj", "down_proj")
    }
    if set(targets) != expected:
        raise ValueError("Declared expert target coverage differs from the architecture")
    ignore = set(quant.get("ignore", []))
    if ignore.intersection(expected):
        raise ValueError("An explicit MXFP4 expert is also ignored")
    tensors: dict[str, dict[str, Any]] = {}
    for shard, values in shards.items():
        for name, spec in values.items():
            if name in tensors or weight_map.get(name) != shard:
                raise ValueError("Duplicate, unindexed, or incorrectly owned checkpoint tensor")
            tensors[name] = spec
    if set(tensors) != set(weight_map):
        raise ValueError("Checkpoint headers and index have different tensor coverage")
    weighted: dict[str, tuple[str, dict[str, Any]]] = {}
    for name, spec in tensors.items():
        if name.endswith(".weight_packed"):
            module = name.removesuffix(".weight_packed")
        elif name.endswith(".weight"):
            module = name.removesuffix(".weight")
        else:
            continue
        if module in weighted:
            raise ValueError("A module has both packed and unpacked weight tensors")
        weighted[module] = name, spec
    groups_by_family: dict[str, dict[str, Any]] = {}
    matrix_rows: list[dict[str, Any]] = []
    owned_scales: set[str] = set()
    physical_experts: set[str] = set()
    mx8_modules: list[str] = []
    for module, (name, spec) in sorted(weighted.items()):
        shape = spec["shape"]
        scale_name = module + ".weight_scale"
        scale = tensors.get(scale_name)
        size = spec["data_offsets"][1] - spec["data_offsets"][0]
        scale_size = 0
        if module in expected:
            match = EXPERT.fullmatch(module)
            if match is None:
                raise ValueError("Explicit target is not a GLM expert projection")
            columns = intermediate if match.group(3) == "down_proj" else hidden
            rows = hidden if match.group(3) == "down_proj" else intermediate
            if (
                name != module + ".weight_packed"
                or spec["dtype"] != "U8"
                or shape != [rows, columns // 2]
            ):
                raise ValueError(
                    f"Expert packing differs from its declared logical shape: {module}"
                )
            kind = "MXFP4"
            physical_experts.add(module)
        elif spec["dtype"] == "F8_E4M3":
            if module in ignore or len(shape) != 2:
                raise ValueError(f"Ignored or non-matrix module carries FP8 weights: {module}")
            rows, columns = shape
            kind = "MXFP8"
            mx8_modules.append(module)
        else:
            if name.endswith(".weight_packed") or spec["dtype"] == "U8":
                raise ValueError(
                    f"Unexpected packed module outside the declared MXFP4 targets: {module}"
                )
            if len(shape) == 2 and module not in ignore:
                raise ValueError(
                    f"Unclassified floating matrix outside quantized/ignored groups: {module}"
                )
            if scale is not None:
                raise ValueError(f"Unexpected quantization scale for passthrough module: {module}")
            kind = spec["dtype"]
            rows = columns = 0
        if kind in ("MXFP4", "MXFP8"):
            if (
                columns % 32
                or scale is None
                or scale["dtype"] != "U8"
                or scale["shape"] != [rows, columns // 32]
            ):
                raise ValueError(f"Missing, mistyped or incorrectly shaped E8M0 scales: {module}")
            scale_size = scale["data_offsets"][1] - scale["data_offsets"][0]
            owned_scales.add(scale_name)
        family = classify_family(module)
        key = family + "/" + kind
        summary = groups_by_family.setdefault(
            key,
            {
                "modules": 0,
                "weight_bytes": 0,
                "scale_bytes": 0,
            },
        )
        summary["modules"] += 1
        summary["weight_bytes"] += size
        summary["scale_bytes"] += scale_size
        matrix_rows.append(
            {
                "module": module,
                "family": family,
                "format": kind,
                "shape": shape,
                "weight_bytes": size,
                "scale_bytes": scale_size,
            }
        )
    if physical_experts != expected:
        raise ValueError("Declared expert targets lack physical weight/scale coverage")
    all_scales = {name for name in tensors if name.endswith(".weight_scale")}
    if all_scales != owned_scales:
        raise ValueError("Some checkpoint scales have no classified quantized owner")
    base_experts = {m for m in expected if int(EXPERT.fullmatch(m).group(1)) < layers}
    dtype_bytes: dict[str, int] = collections.defaultdict(int)
    for spec in tensors.values():
        dtype_bytes[spec["dtype"]] += spec["data_offsets"][1] - spec["data_offsets"][0]
    return {
        "status": "complete",
        "scope": "All indexed tensor headers; no weight payloads or inference",
        "shards": len(shards),
        "tensors": len(tensors),
        "indexed_payload_bytes_recomputed": sum(dtype_bytes.values()),
        "payload_bytes_by_dtype": dict(dtype_bytes),
        "declared_expert_targets": len(expected),
        "base_expert_matrices": len(base_experts),
        "mtp_expert_matrices": len(expected - base_experts),
        "base_expert_layers": list(range(first, layers)),
        "experts_per_layer": experts,
        "other_mxfp8_matrices": len(mx8_modules),
        "checks": {
            "index_header_coverage_exact": True,
            "expert_config_coverage_exact": True,
            "expert_weight_shapes_match_architecture": True,
            "all_quantized_scale_shapes_and_dtypes_valid": True,
            "no_orphan_scale_tensors": True,
            "no_unclassified_floating_matrices": True,
        },
        "format_groups": [
            {k: v for k, v in group.items() if k != "targets"} for group in (four[0], eight[0])
        ],
        "families": dict(sorted(groups_by_family.items())),
        "other_mxfp8_modules": mx8_modules,
        "proposed_mxwave_comparison": {
            "glm_quantization_policy_implemented": False,
            "same_base_routed_expert_selection": True,
            "additional_upstream_mtp_matrices": len(expected - base_experts),
            "upstream_other_linears": "MXFP8, including original E8M0 scales",
            "proposed_expert_only_from_bf16_other_linears": "BF16 passthrough",
            "scale_encoding_matches_mxwave_mxfp4": True,
            "scale_selection_values_compared": False,
        },
        "limitations": [
            "No numerical scale values or weight codes were read; no byte-equality claim.",
            "Correct tensor ownership and shape do not verify full tensor/shard content hashes.",
            "Generic Linear selection and fused aliases still require inference-loader validation.",
            "No runtime kernel, activation precision, quality or memory claim is established.",
        ],
        "weighted_modules": matrix_rows,
    }


def run(root: Path, repository: str, revision: str) -> None:
    """Collect resumable pinned headers and save a complete, hash-bound audit."""
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("An immutable 40-character checkpoint revision is required")
    base = f"https://huggingface.co/{repository}/resolve/{revision}"
    identity = {"repository": repository, "revision": revision}
    identity_path = root / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_bytes()) != identity:
        raise ValueError("The audit directory already belongs to a different source")
    write_json(identity_path, identity)
    started = time.monotonic()
    metadata: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    for name in ("config.json", "model.safetensors.index.json"):
        path = root / "metadata" / name
        if not path.exists():
            raw = fetch(base + "/" + name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        raw = path.read_bytes()
        metadata[name] = json.loads(raw)
        hashes[name] = digest(raw)
    api = json.loads(
        fetch(f"https://huggingface.co/api/models/{repository}/revision/{revision}?blobs=true")
    )
    file_sizes = {
        item["rfilename"]: item["size"]
        for item in api["siblings"]
        if item["rfilename"].endswith(".safetensors")
    }
    index = metadata["model.safetensors.index.json"]
    weight_map = index["weight_map"]
    names = sorted(set(weight_map.values()))
    if any(Path(name).name != name or not name.endswith(".safetensors") for name in names):
        raise ValueError("Unexpected checkpoint shard filename")
    all_headers: dict[str, dict[str, dict[str, Any]]] = {}
    header_records: dict[str, Any] = {}

    def load_header(name: str) -> tuple[str, dict[str, dict[str, Any]], dict[str, Any]]:
        path = root / "headers" / (name + ".json")
        if path.exists():
            saved = json.loads(path.read_bytes())
        else:
            url = base + "/" + name
            size = struct.unpack("<Q", fetch(url, 0, 8))[0]
            if not 1 <= size <= METADATA_LIMIT:
                raise ValueError("Unsupported safetensors header size")
            raw = fetch(url, 8, 8 + size)
            saved = {
                "header_size": size,
                "raw_header_sha256": digest(raw),
                "header": json.loads(raw),
            }
            write_json(path, saved)
        specs = validate_header(saved["header"], saved["header_size"], file_sizes[name])
        record = {
            "header_bytes": saved["header_size"],
            "raw_header_sha256": saved["raw_header_sha256"],
            "cached_header_sha256": digest(path.read_bytes()),
            "file_bytes": file_sizes[name],
            "tensors": len(specs),
        }
        return name, specs, record

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(load_header, name) for name in names]
        for future in concurrent.futures.as_completed(jobs):
            name, specs, record = future.result()
            all_headers[name] = specs
            header_records[name] = record
            write_json(
                root / "state.json",
                {
                    "status": "reading_headers",
                    "completed": len(all_headers),
                    "total": len(names),
                    "seconds": time.monotonic() - started,
                },
            )
            print(f"Headers {len(all_headers)}/{len(names)}: {name}", flush=True)
    report = analyze(metadata["config.json"], weight_map, all_headers)
    weighted_modules = report.pop("weighted_modules")
    write_json(root / "weighted-modules.json", weighted_modules)
    report["evidence"] = {
        **identity,
        "metadata_sha256": hashes,
        "headers": dict(sorted(header_records.items())),
        "weighted_modules_sha256": digest((root / "weighted-modules.json").read_bytes()),
        "script_sha256": digest(Path(__file__).read_bytes()),
        "seconds": time.monotonic() - started,
    }
    report["source_index_reported_bytes"] = index.get("metadata", {}).get("total_size")
    report["total_safetensors_file_bytes"] = sum(file_sizes[n] for n in names)
    write_json(root / "report.json", report)
    write_json(
        root / "state.json",
        {
            "status": "complete",
            "completed": len(names),
            "total": len(names),
            "seconds": time.monotonic() - started,
        },
    )


def main() -> None:
    """Audit metadata without importing PyTorch or accessing a GPU."""
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    args = parser.parse_args()
    try:
        run(args.root, args.repository, args.revision)
    except Exception as exc:
        write_json(args.root / "state.json", {"status": "failed", "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
