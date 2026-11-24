"""Screen lossless codecs on complete base-expert scales in Marlin's CPU layout.

This probe does not load weights, initialize CUDA, or measure inference latency.
It executes hash-pinned installed vLLM layout functions rather than vendoring them.
"""

from __future__ import annotations

import argparse
import ast
import json
import resource
import runpy
import time
from pathlib import Path
from typing import Any

from audit_checkpoint import DEFAULT_REPOSITORY, DEFAULT_REVISION, digest, fetch, write_json

VLLM_LAYOUT_SOURCES = {
    "model_executor/layers/quantization/utils/marlin_utils.py": (
        "e60713c15080347b01425f90eaf03840ca80a1cd226ec57e526e8f6286eeeb22",
        ("get_scale_perms", "marlin_permute_scales"),
    ),
    "model_executor/layers/quantization/utils/marlin_utils_fp4.py": (
        "7c863610b4b9c2f22e84d8690628c763bd0c81aa800b9b51e4c610a37427ee14",
        ("mxfp4_marlin_process_scales",),
    ),
}


def prepare(root: Path) -> None:
    """Download nine complete scale matrices, excluding MTP and side linears."""
    destination = root / "scale-probe-v1"
    index_raw = (root / "metadata/model.safetensors.index.json").read_bytes()
    index = json.loads(index_raw)["weight_map"]
    selection: dict[str, Any] = {
        "repository": DEFAULT_REPOSITORY,
        "revision": DEFAULT_REVISION,
        "index_sha256": digest(index_raw),
        "matrices": [],
    }
    base = f"https://huggingface.co/{DEFAULT_REPOSITORY}/resolve/{DEFAULT_REVISION}"
    for layer, expert in ((3, 0), (23, 143), (44, 287)):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            name = (
                f"model.language_model.layers.{layer}.mlp.experts.{expert}."
                f"{projection}.weight_scale"
            )
            shard = index[name]
            saved = json.loads((root / "headers" / (shard + ".json")).read_bytes())
            spec = saved["header"][name]
            shape = [4096, 64] if projection == "down_proj" else [2048, 128]
            if spec["dtype"] != "U8" or spec["shape"] != shape:
                raise ValueError("Unexpected base-expert E8M0 scale representation")
            start, stop = spec["data_offsets"]
            if stop - start != shape[0] * shape[1]:
                raise ValueError("Scale payload size disagrees with its shape")
            start += 8 + saved["header_size"]
            stop += 8 + saved["header_size"]
            path = destination / "samples" / (name + ".bin")
            sidecar = path.with_suffix(".json")
            if path.exists() and sidecar.exists():
                record = json.loads(sidecar.read_bytes())
                if record["sha256"] != digest(path.read_bytes()):
                    raise ValueError("Cached full-scale payload hash has changed")
                if record["name"] != name or record["shape"] != shape:
                    raise ValueError("Cached scale selection has changed")
            else:
                raw = fetch(base + "/" + shard, start, stop)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
                record = {
                    "name": name,
                    "layer": layer,
                    "expert": expert,
                    "projection": projection,
                    "shape": shape,
                    "dtype": "U8",
                    "shard": shard,
                    "range": [start, stop],
                    "header_sha256": saved["raw_header_sha256"],
                    "bytes": len(raw),
                    "sha256": digest(raw),
                    "path": str(path.relative_to(destination)),
                }
                write_json(sidecar, record)
            selection["matrices"].append(record)
            write_json(destination / "selection.json", selection)
            print(f"Complete scale matrices: {len(selection['matrices'])}/9", flush=True)
    write_json(
        destination / "prepare-state.json",
        {
            "status": "complete",
            "matrices": len(selection["matrices"]),
            "payload_bytes": sum(x["bytes"] for x in selection["matrices"]),
        },
    )


def measure(root: Path, codec_root: Path, vllm_root: Path) -> None:
    """Verify exact CPU records after the installed BF16-activation MoE layout."""
    import torch

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    destination = root / "scale-probe-v1"
    selection = json.loads((destination / "selection.json").read_bytes())
    if selection["revision"] != DEFAULT_REVISION:
        raise ValueError("Scale selection is not pinned to the qualified checkpoint")
    layout: dict[str, Any] = {"torch": torch}
    for relative, (sha256, names) in VLLM_LAYOUT_SOURCES.items():
        path = vllm_root / relative
        raw = path.read_bytes()
        if digest(raw) != sha256:
            raise ValueError(f"Installed vLLM layout source changed: {relative}")
        tree = ast.parse(raw, filename=str(path))
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        if {n.name for n in functions} != set(names):
            raise ValueError("Pinned vLLM layout functions are missing")
        # Only the named functions from the exact SHA-256-pinned local dependency execute.
        exec(  # noqa: S102
            compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), layout
        )
    fast_path = codec_root / "integrations/marlin_scales/records.py"
    range_path = codec_root / "mxwave/lossless.py"
    fast = runpy.run_path(str(fast_path))
    ranges = runpy.run_path(str(range_path))
    matrices: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
    for record in selection["matrices"]:
        raw = (destination / record["path"]).read_bytes()
        if digest(raw) != record["sha256"]:
            raise ValueError("Full-scale sample payload changed")
        tensor = torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(record["shape"])
        if (tensor == 255).any():
            raise ValueError("Reserved E8M0 NaN found in base-expert scales")
        key = (record["layer"], record["expert"])
        matrices.setdefault(key, {})[record["projection"]] = tensor
    report: dict[str, Any] = {
        "status": "running",
        "device": "cpu",
        "activation_path": "BF16 W4A16; no FP8 activation exponent adjustment",
        "selection": selection,
        "torch": torch.__version__,
        "script_sha256": digest(Path(__file__).read_bytes()),
        "layout_sources": {k: v[0] for k, v in VLLM_LAYOUT_SOURCES.items()},
        "codecs": {str(p): digest(p.read_bytes()) for p in (fast_path, range_path)},
        "cases": [],
        "limitations": [
            "Three deterministic base-expert samples; not all layers or experts.",
            "CPU layout/codec validation only, without any CUDA or grouped MoE kernel.",
            "No runtime latency, allocated GPU memory, PPL, KL or task score is measured.",
            "Retained bytes include exact escapes but exclude GPU descriptors and allocator rounding.",
            "MTP experts and all non-expert linears are intentionally excluded.",
        ],
    }
    for tp in (1, 2):
        for (layer, expert), projections in matrices.items():
            for rank in range(tp):
                lo, hi = rank * (2048 // tp), (rank + 1) * (2048 // tp)
                w13 = torch.cat(
                    [projections[p][lo:hi] for p in ("gate_proj", "up_proj")], dim=0
                )
                w2 = projections["down_proj"][:, lo // 32 : hi // 32]
                for kind, scales in (("w13", w13), ("w2", w2)):
                    size_n, groups = scales.shape
                    permuted_bytes = layout["marlin_permute_scales"](
                        scales.T, groups * 32, size_n, 32
                    )
                    expected = permuted_bytes.reshape(-1, 4)[:, [0, 2, 1, 3]].reshape(
                        groups, size_n
                    )
                    # Execute the actual pinned scale-conversion functions as the loader does.
                    floating = scales.view(torch.float8_e8m0fnu).to(torch.bfloat16)
                    permuted = layout["marlin_permute_scales"](
                        floating.T, groups * 32, size_n, 32
                    )
                    actual = layout["mxfp4_marlin_process_scales"](permuted).view(torch.uint8)
                    if not torch.equal(actual, expected):
                        raise ValueError("BF16 loader scale conversion changes E8M0 bits")
                    raw = actual.contiguous().numpy().tobytes()
                    records = fast["pack_records"](raw)
                    if fast["unpack_records"](records) != raw:
                        raise ValueError("Fast record decoder is not byte-exact")
                    packed_range = ranges["pack_bytes_numpy"](raw, block_size=512)
                    if ranges["unpack_bytes_numpy"](packed_range) != raw:
                        raise ValueError("Range decoder is not byte-exact")
                    escapes = len(records.exceptions)
                    report["cases"].append(
                        {
                            "tp": tp,
                            "rank": rank,
                            "layer": layer,
                            "expert": expert,
                            "kind": kind,
                            "runtime_shape": list(actual.shape),
                            "runtime_sha256": digest(raw),
                            "raw_bytes": len(raw),
                            "fast_record_bytes": records.retained_bytes,
                            "fast_escape_vectors": escapes,
                            "fast_escape_fraction": escapes / len(records.records),
                            "range_storage_bytes": packed_range.storage_bytes,
                            "byte_exact": True,
                        }
                    )
    report["totals_by_tp"] = {}
    for tp in (1, 2):
        cases = [c for c in report["cases"] if c["tp"] == tp]
        totals = {
            key: sum(c[key] for c in cases)
            for key in ("raw_bytes", "fast_record_bytes", "range_storage_bytes")
        }
        for key in ("fast_record_bytes", "range_storage_bytes"):
            totals[key + "_saving_fraction"] = 1 - totals[key] / totals["raw_bytes"]
        report["totals_by_tp"][str(tp)] = totals
    report.update(
        status="complete",
        seconds=time.monotonic() - started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    )
    write_json(destination / "comparison.json", report)
    print(json.dumps({"status": "complete", "totals_by_tp": report["totals_by_tp"]}))


def main() -> None:
    """Prepare bounded public scale payloads or execute the CPU-only codec check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "measure"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--codec-root", type=Path)
    parser.add_argument("--vllm-root", type=Path)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.root)
    else:
        if args.codec_root is None or args.vllm_root is None:
            parser.error("measure requires --codec-root and --vllm-root")
        measure(args.root, args.codec_root, args.vllm_root)


if __name__ == "__main__":
    main()
