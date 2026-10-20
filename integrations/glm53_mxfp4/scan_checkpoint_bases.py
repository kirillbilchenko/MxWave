"""Audit every current GLM compressed BF16 target without CUDA or full-model allocation."""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import os
import resource
import struct
import time
from pathlib import Path
from typing import Any

import torch
from kda_policy import KDA_PARTS, load_policy
from prefill_pack_oracle import synthetic_matrices
from prepare_bf16_attention import LIBRARY_SHA, RUNTIME_SHA, rebuild_weight, tensor_digest
from zipserv_contract import require_qualified_exponent_base

SYMBOL = "_Z32analyzeExponentDistribution_BF16P13__nv_bfloat16iiPii"


def file_digest(path: Path) -> str:
    """Hash source bindings and artifacts with at most sixteen MiB of temporary bytes."""
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def source_case(layer: int, kind: str, rank: int, config: dict[str, Any]) -> dict[str, Any]:
    """Build the same explicit TP slices as the configuration-bound KDA loader."""
    text = config["text_config"]
    linear = text["linear_attn_config"]
    heads, dim = linear["num_heads"], linear["head_dim"]
    if kind == "output":
        return {"source_tensors": [{"name": f"model.language_model.layers.{layer}.self_attn."
            "o_proj.weight", "axis": 1, "range": [rank * heads * dim // 2,
                                                     (rank + 1) * heads * dim // 2]}]}
    parts = []
    for projection in KDA_PARTS:
        rows = heads * dim // 2 if projection in ("q_proj", "k_proj", "v_proj") else (
            heads // 2 if projection == "b_proj" else dim)
        first = rank * rows if projection in ("q_proj", "k_proj", "v_proj", "b_proj") else 0
        parts.append({"name": f"model.language_model.layers.{layer}.self_attn.{projection}.weight",
                      "axis": 0, "range": [first, first + rows]})
    return {"source_tensors": parts}


def embedding_bases(path: Path, record: dict[str, Any]) -> dict[str, Any]:
    """Read and hash only the base array; this codec stores the first exponent, not base minus one."""
    header = struct.Struct("<8sIQIIQ32s")
    entry = struct.Struct("<Q32s")
    with path.open("rb") as handle:
        magic, version, rows, width, block, _, source = header.unpack(handle.read(header.size))
        assert magic == b"MXBEMB01" and version == 1
        assert [rows, width, block] == record["metadata"]
        assert source.hex() == record["source_sha256"]
        entries = [entry.unpack(handle.read(entry.size)) for _ in range(6)]
        assert entries[0][0] == rows
        bases = handle.read(rows)
        assert len(bases) == rows and hashlib.sha256(bases).digest() == entries[0][1]
    return {"kind": "embedding", "rank": record["rank"], "rows": rows,
            "codec": "row_addressable_bf16", "base_minimum": min(bases),
            "base_maximum": max(bases), "unsafe_rows": sum(base > 249 for base in bases),
            "base_sha256": hashlib.sha256(bases).hexdigest(),
            "convention": "Exponent = base + code - 1; base 249 is safe for codes 1..7."}


def main() -> None:
    """Enumerate both TP ranks, verify writer-equivalent selection and refuse unsafe bases."""
    from mxwave_zipserv import runtime

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--loaded-records", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("Preserve the first checkpoint-wide base audit")
    source = args.root / "source"
    targets = load_policy(source)
    config = json.loads((source / "config.json").read_bytes())
    index = json.loads((source / "model.safetensors.index.json").read_bytes())["weight_map"]
    loaded = json.loads(args.loaded_records.read_bytes())
    loaded_rows = {(row["rank"], row["prefix"]): row for row in loaded["records"]}
    assert len(loaded_rows) == 136
    assert file_digest(Path(os.environ["MXWAVE_ZIPSERV_LIBRARY"])) == LIBRARY_SHA
    assert file_digest(Path(runtime.__file__)) == RUNTIME_SHA
    torch.set_num_threads(1)
    torch.manual_seed(9417)
    function = getattr(runtime._library(), SYMBOL)
    function.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                         ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    function.restype = None

    def base_for(weight: torch.Tensor) -> tuple[int, list[int]]:
        assert weight.device.type == "cpu" and weight.dtype == torch.bfloat16
        assert weight.ndim == 2 and weight.is_contiguous() and weight.numel() < 2**31
        exponents = (ctypes.c_int * 7)()
        function(weight.data_ptr(), *weight.shape, exponents, 7)
        values = list(exponents)
        assert values == list(range(values[0], values[0] + 7))
        return (values[0] - 1) & 255, values

    started = time.time()
    report: dict[str, Any] = {"status": "running", "cases": [], "embedding_cases": [],
        "cuda_initialized": False, "full_model_loaded": False, "started_epoch": started,
        "code_sha256": file_digest(Path(__file__)), "library_sha256": LIBRARY_SHA,
        "runtime_sha256": RUNTIME_SHA, "analyzer_symbol": SYMBOL,
        "loaded_records_sha256": file_digest(args.loaded_records),
        "loaded_records_source_sha256": loaded["candidate_speed_sha256"],
        "selection": "Call the unchanged qualified packer's exported CPU analyzer with top_n=7.",
        "scope": "All 136 KDA TP shards, both prepared head shards and both embedding base arrays.",
        "other_codecs": "MXFP4 weights and expert-scale palettes do not use this ZipServ base field."}

    def save() -> None:
        temporary = args.report.with_suffix(".incomplete")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.report)

    save()
    try:
        underflow = ((torch.arange(16384, dtype=torch.int64) * 37) & 65535).to(
            torch.uint16).view(torch.bfloat16).reshape(128, 128)
        controls = [underflow, *synthetic_matrices()]
        report["analyzer_controls"] = []
        for cpu in controls:
            actual, values = base_for(cpu)
            packed = runtime.compress(cpu)
            assert actual == packed.start_exp
            report["analyzer_controls"].append({"shape": list(cpu.shape), "base": actual,
                                                "writer_matches": True, "top_exponents": values})
        del controls, cpu, packed, underflow
        fixture = json.loads((args.root / "bf16-attention-v1/artifacts-v2/prepared.json").read_bytes())
        fixtures = {row["name"]: row for row in fixture["cases"]}
        for rank in (0, 1):
            for target in targets.values():
                case = source_case(target.layer, target.kind, rank, config)
                cpu = rebuild_weight(source, index, case)
                assert tuple(cpu.shape) == (target.rows, target.width)
                source_sha = tensor_digest(cpu)
                old_loaded = loaded_rows[(rank, target.prefix)]
                assert source_sha == old_loaded["loaded_weight_sha256"]
                assert list(cpu.shape) == old_loaded["logical_shape"]
                padded = torch.nn.functional.pad(cpu, (0, 0, 0, target.native_rows - target.rows))
                base, values = base_for(padded)
                name = f"kda-{target.kind}-l{target.layer}-r{rank}"
                if name in fixtures:
                    old = fixtures[name]
                    assert source_sha == old["logical_weight_sha256"]
                    assert tensor_digest(padded) == old["padded_weight_sha256"]
                    assert base == old["native_metadata"][4]
                report["cases"].append({"name": name, "rank": rank, "layer": target.layer,
                    "kind": target.kind, "prefix": target.prefix, "source_tensors": case["source_tensors"],
                    "shape": list(cpu.shape), "padded_shape": list(padded.shape),
                    "source_sha256": source_sha, "base": base, "top_exponents": values,
                    "matches_actual_loaded_source": True,
                    "unsafe": base >= 249})
                del cpu, padded
                gc.collect()
                save()
        vocab = args.root / "vocab-v2"
        manifest = json.loads((vocab / "prepared.json").read_bytes())
        assert manifest["status"] == "packed_gpu_verification_pending" and len(manifest["cases"]) == 4
        report["vocabulary_manifest_sha256"] = file_digest(vocab / "prepared.json")
        for row in manifest["cases"]:
            path = vocab / row["artifact"]
            assert path.name == row["artifact"] and file_digest(path) == row["artifact_sha256"]
            if row["kind"] == "head":
                base = row["metadata"][4]
                report["cases"].append({"name": f"head-r{row['rank']}", "rank": row["rank"],
                    "kind": "head", "prefix": row["prefix"], "shape": row["shape"],
                    "source_sha256": row["source_sha256"], "base": base, "unsafe": base >= 249,
                    "artifact_sha256": row["artifact_sha256"], "native_metadata": row["metadata"]})
            else:
                report["embedding_cases"].append(embedding_bases(path, row))
        assert len(report["cases"]) == 138 and len(report["embedding_cases"]) == 2
        assert not torch.cuda.is_initialized()
        unsafe = [row["name"] for row in report["cases"] if row["unsafe"]]
        if not unsafe:
            for row in report["cases"]:
                require_qualified_exponent_base(row["base"], where=row["name"])
        report.update(status="complete", unsafe_zipserv_targets=unsafe,
            unsafe_embedding_rows=sum(row["unsafe_rows"] for row in report["embedding_cases"]),
            cuda_initialized=False, peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * 1024, seconds=time.time() - started, finished_epoch=time.time())
    except BaseException as exc:
        report.update(status="failed", error=repr(exc), finished_epoch=time.time())
        raise
    finally:
        save()
    print(json.dumps({key: report[key] for key in ("status", "unsafe_zipserv_targets",
        "unsafe_embedding_rows", "cuda_initialized", "peak_host_rss_bytes", "seconds")}))


if __name__ == "__main__":
    main()
