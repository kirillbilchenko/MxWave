"""Replay the first changed expert GEMM with stock, raw-bridge and packed scales."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import struct
import time
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
    """Hash a retained input without reading the whole file into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def weight_records(source: Path, family: str = "w13") -> dict[str, Any]:
    """Require the original CT source metadata and exact layer-four expert geometry."""
    index = source / "model.safetensors.index.json"
    assert digest(index) == "6d1364a27eef78675390422b1b5230601817fb7460ce816951b2cd515934c85d"
    assert digest(source / "config.json") == (
        "194d290ebb0f19aedc60d24ea9e236e33c86616e460f4a108a842f6fd5b8b198"
    )
    mapping = json.loads(index.read_bytes())["weight_map"]
    headers, result = {}, {}
    assert family in ("w13", "w2")
    projections = ("gate_proj", "up_proj") if family == "w13" else ("down_proj",)
    n, k = (2048, 4096) if family == "w13" else (4096, 2048)
    for expert in range(288):
        for projection in projections:
            for suffix in ("weight_packed", "weight_scale"):
                name = f"model.language_model.layers.4.mlp.experts.{expert}.{projection}.{suffix}"
                shard = mapping[name]
                if shard not in headers:
                    with (source / shard).open("rb") as stream:
                        size = struct.unpack("<Q", stream.read(8))[0]
                        raw = stream.read(size)
                    headers[shard] = (size, json.loads(raw), hashlib.sha256(raw).hexdigest())
                size, header, sha = headers[shard]
                row = header[name]
                shape = [n, k // (2 if suffix == "weight_packed" else 32)]
                assert row["dtype"] == "U8" and row["shape"] == shape
                lo, hi = row["data_offsets"]
                assert hi - lo == shape[0] * shape[1]
                result[name] = {"shard": shard, "offset": 8 + size + lo,
                                "bytes": hi - lo, "shape": shape, "header_sha256": sha}
    return result


def load_weights(
    source: Path, records: dict[str, Any], rank: int, family: str = "w13",
) -> tuple[Any, Any, Any]:
    """Apply installed stock repacking to one real TP partition, one expert at a time."""
    import torch
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_permute_scales
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        mxfp4_marlin_process_scales,
    )

    assert family in ("w13", "w2")
    projections = ("gate_proj", "up_proj") if family == "w13" else ("down_proj",)
    n, k = (2048, 4096) if family == "w13" else (4096, 1024)
    weight = torch.empty((288, k // 16, n * 2), dtype=torch.int32, device="cuda")
    scales = torch.empty((288, k // 32, n), dtype=torch.float8_e8m0fnu, device="cuda")
    perm = torch.empty(0, dtype=torch.int32, device="cuda")
    proof = []
    for expert in range(288):
        sources: dict[str, list[Any]] = {"weight_packed": [], "weight_scale": []}
        for projection in projections:
            for suffix, pieces in sources.items():
                name = f"model.language_model.layers.4.mlp.experts.{expert}.{projection}.{suffix}"
                row = records[name]
                with (source / row["shard"]).open("rb") as stream:
                    stream.seek(row["offset"])
                    raw = bytearray(stream.read(row["bytes"]))
                assert len(raw) == row["bytes"]
                proof.append({"name": name, "sha256": hashlib.sha256(raw).hexdigest(), **row})
                value = torch.frombuffer(raw, dtype=torch.uint8).reshape(row["shape"])
                if family == "w13":
                    piece = value[rank * 1024:(rank + 1) * 1024]
                else:
                    width = 512 if suffix == "weight_packed" else 32
                    piece = value[:, rank * width:(rank + 1) * width]
                pieces.append(piece.clone())
        codes = torch.cat(sources["weight_packed"]).to("cuda")
        packed = codes.view(torch.int32).T.contiguous()
        weight[expert].copy_(ops.gptq_marlin_repack(packed, perm, k, n, 4, False))
        raw_scale = torch.cat(sources["weight_scale"]).to("cuda")
        assert not bool((raw_scale == 255).any())
        floating = raw_scale.view(torch.float8_e8m0fnu).to(torch.bfloat16)
        permuted = marlin_permute_scales(floating.T, k, n, 32)
        scales[expert].copy_(mxfp4_marlin_process_scales(permuted))
    torch.cuda.synchronize()
    return weight, scales, proof


def frame(
    evidence: Path, arm: str, rank: int, position: int, projection: int = 0,
) -> tuple[Any, Any, Any]:
    """Read only operands and the retained result for the first suspect GEMM."""
    import torch
    from safetensors.torch import load_file

    number = 4 if position == 46 else 5
    path = evidence / f"{arm}-r{rank}" / (
        f"context-opencoder-evol-15654-capture1-{number:04d}-r{rank}.safetensors"
    )
    meta = json.loads(path.with_suffix(".json").read_bytes())
    assert digest(path) == meta["payload_sha256"]
    tensors = load_file(path)
    assert int(tensors["target_positions"][0]) == position
    events = {e["name"]: e for e in meta["components"]["events"]}
    assert projection in (0, 1)
    event = events[f"layer4.moe_gemm#{projection}/input"]
    operands = {}
    for leaf in event["tensors"]:
        value = tensors[leaf["payload_key"]]
        if leaf["dtype"] == "torch.bfloat16":
            value = value.view(torch.bfloat16)
        operands[leaf["path"]] = value.to("cuda").contiguous()
    leaf = events[f"layer4.moe_gemm#{projection}/output"]["tensors"][0]
    reference = tensors[leaf["payload_key"]].view(torch.bfloat16).clone()
    return operands, reference, event["parameters"]


def bits_equal(a: Any, b: Any) -> bool:
    """Compare raw floating point payload bits."""
    import torch

    return bool(torch.equal(a.view(torch.uint8), b.view(torch.uint8)))


def compare(a: Any, b: Any, shared: int) -> dict[str, Any]:
    """Compare token-major expert contributions without treating routes as tokens."""
    import torch

    x, y = a.reshape(8, 8, -1), b.reshape(8, 8, -1)
    changed = x.view(torch.int16) != y.view(torch.int16)
    return {"all_bits_exact": bits_equal(a, b),
            "shared_rows": shared, "shared_bits_exact": bits_equal(x[:shared], y[:shared]),
            "different_elements_per_row": changed.flatten(1).sum(1).tolist(),
            "shared_max_abs": float((x[:shared].float() - y[:shared].float()).abs().max())}


def run(root: Path, source: Path, evidence: Path, frozen: Path) -> dict[str, Any]:
    """Retain native matches, twenty interleaved graph repeats and stock comparisons."""
    import torch
    from moe_runtime import NativeMoe, allocate_buffers, pack_scales
    from safetensors.torch import save_file
    from vllm import _custom_ops as ops
    from vllm.scalar_type import scalar_types

    assert not (root / "report.json").exists()
    plan = json.loads((root / "plan.json").read_bytes())
    for relative, expected in plan["frozen_bindings"].items():
        assert digest(frozen / relative) == expected, relative
    records = weight_records(source)
    raw = NativeMoe(frozen / plan["raw_library"])
    packed = NativeMoe(frozen / plan["packed_library"])
    assert not raw.compressed and packed.encoding == "sparse_palette" and packed.shared_stage
    result: dict[str, Any] = {
        "status": "running", "started_epoch": time.time(), "ranks": [],
        "source_sha256": digest(Path(__file__)), "plan_sha256": digest(root / "plan.json"),
        "torch": str(torch.__version__), "cuda": str(torch.version.cuda),
        "scope": "Layer 4 gate/up only; each TP partition replayed on one GPU; no speed claim.",
    }
    outputs = {}
    for rank in (0, 1):
        weight, scales, proof = load_weights(source, records, rank)
        (root / f"weight-sources-r{rank}.json").write_text(json.dumps(proof, indent=2) + "\n")
        compressed = pack_scales(scales, Path("/unused"), encoding="sparse_palette")
        assert bits_equal(packed.decode(compressed), scales.view(torch.uint8))
        tile = raw.plan(8, 2048, 4096, 8, 8, 288, 0)
        row: dict[str, Any] = {"rank": rank, "tile": list(tile), "cases": [],
                               "scale_bytes_exact": True, "comparisons": []}
        for position in (46, 54):
            calls, owners = {}, []
            for arm in ("a1", "b1"):
                values, reference, parameters = frame(evidence, arm, rank, position)
                assert parameters == plan["gemm_parameters"]
                x, route_weights = values["/0"], values["/14"]
                routing = tuple(values[f"/{i}"] for i in (11, 12, 13))
                outputs[f"r{rank}_{position}_{arm}_retained"] = reference
                for mode in ("stock", "raw", "packed"):
                    buffers = allocate_buffers(8, 2048, 8, 8, len(routing[0]), x.device)
                    if mode == "stock":
                        kwargs = parameters | {"b_q_type": scalar_types.float4_e2m1f}

                        def operation(
                            x: Any = x, buffers: Any = buffers, routing: Any = routing,
                            route_weights: Any = route_weights, kwargs: Any = kwargs,
                            weight: Any = weight, scales: Any = scales,
                        ) -> Any:
                            return ops.moe_wna16_marlin_gemm(
                                x, buffers.output, weight, None, scales, None, None, None,
                                None, None, buffers.locks, *routing, route_weights, **kwargs,
                            )
                    else:
                        native = raw if mode == "raw" else packed
                        operation = native.bind(
                            x, weight, scales if mode == "raw" else compressed,
                            buffers, routing, route_weights, 8, 8, False, tile,
                        )
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            operation()
                    stream.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        output = operation()
                    torch.cuda.current_stream().wait_stream(stream)
                    graph.replay()
                    torch.cuda.synchronize()
                    first = output.cpu().clone()
                    calls[arm, mode] = (graph, output, first)
                    owners.append((operation, buffers, values, stream))
                    name = f"r{rank}_{position}_{arm}_{mode}"
                    outputs[name] = first
                    row["cases"].append({
                        "position": position, "arm": arm, "mode": mode,
                        "padded_tokens": int(routing[2].item()),
                        "retained_match": compare(first, reference, 8),
                    })
            repeats = []
            for iteration in range(20):
                order = list(calls) if iteration % 2 == 0 else list(reversed(calls))
                for key in order:
                    graph, value, expected = calls[key]
                    graph.replay()
                    torch.cuda.synchronize()
                    repeats.append(bits_equal(value.cpu(), expected))
            assert all(repeats), "Exact-input component graph replay is unstable"
            for mode in ("stock", "raw", "packed"):
                row["comparisons"].append({
                    "position": position, "mode": mode,
                    **compare(calls["a1", mode][2], calls["b1", mode][2],
                              8 if position == 46 else 7),
                })
            for arm in ("a1", "b1"):
                row["comparisons"].append({
                    "position": position, "arm": arm, "mode": "stock_vs_packed",
                    **compare(calls[arm, "stock"][2], calls[arm, "packed"][2], 8),
                })
            row.setdefault("repeat_checks", []).append({
                "position": position, "comparisons": len(repeats), "all_bitwise_exact": True,
            })
            del calls, owners, operation, buffers, graph, output, values
            gc.collect()
        result["ranks"].append(row)
        (root / "progress.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"rank": rank, "comparisons": row["comparisons"]}), flush=True)
        del weight, scales, compressed
        gc.collect()
        torch.cuda.empty_cache()
    save_file(outputs, root / "outputs.safetensors")
    result.update(status="complete", finished_epoch=time.time(),
                  outputs_sha256=digest(root / "outputs.safetensors"),
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    (root / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    """Run the frozen, isolated component experiment; keep errors in its result directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "source", "evidence", "frozen"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.root, args.source, args.evidence, args.frozen)
    except BaseException as error:
        (args.root / "failure.json").write_text(
            json.dumps({"error": repr(error), "epoch": time.time()}, indent=2) + "\n",
        )
        raise


if __name__ == "__main__":
    main()
