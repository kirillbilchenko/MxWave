"""Reconstruct one actual projection and vary only its captured token permutation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import time
from collections import Counter
from pathlib import Path
from typing import Any

from audit_checkpoint import validate_header


def canonical_order(
    tokens: list[int], experts: list[int], padded: int, block: int,
) -> list[int]:
    """Sort each contiguous expert segment, retaining padding, membership and buffer tails."""
    if block <= 0 or padded < 0 or padded % block or padded > len(tokens):
        raise ValueError("Invalid consumed dispatch range")
    owners = experts[:padded // block]
    if len(owners) != padded // block or any(e < 0 for e in owners):
        raise ValueError("Missing or invalid consumed expert ids")
    if owners != sorted(owners):
        raise ValueError("Require ascending contiguous expert segments")
    result = list(tokens)
    start = 0
    while start < len(owners):
        end = start + 1
        while end < len(owners) and owners[end] == owners[start]:
            end += 1
        result[start * block:end * block] = sorted(tokens[start * block:end * block])
        start = end
    return result


class TensorReader:
    """Read bounded U8 source tensors through verified metadata and complete shard headers."""

    def __init__(self, root: Path, expected: dict[str, str]) -> None:
        self.root = root
        for name, digest in expected.items():
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
                raise ValueError("Source metadata changed")
        self.index = json.loads((root / "model.safetensors.index.json").read_bytes())["weight_map"]
        self.headers: dict[str, tuple[int, dict[str, Any]]] = {}
        self.reads: list[dict[str, Any]] = []

    def raw(self, name: str) -> tuple[bytes, list[int]]:
        """Read at most 16 MiB, checking dtype, complete file coverage and exact byte count."""
        file = self.index[name]
        if not re.fullmatch(r"model-\d{5}-of-\d{5}\.safetensors", file):
            raise ValueError("Unsafe source shard name")
        path = self.root / file
        if file not in self.headers:
            with path.open("rb") as f:
                prefix = f.read(8)
                if len(prefix) != 8:
                    raise ValueError("Truncated source header")
                length = struct.unpack("<Q", prefix)[0]
                if length > 32 * 1024**2:
                    raise ValueError("Source header exceeds bound")
                raw_header = f.read(length)
            if len(raw_header) != length:
                raise ValueError("Truncated source header")
            header = json.loads(raw_header)
            validate_header(header, length, path.stat().st_size)
            self.headers[file] = (length + 8, header)
        offset, header = self.headers[file]
        item = header[name]
        start, stop = item["data_offsets"]
        if item["dtype"] != "U8" or stop - start > 16 * 1024**2:
            raise ValueError("Unsupported or oversized source tensor")
        fd = os.open(path, os.O_RDONLY)
        try:
            raw = os.pread(fd, stop - start, offset + start)
        finally:
            os.close(fd)
        if len(raw) != stop - start:
            raise ValueError("Truncated source tensor")
        self.reads.append({"name": name, "bytes": len(raw),
                           "sha256": hashlib.sha256(raw).hexdigest()})
        return raw, item["shape"]


def main() -> None:
    """Require source reconstruction to match original gold outputs before testing order."""
    import torch
    from analyze_repeat_model import read_tensor
    from full_model_control import METADATA
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_permute_scales
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        mxfp4_marlin_process_scales,
    )
    from vllm.scalar_type import scalar_types

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("Preserve earlier controlled probes")
    plan = json.loads(args.plan.read_bytes())
    if (plan["layer"], plan["rank"], plan["gemm"]) != (4, 1, 3):
        raise ValueError("This bounded first probe covers only the captured layer-4 down projection")
    report: dict[str, Any] = {
        "status": "running", "started_epoch": time.time(), "cases": [],
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        "quality_qualified": False,
    }

    def save(phase: str) -> None:
        report.update(phase=phase, updated_epoch=time.time())
        tmp = args.report.with_suffix(".incomplete")
        tmp.write_text(json.dumps(report, indent=2) + "\n")
        tmp.replace(args.report)

    try:
        torch.set_num_threads(1)
        torch.manual_seed(42)
        device = torch.device("cuda", 0)
        props = torch.cuda.get_device_properties(device)
        if (props.major, props.minor) != (12, 1):
            raise ValueError("Require the qualified SM121 target")
        torch.cuda.set_per_process_memory_fraction(2.5 * 1024**3 / props.total_memory)
        reader = TensorReader(args.source, METADATA)
        n, k = 4096, 1024
        weight = torch.empty((288, k // 16, n * 2), dtype=torch.int32, device=device)
        scales = torch.empty((288, k // 32, n), dtype=torch.float8_e8m0fnu, device=device)
        perm = torch.empty(0, dtype=torch.int32, device=device)
        save("repacking_source")
        for expert in range(288):
            prefix = f"model.language_model.layers.4.mlp.experts.{expert}.down_proj."
            raw, shape = reader.raw(prefix + "weight_packed")
            tensor = torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(shape)
            packed = tensor[:, 512:1024].contiguous().to(device)
            repacked = ops.gptq_marlin_repack(packed.view(torch.int32).T.contiguous(),
                                            perm, k, n, 4, False)
            weight[expert].copy_(repacked)
            raw, shape = reader.raw(prefix + "weight_scale")
            tensor = torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(shape)
            raw_scale = tensor[:, 32:64].contiguous().to(device)
            if bool((raw_scale == 255).any()):
                raise ValueError("Reserved E8M0 scale in source")
            floating = raw_scale.view(torch.float8_e8m0fnu).to(torch.bfloat16)
            prepared = marlin_permute_scales(floating.T, k, n, 32)
            scales[expert].copy_(mxfp4_marlin_process_scales(prepared))
            if expert % 48 == 0:
                report["experts_loaded"] = expert + 1
                save("repacking_source")
        del packed, repacked, raw_scale, floating, prepared, tensor, raw
        report["source_tensor_hashes"] = reader.reads
        for label, value in (("weight", weight), ("scale", scales)):
            digest = hashlib.sha256()
            for expert in value:
                digest.update(expert.view(torch.uint8).cpu().numpy())
            report[f"prepared_{label}_sha256"] = digest.hexdigest()
        save("source_prepared")

        roots, manifests, inputs, gold = [], [], [], []
        event_name = "language_model.model.layers.4.mlp/gemm3"
        dtypes = {"torch.bfloat16": torch.bfloat16, "torch.int32": torch.int32,
                  "torch.float32": torch.float32}
        for capture in plan["captures"]:
            root = args.traces / f"repeat-{capture['label']}-rank1"
            path = root / "snapshots.json"
            if hashlib.sha256(path.read_bytes()).hexdigest() != capture["manifest_sha256"]:
                raise ValueError("Captured manifest changed")
            manifest = json.loads(path.read_bytes())
            records = {e["name"]: e for e in manifest["events"]}

            def read(item: dict[str, Any], root: Path = root) -> torch.Tensor:
                raw, _ = read_tensor(root, item)
                return torch.frombuffer(bytearray(raw), dtype=dtypes[item["dtype"]]).reshape(
                    item["shape"]
                ).to(device)

            values = {v["path"]: read(v) for v in records[event_name + "/input"]["tensors"]}
            expected = read(records[event_name + "/output"]["tensors"][0])
            roots.append(root)
            manifests.append(manifest)
            inputs.append(values)
            gold.append(expected)
        if set(inputs[0]) != {"/0", "/10", "/11", "/12", "/13", "/14"}:
            raise ValueError("Incomplete captured GEMM inputs")
        for key in inputs[0]:
            if key != "/11" and not torch.equal(inputs[0][key].view(torch.uint8),
                                              inputs[1][key].view(torch.uint8)):
                raise ValueError("An input other than permutation changed")
        params = dict(plan["parameters"])
        if (params["use_atomic_add"] or not params["use_fp32_reduce"]
                or params["size_m"] != 2064 or params["size_n"] != n or params["size_k"] != k):
            raise ValueError("The captured arithmetic/shape contract changed")
        params["b_q_type"] = scalar_types.float4_e2m1f
        x = inputs[0]["/0"]
        locks = inputs[0]["/10"].clone()
        sorted_ids = inputs[0]["/11"].clone()
        experts, padded, routing = (inputs[0][key] for key in ("/12", "/13", "/14"))
        output = torch.empty_like(gold[0])

        def run(order: torch.Tensor) -> torch.Tensor:
            sorted_ids.copy_(order)
            locks.copy_(inputs[0]["/10"])
            return ops.moe_wna16_marlin_gemm(
                x, output, weight, None, scales, None, None, None, None, None,
                locks, sorted_ids, experts, padded, routing, **params,
            )

        def stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
            a, b = a.cpu(), b.cpu()
            delta = a.float() - b.float()
            return {"bits_exact": bool(torch.equal(a.view(torch.uint8), b.view(torch.uint8))),
                    "changed_values": int(torch.count_nonzero(a != b)),
                    "max_abs": float(delta.abs().max())}

        save("reconstruction_gate")
        for i in (0, 1):
            result = run(inputs[i]["/11"])
            check = stats(gold[i], result)
            report.setdefault("reconstruction", []).append(check)
            if not check["bits_exact"]:
                raise RuntimeError("Reconstructed source projection does not reproduce captured gold")
        base = run(inputs[0]["/11"]).clone()
        for attempt in range(12):
            side = attempt % 2
            report["cases"].append({"attempt": attempt, "permutation": side,
                                    "versus_a": stats(base, run(inputs[side]["/11"])),
                                    "versus_captured_gold": stats(gold[side], output)})
        orders = [
            canonical_order(v["/11"].cpu().tolist(), experts.cpu().tolist(),
                            int(padded.item()), params["moe_block_size"])
            for v in inputs
        ]
        consumed = int(padded.item())
        if (orders[0][:consumed] != orders[1][:consumed]
                or Counter(orders[0]) != Counter(inputs[0]["/11"].cpu().tolist())):
            raise RuntimeError("Canonicalization changed membership or failed to normalize order")
        canonical = [torch.tensor(v, dtype=torch.int32, device=device) for v in orders]
        reference = run(canonical[0]).clone()
        report["canonical_repeats"] = [
            stats(reference, run(canonical[i % 2])) for i in range(12)
        ]
        report.update(status="passed", finished_epoch=time.time(),
                      cuda_peak_bytes=torch.cuda.max_memory_allocated(),
                      only_permutation_changed=True,
                      limits=["Single projection, one TP partition; no full-model quality claim.",
                              "Canonical order is a CPU diagnostic, not a timed runtime patch."])
        save("complete")
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        save("failed")
        raise


if __name__ == "__main__":
    main()
