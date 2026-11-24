"""Exercise the exact diagnostic checker with CUDA references before another full load."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from diagnose_mxfp8 import CONFIG_SHA, INDEX_SHA, memory, write
from diagnostic_worker import compare_tensors
from safetensors import safe_open


def main() -> None:
    """Check mixed-device hashes and a real prepared source expert in bounded memory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("Preserve prior checker regression evidence")
    for name, want in (("config.json", CONFIG_SHA), ("model.safetensors.index.json", INDEX_SHA)):
        if hashlib.sha256((args.source / name).read_bytes()).hexdigest() != want:
            raise ValueError("Diagnostic source metadata changed")
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        prepare_moe_fp4_layer_for_marlin,
    )

    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(
        1.5 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    state: dict[str, Any] = {"status": "running", "started_epoch": time.time(), "cases": [],
        "code_sha256": {
            name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ("diagnostic_worker.py", "diagnose_checker.py")},
        "scope": "Checker device/hash regression and one source expert; not full loader coverage"}

    def check(name: str, actual: torch.Tensor, expected: torch.Tensor, exact: bool = True) -> None:
        record = {"name": name, "actual_device": str(actual.device),
                  "expected_device": str(expected.device), **compare_tensors(actual, expected)}
        state["cases"].append(record)
        if record["bits_exact"] != exact:
            raise RuntimeError(f"Checker regression failed: {name}")

    try:
        state["memory_before"] = memory()
        write(args.report, state)
        for dtype in (torch.int32, torch.bfloat16, torch.float32,
                      torch.float8_e4m3fn, torch.float8_e8m0fnu):
            if dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu):
                cpu = torch.arange(256, dtype=torch.uint8).view(dtype)
            else:
                cpu = torch.arange(24).to(dtype).reshape(4, 6).T
            gpu = cpu.cuda()
            check(f"{dtype}/cpu-cuda", cpu, gpu)
            check(f"{dtype}/cuda-cpu", gpu, cpu)
            check(f"{dtype}/cuda-cuda", gpu, gpu.clone())
        check("signed-zero", torch.tensor([0.]), torch.tensor([-0.], device="cuda"), False)
        check("shape-mismatch", torch.ones(2, 2), torch.ones(4, device="cuda"), False)
        check("scalar", torch.tensor(.5), torch.tensor(.5, device="cuda"))
        check("empty", torch.empty(0), torch.empty(0, device="cuda"))

        metadata = json.loads((args.source / "model.safetensors.index.json").read_bytes())
        index = metadata["weight_map"]
        expert = torch.nn.Module()
        expert.params_dtype = torch.bfloat16
        expert.moe_config = SimpleNamespace(num_local_experts=1, hidden_dim=4096,
                                            intermediate_size_per_partition=1024)
        for family in ("w13", "w2"):
            for suffix in ("weight_packed", "weight_scale"):
                parts = []
                projections = ("gate_proj", "up_proj") if family == "w13" else ("down_proj",)
                for projection in projections:
                    key = f"model.language_model.layers.3.mlp.experts.0.{projection}.{suffix}"
                    with safe_open(args.source / index[key], framework="pt", device="cpu") as handle:
                        value = handle.get_tensor(key)
                    stride = 32 if suffix == "weight_scale" else 2
                    value = value[:1024] if family == "w13" else value[:, :1024 // stride]
                    parts.append(value.contiguous())
                name = family + ("_weight" if suffix == "weight_packed" else "_weight_scale")
                expert.register_parameter(name, torch.nn.Parameter(
                    torch.cat(parts).unsqueeze(0).cuda(), requires_grad=False))
        prepare_moe_fp4_layer_for_marlin(expert)
        for name in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"):
            value = getattr(expert, name)[0]
            check(f"prepared-expert/{name}", value.cpu(), value)
            changed = value.clone()
            changed.reshape(-1).view(torch.uint8)[0] ^= 1
            check(f"prepared-expert/{name}/changed-bit", value.cpu(), changed, False)
        state.update(status="complete", passed=True, finished_epoch=time.time(),
                     cuda_peak_bytes=torch.cuda.max_memory_allocated())
        write(args.report, state)
    except BaseException as exc:
        state.update(status="failed", passed=False, error=repr(exc), finished_epoch=time.time())
        write(args.report, state)
        raise


if __name__ == "__main__":
    main()
