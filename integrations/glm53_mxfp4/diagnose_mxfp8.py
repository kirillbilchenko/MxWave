"""Check installed stock MXFP8 Marlin linears against independent source decoding.

This bounded component reads the already verified checkpoint. It does not load
the whole model or install an adapter. Basis vectors require exact BF16 weight
values; random inputs compare GEMM arithmetic with a CPU FP32 reference.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import resource
import time
from functools import partial
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

CONFIG_SHA = "194d290ebb0f19aedc60d24ea9e236e33c86616e460f4a108a842f6fd5b8b198"
INDEX_SHA = "6d1364a27eef78675390422b1b5230601817fb7460ce816951b2cd515934c85d"


def digest(value: torch.Tensor) -> str:
    """Hash exact tensor bytes without converting FP8 values numerically."""
    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def decode(weight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Independently decode E4M3 values and biased E8M0 exponents on CPU."""
    if (weight.device.type != "cpu" or weight.dtype != torch.float8_e4m3fn
            or scales.dtype != torch.uint8 or weight.ndim != 2
            or weight.shape[1] % 32 or scales.shape != (weight.shape[0], weight.shape[1] // 32)
            or bool((scales == 255).any())):
        raise ValueError("Expected finite CPU group-32 E4M3/E8M0 source tensors")
    values = torch.ldexp(weight.float().reshape(*scales.shape, 32),
                         scales.to(torch.int32).unsqueeze(-1) - 127).reshape(weight.shape)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Source dequantization contains non-finite values")
    return values


def difference(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    """Report exact output bits and numeric error without hiding non-finite values."""
    actual = actual.detach().cpu()
    error = actual.float() - reference.float()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(reference).all())
    denominator = float(torch.linalg.vector_norm(reference.float()))
    return {"bits_exact": bool(torch.equal(actual.view(torch.uint8),
                                           reference.view(torch.uint8))),
            "finite": finite, "max_absolute_error": float(error.abs().max()) if finite else None,
            "relative_l2": float(torch.linalg.vector_norm(error)) / max(denominator, 1e-30)
            if finite else None}


def write(path: Path, state: dict[str, Any]) -> None:
    """Retain partial evidence atomically if a numerical assertion fails."""
    temporary = path.with_suffix(".incomplete")
    temporary.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def memory() -> dict[str, Any]:
    """Record unified-memory and container headroom without allocating a model tensor."""
    free, total = torch.cuda.mem_get_info()
    result = {"cuda_free_bytes": free, "cuda_total_bytes": total,
              "cuda_allocated_bytes": torch.cuda.memory_allocated(),
              "cuda_reserved_bytes": torch.cuda.memory_reserved(),
              "max_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    for name in ("memory.current", "memory.max", "memory.events"):
        path = Path("/sys/fs/cgroup") / name
        if path.is_file():
            result[name] = path.read_text().strip()
    return result


def main() -> None:
    """Check dense and early/late shared experts, both TP2 slices and stock graphs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("Preserve previous diagnostic evidence")
    for name, want in (("config.json", CONFIG_SHA), ("model.safetensors.index.json", INDEX_SHA)):
        if hashlib.sha256((args.source / name).read_bytes()).hexdigest() != want:
            raise ValueError("Source metadata differs from the verified full-model pilot")
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
        apply_mxfp8_marlin_linear,
        prepare_mxfp8_layer_for_marlin,
    )

    torch.set_num_threads(2)
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(749)
    device = torch.device("cuda", 0)
    torch.cuda.set_per_process_memory_fraction(
        1.5 * 1024**3 / torch.cuda.get_device_properties(device).total_memory, device)
    index = json.loads((args.source / "model.safetensors.index.json").read_bytes())["weight_map"]
    state: dict[str, Any] = {"status": "running", "started_epoch": time.time(), "layers": [],
        "torch": torch.__version__, "vllm": importlib.metadata.version("vllm"),
        "gates": {"basis_bits_exact": True, "random_relative_l2_maximum": .01},
        "scope": "Unmodified MXFP8 Marlin W8A16, component-only; not model quality",
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    def source(name: str) -> torch.Tensor:
        with safe_open(args.source / index[name], framework="pt", device="cpu") as handle:
            return handle.get_tensor(name)

    try:
        state["memory_before"] = memory()
        write(args.report, state)
        for layer_index in (0, 3, 44):
            prefix = f"model.language_model.layers.{layer_index}.mlp."
            if layer_index:
                prefix += "shared_experts."
            for rank in (0, 1):
                for family in ("gate_up", "down"):
                    names = ("gate_proj", "up_proj") if family == "gate_up" else ("down_proj",)
                    weights, scales, hashes = [], [], []
                    for projection in names:
                        name = prefix + projection
                        weight, scale = source(name + ".weight"), source(name + ".weight_scale")
                        hashes.append({"name": name, "weight_sha256": digest(weight),
                                       "scale_sha256": digest(scale)})
                        if family == "gate_up":
                            width = weight.shape[0] // 2
                            weight, scale = weight[rank * width:(rank + 1) * width], \
                                scale[rank * width:(rank + 1) * width]
                        else:
                            width = weight.shape[1] // 2
                            weight = weight[:, rank * width:(rank + 1) * width]
                            scale = scale[:, rank * width // 32:(rank + 1) * width // 32]
                        weights.append(weight.contiguous())
                        scales.append(scale.contiguous())
                    weight, scale = torch.cat(weights), torch.cat(scales)
                    del weights, scales
                    reference_weight = decode(weight, scale)
                    if not torch.equal(reference_weight, reference_weight.bfloat16().float()):
                        raise RuntimeError("Source dequantization is not exactly representable in BF16")
                    n, k = weight.shape
                    state["current_layer"] = {"index": layer_index, "rank": rank, "family": family,
                                               "shape_nk": [n, k], "memory": memory()}
                    write(args.report, state)
                    layer = torch.nn.Module()
                    layer.input_size_per_partition, layer.output_size_per_partition = k, n
                    layer.register_parameter("weight", torch.nn.Parameter(
                        weight.to(device), requires_grad=False))
                    layer.register_parameter("weight_scale", torch.nn.Parameter(
                        scale.to(device), requires_grad=False))
                    prepare_mxfp8_layer_for_marlin(layer)
                    row: dict[str, Any] = {"prefix": prefix, "rank": rank, "family": family,
                        "shape_nk": [n, k], "source_hashes": hashes, "cases": []}
                    state["layers"].append(row)
                    columns = sorted({0, 1, 7, 8, 15, 16, 31, 32, 33, 63, k // 2, k - 1})
                    basis = torch.zeros((len(columns), k), dtype=torch.bfloat16)
                    basis[torch.arange(len(columns)), columns] = 1
                    for label, x in [("basis", basis), *[(f"random_{m}", torch.randn(
                            (m, k), dtype=torch.float32).bfloat16() * .125) for m in (1, 8, 64)]]:
                        reference = (x.float() @ reference_weight.T).bfloat16()
                        activation = x.to(device)

                        run = partial(apply_mxfp8_marlin_linear,
                            input=activation, weight=layer.weight,
                            weight_scale=layer.weight_scale, workspace=layer.workspace,
                            size_n=n, size_k=k)

                        actual = run().clone()
                        repeat = run()
                        torch.cuda.synchronize()
                        result = difference(actual, reference)
                        result.update(label=label, repeat_bits_exact=bool(torch.equal(
                            actual.view(torch.uint8), repeat.view(torch.uint8))))
                        if label in ("basis", "random_8"):
                            stream = torch.cuda.Stream()
                            stream.wait_stream(torch.cuda.current_stream())
                            with torch.cuda.stream(stream):
                                for _ in range(3):
                                    run()
                            stream.synchronize()
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph, stream=stream):
                                captured = run()
                            graph.replay()
                            torch.cuda.synchronize()
                            result["graph_bits_exact_to_eager"] = bool(torch.equal(
                                actual.view(torch.uint8), captured.view(torch.uint8)))
                            del graph, captured
                        result["passed"] = bool(result["finite"] and result["repeat_bits_exact"]
                            and result.get("graph_bits_exact_to_eager", True)
                            and (result["bits_exact"] if label == "basis"
                                 else result["relative_l2"] <= .01))
                        row["cases"].append(result)
                        write(args.report, state)
                    print(json.dumps({"prefix": prefix, "rank": rank, "family": family,
                                      "cases": row["cases"]}), flush=True)
                    del run, layer, actual, repeat, activation, weight, scale
                    del reference_weight, reference
                    gc.collect()
                    torch.cuda.empty_cache()
        passed = all(case["passed"] for row in state["layers"] for case in row["cases"])
        state.update(status="complete" if passed else "numerical_failure", passed=passed,
                     cuda_peak_bytes=torch.cuda.max_memory_allocated(), finished_epoch=time.time())
        write(args.report, state)
        if not passed:
            raise RuntimeError("Stock MXFP8 component failed its independent reference checks")
    except BaseException as exc:
        state.update(error=repr(exc), finished_epoch=time.time())
        try:
            state["memory_failure"] = memory()
        except (RuntimeError, OSError) as memory_exc:
            state["memory_failure"] = {"error": repr(memory_exc)}
        if state["status"] == "running":
            state["status"] = "failed"
        write(args.report, state)
        raise


if __name__ == "__main__":
    main()
