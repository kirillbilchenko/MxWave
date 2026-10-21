"""Small native GPU qualification before allocating full GLM model weights."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def check(bundle: Path) -> dict:
    """Verify native exact decoding, GEMM calls and graph replay on SM121."""
    import torch

    from mxwave_glm53.cli import doctor

    report = {"started_epoch": time.time(), "scope": "synthetic GPU components only"}
    installation = doctor(bundle, gpu=True)
    if installation["status"] != "passed":
        raise RuntimeError(installation)
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
        prepare_mxfp8_layer_for_marlin,
    )

    from mxwave_glm53._runtime import head
    from mxwave_glm53._runtime.kda_mxfp8_runtime import encode_weight, method_type
    from mxwave_glm53._runtime.moe_runtime import NativeMoe, allocate_buffers, pack_scales

    torch.set_default_dtype(torch.bfloat16)
    device = torch.device("cuda", 0)
    generator = torch.Generator(device=device).manual_seed(42)
    raw = NativeMoe(bundle / "native/libmxwave_moe_raw.so")
    compressed = NativeMoe(bundle / "native/libmxwave_moe_sparse.so")
    # Include non-palette exceptions and a range crossing the nibble-base boundary.
    bits = torch.full((2, 8, 128), 125, dtype=torch.uint8, device=device)
    bits.reshape(-1)[::7] = 126
    bits.reshape(-1)[::17] = 127
    bits.reshape(-1)[::53] = 128
    scales = bits.view(torch.float8_e8m0fnu)
    packed = pack_scales(scales, bundle / "bundle.json", encoding="sparse_palette")
    decoded = compressed.decode(packed)
    if not torch.equal(decoded, bits):
        raise RuntimeError("Expert scale decode differs")
    reordered = bits.reshape(-1, 4)[:, [0, 2, 1, 3]].to(torch.int32)
    bf16_bits = (reordered << 7).to(torch.int16).contiguous().view(torch.uint8)
    if not torch.equal(compressed.decode_bf16(packed).reshape(-1), bf16_bits.reshape(-1)):
        raise RuntimeError("Expert scale BF16 fragments differ")
    weight = torch.randint(-(2**31), 2**31 - 1, (2, 16, 256),
                           dtype=torch.int32, device=device, generator=generator)
    cases = []
    for rows in (1, 6, 12, 16, 48):
        padded_rows = (rows + 7) // 8 * 8
        ids = torch.arange(padded_rows, dtype=torch.int32, device=device).clamp_max(rows)
        experts = torch.zeros(padded_rows // 8, dtype=torch.int32, device=device)
        count = torch.tensor([padded_rows], dtype=torch.int32, device=device)
        routing = (ids, experts, count)
        topk_weights = torch.ones((rows, 1), dtype=torch.float32, device=device)
        x = torch.randn((rows, 256), dtype=torch.float32, device=device,
                        generator=generator).bfloat16() * 0.01
        plan = raw.plan(rows, 128, 256, 8, 1, 2, 0)
        buffers = [allocate_buffers(rows, 128, 1, 8, padded_rows, device) for _ in range(2)]
        operations = [native.bind(x, weight, scale, buf, routing, topk_weights, 1, 8, False,
                                  plan) for native, scale, buf in
                      zip((raw, compressed), (scales, packed), buffers)]
        reference = operations[0]().clone()
        actual = operations[1]()
        if not torch.equal(reference.view(torch.uint8), actual.view(torch.uint8)):
            raise RuntimeError(f"Raw/compressed expert GEMM differs at M={rows}")
        if not bool(torch.isfinite(actual).all()):
            raise RuntimeError("Expert GEMM produced nonfinite values")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                operations[1]()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            operations[1]()
        for _ in range(4):
            graph.replay()
        torch.cuda.synchronize()
        if not torch.equal(reference.view(torch.uint8), actual.view(torch.uint8)):
            raise RuntimeError(f"Expert graph replay differs at M={rows}")
        cases.append({"rows": rows, "raw_compressed_bitwise": True, "graph_replays": 4})
    # Exercise the upstream native repacker and MXFP8 helper with known rounded weights.
    dense = torch.randn((128, 256), dtype=torch.float32, device=device,
                        generator=generator).bfloat16() * 0.125
    codes, exponents, rounded, _ = encode_weight(dense)
    layer = torch.nn.Module()
    layer.bias = None
    layer.weight = torch.nn.Parameter(codes.view(torch.float8_e4m3fn), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(exponents, requires_grad=False)
    layer.output_size_per_partition = 128
    layer.input_size_per_partition = 256
    prepare_mxfp8_layer_for_marlin(layer)
    method = method_type()()
    columns = [0, 31, 128, 255]
    basis = torch.zeros((4, 256), device=device)
    basis[torch.arange(4, device=device), columns] = 1
    if not torch.equal(method.apply(layer, basis), rounded[:, columns].T):
        raise RuntimeError("Upstream MXFP8 native basis decode differs")
    # The compressed head's CPU encoder and GPU decoder must preserve the BF16 payload.
    os.environ["MXWAVE_ZIPSERV_LIBRARY"] = str(bundle / "native/libmxwave_kda_zipserv.so")
    packed_head = head.compress(dense.cpu()).to(device)
    if not torch.equal(head.decompress(packed_head).view(torch.uint8), dense.view(torch.uint8)):
        raise RuntimeError("Vocabulary decoder differs")
    head_result = head.linear(basis, packed_head)
    if not torch.equal(head_result, dense[:, columns].T):
        raise RuntimeError("Vocabulary GEMM basis decode differs")
    # Attribute access above must resolve a real vLLM GPU operation, not a CPU stub.
    assert callable(ops.gptq_marlin_repack)
    report.update(status="passed", installation=installation, expert_cases=cases,
                  exact_expert_scale_decode=True, exact_bf16_fragments=True,
                  exact_mxfp8_basis=True, exact_vocab_decode=True, exact_vocab_basis=True,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  full_model_qualified=False, finished_epoch=time.time())
    return report


def main() -> None:
    """Retain either a complete component result or the exact failed check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    try:
        result = check(args.bundle)
    except Exception as exc:
        args.output.write_text(json.dumps({"status": "failed", "error": repr(exc)}, indent=2))
        raise
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
