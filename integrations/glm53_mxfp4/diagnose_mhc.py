"""Reproduce missed mHC specializations and require zero JIT after extended warmup."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch
from diagnose_mxfp8 import CONFIG_SHA, INDEX_SHA, memory, write
from mhc_warmup import token_sizes
from safetensors import safe_open


def main() -> None:
    """Warm installed mHC operations with actual source weights, leaving serving untouched."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("Preserve prior warmup evidence")
    for name, want in (("config.json", CONFIG_SHA), ("model.safetensors.index.json", INDEX_SHA)):
        if hashlib.sha256((args.source / name).read_bytes()).hexdigest() != want:
            raise ValueError("Source metadata changed")
    from vllm.model_executor.kernels.mhc import tilelang_kernels
    from vllm.model_executor.kernels.mhc.tilelang import (
        mhc_fused_post_pre_tilelang,
        mhc_pre_tilelang,
    )
    from vllm.utils.deep_gemm import is_deep_gemm_supported
    from vllm.utils.jit_monitor import activate

    torch.set_num_threads(2)
    torch.set_default_dtype(torch.bfloat16)
    torch.cuda.set_per_process_memory_fraction(
        1.5 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    config = json.loads((args.source / "config.json").read_bytes())["text_config"]
    hidden, hc = config["hidden_size"], config["hc_mult"]
    index = json.loads((args.source / "model.safetensors.index.json").read_bytes())["weight_map"]
    state: dict[str, Any] = {"status": "running", "started_epoch": time.time(), "warmup": [],
        "code_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                        for name in ("mhc_warmup.py", "diagnose_mhc.py")},
        "source_tensor_hashes": {}, "scope": "mHC warmup component, not full-model timing/quality"}

    def source(tail: str, dtype: torch.dtype) -> torch.Tensor:
        name = "model.language_model.layers.0." + tail
        with safe_open(args.source / index[name], framework="pt", device="cpu") as handle:
            value = handle.get_tensor(name)
        state["source_tensor_hashes"][name] = hashlib.sha256(
            value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        return value.to(dtype=dtype, device="cuda")

    def caches() -> dict[str, int]:
        return {name: len(getattr(tilelang_kernels, name)._kernel_cache) for name in (
            "mhc_pre_big_fuse_with_norm_tilelang", "mhc_fused_tilelang", "mhc_post_tilelang")}

    try:
        state["memory_before"] = memory()
        fn = source("hc_attn_fn", torch.float32)
        scale = source("hc_attn_scale", torch.float32)
        base = source("hc_attn_base", torch.float32)
        norm = source("input_layernorm.weight", torch.bfloat16)
        if not is_deep_gemm_supported():
            raise RuntimeError("Expected the full-model pilot's DeepGEMM mHC dispatch")
        maximum = 2048
        residual = torch.full((maximum, hc, hidden), .125, dtype=torch.bfloat16, device="cuda")
        x = torch.full((maximum, hidden), .0625, dtype=torch.bfloat16, device="cuda")
        post = torch.full((maximum, hc, 1), .25, dtype=torch.float32, device="cuda")
        comb = torch.eye(hc, dtype=torch.float32, device="cuda").expand(maximum, -1, -1).contiguous()
        constants = {"fn": fn, "hc_scale": scale, "hc_base": base,
            "rms_eps": config["rms_norm_eps"], "hc_pre_eps": config["hc_eps"],
            "hc_sinkhorn_eps": config["hc_eps"], "hc_post_mult_value": 2.,
            "sinkhorn_repeat": config["hc_sinkhorn_iters"], "norm_weight": norm,
            "norm_eps": config["rms_norm_eps"]}
        state["constants"] = {k: v for k, v in constants.items() if not isinstance(v, torch.Tensor)}

        def run(tokens: int) -> tuple[torch.Tensor, ...]:
            first = mhc_pre_tilelang(residual=residual[:tokens], **constants)
            second = mhc_fused_post_pre_tilelang(x=x[:tokens], residual=residual[:tokens],
                post_layer_mix=post[:tokens], comb_res_mix=comb[:tokens], **constants)
            return (*first, *second)

        def timed(tokens: int) -> dict[str, Any]:
            before = caches()
            start = time.perf_counter()
            output = run(tokens)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            if not all(bool(torch.isfinite(t).all()) for t in output):
                raise RuntimeError("Non-finite warmed mHC component")
            return {"tokens": tokens, "wall_seconds_including_compile": seconds,
                    "cache_before": before, "cache_after": caches()}

        # This is a component characterization of a common warmup pattern,
        # not a replay of all shapes executed by the v4 model runner.
        state["power_of_two_warmup"] = []
        for tokens in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048):
            state["power_of_two_warmup"].append(timed(tokens))
            write(args.report, state)
        state["mixed_257_after_power_of_two"] = timed(257)
        reference_257 = tuple(value.clone() for value in run(257))
        policy = lambda tokens: tilelang_kernels.compute_num_split(64, hc * hidden, (tokens + 63) // 64)
        sizes = token_sizes(maximum, policy)
        state["selected_sizes"] = sizes
        state["split_representatives"] = {str(policy(size)): size for size in reversed(sizes)}
        for size in sizes:
            state["warmup"].append(timed(size))
            write(args.report, state)
        before = caches()
        activate(mode="error", verbose=True)
        # Exercise both sides of every 64-token dispatch boundary, including
        # mixed prefill/decode sizes absent from power-of-two warmup.
        validation = sorted({n for base_size in range(64, maximum + 1, 64)
                             for n in (base_size - 1, base_size, base_size + 1)
                             if 1 <= n <= maximum} | {1, 7, 8, 16, 17, 257})
        durations = []
        for size in validation:
            durations.append(timed(size))
        after = caches()
        exact = all(torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                    for a, b in zip(reference_257, run(257), strict=True))
        if before != after or not exact:
            raise RuntimeError("Extended warmup still misses kernels or changed deterministic output")
        state.update(status="complete", validated_sizes=validation, validation=durations,
            zero_monitored_jit_after_warmup=True, mixed_257_output_bits_exact_before_after=True,
            cache_before_validation=before, cache_after_validation=after,
            cuda_peak_bytes=torch.cuda.max_memory_allocated(), finished_epoch=time.time())
        write(args.report, state)
    except BaseException as exc:
        state.update(status="failed", error=repr(exc), finished_epoch=time.time())
        write(args.report, state)
        raise


if __name__ == "__main__":
    main()
