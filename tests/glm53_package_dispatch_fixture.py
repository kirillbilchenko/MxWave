"""Exercise the 0.32 Marlin call boundary in an isolated CPU process."""

from __future__ import annotations

import inspect
import sys
from types import ModuleType, SimpleNamespace

import torch


def main() -> None:
    """Check pointer/argument ownership, stock passthrough and unsupported modes."""
    from mxwave_glm53._runtime import model_plugin as runtime

    fallback = []

    def original(
        input, output, b_qweight, b_bias, b_scales, a_scales, global_scale, b_qzeros,
        workspace, sorted_token_ids, expert_ids, num_tokens_past_padded, topk_weights,
        moe_block_size, top_k, mul_topk_weights, b_q_type, size_m, size_n, size_k,
        use_atomic_add, use_fp32_reduce, is_zp_float, thread_k=-1, thread_n=-1,
        blocks_per_sm=-1,
    ):
        fallback.append((input, output, b_scales))
        return output

    vllm = ModuleType("vllm")
    ops = SimpleNamespace(moe_wna16_marlin_gemm=original)
    vllm._custom_ops = ops
    sys.modules["vllm"] = vllm
    module_name = (
        "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe."
        "compressed_tensors_moe_w4a4_mxfp4"
    )
    module = ModuleType(module_name)
    module.CompressedTensorsW4A4Mxfp4MoEMethod = type(
        "Method", (), {"process_weights_after_loading": lambda self, layer: None})
    sys.modules[module_name] = module
    import os

    os.environ["MXWAVE_MOE_MODE"] = "aligned"
    runtime.install()
    x = torch.empty((3, 128), dtype=torch.bfloat16)
    output = torch.empty((3, 128), dtype=torch.bfloat16)
    weight = torch.empty((2, 8, 256), dtype=torch.int32)
    scale = torch.empty(1, dtype=torch.uint8)
    scratch = torch.empty(1)
    routing = [torch.empty(1) for _ in range(5)]
    runtime._STORAGE[id(scale)] = {"shape": (2, 4, 128), "kind": 1}
    launches = []
    runtime._GEMM_OP = lambda *args: launches.append(args)
    runtime._temporary = lambda tensor: scratch
    runtime._plan = lambda *args: (64, 128, 2)
    params = {
        "input": x, "output": output, "b_qweight": weight, "b_bias": None, "b_scales": scale,
        "a_scales": None, "global_scale": None, "b_qzeros": None, "workspace": routing[0],
        "sorted_token_ids": routing[1], "expert_ids": routing[2],
        "num_tokens_past_padded": routing[3], "topk_weights": routing[4],
        "moe_block_size": 8, "top_k": 1, "mul_topk_weights": True, "b_q_type": object(),
        "size_m": 3, "size_n": 128, "size_k": 128, "use_atomic_add": False,
        "use_fp32_reduce": True, "is_zp_float": False,
    }
    assert ops.moe_wna16_marlin_gemm(**params) is output
    positional = inspect.signature(original).bind(**params)
    assert ops.moe_wna16_marlin_gemm(*positional.args) is output
    for launch in launches:
        expected_tensors = (x, weight, scale, output, scratch, *routing)
        assert all(actual is expected for actual, expected in zip(launch, expected_tensors))
        assert launch[10:] == (3, 128, 128, 2, 1, 8, True, 64, 128, 2, 1)
    assert not fallback and len(launches) == 2
    owned = params["b_scales"]
    params["b_scales"] = torch.empty(1)
    assert ops.moe_wna16_marlin_gemm(**params) is output and len(fallback) == 1
    assert fallback[0][2] is params["b_scales"]
    params["b_scales"] = owned
    for key, value in (("use_atomic_add", True), ("use_fp32_reduce", False),
                       ("b_bias", torch.ones(1)), ("size_k", 256)):
        invalid = {**params, key: value}
        try:
            ops.moe_wna16_marlin_gemm(**invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Unsupported call accepted: {key}")
    assert len(launches) == 2, "Rejected calls must not enter native dispatch"

    # The 0.32 repacker leaves no layer.workspace; the official op owns its locks.
    linear_name = "vllm.model_executor.layers.linear"
    linear = ModuleType(linear_name)
    linear.UnquantizedLinearMethod = type("UnquantizedLinearMethod", (), {})
    sys.modules[linear_name] = linear
    fp8_name = "vllm.model_executor.layers.quantization.utils.marlin_utils_fp8"
    fp8 = ModuleType(fp8_name)
    forwarded = []
    fp8.apply_mxfp8_marlin_linear = lambda **kw: forwarded.append(kw) or output
    sys.modules[fp8_name] = fp8
    from mxwave_glm53._runtime.kda_mxfp8_runtime import method_type

    layer = SimpleNamespace(weight=weight, weight_scale=scale,
                            output_size_per_partition=128, input_size_per_partition=128)
    assert method_type()().apply(layer, x) is output
    assert forwarded[0]["workspace"] is None
    assert forwarded[0]["weight"] is weight and forwarded[0]["weight_scale"] is scale


if __name__ == "__main__":
    main()
