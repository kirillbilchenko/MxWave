"""Opt-in bounded expert routing padding for the fixed-width CT verification experiment."""

from __future__ import annotations

import functools
from typing import Any

_INSTALLED = False
_COUNTS: dict[tuple[int, ...], int] = {}


def install() -> None:
    """Wrap the frozen compressed expert dispatch after model loading, before graph capture."""
    global _INSTALLED
    if _INSTALLED:
        raise RuntimeError("Expert reduction policy is already installed")
    import model_plugin
    import torch
    from expert_reduction_policy import bind_padding, block_quantum
    from vllm import _custom_ops as ops

    original = ops.moe_wna16_marlin_gemm
    assert model_plugin._STORAGE, "Install only after the real compressed target is loaded"
    assert model_plugin._GEMM_OP is not None

    @functools.wraps(original)
    def gemm(*args: Any, **kwargs: Any) -> Any:
        if len(args) < 15 or id(args[4]) not in model_plugin._STORAGE:
            return original(*args, **kwargs)
        m, n, k = (int(kwargs[key]) for key in ("size_m", "size_n", "size_k"))
        topk, block = (int(kwargs[key]) for key in ("top_k", "moe_block_size"))
        multiply = kwargs["mul_topk_weights"]
        rows = m // 8 if multiply else m
        if rows > 16:
            return original(*args, **kwargs)
        if (len(args) != 15 or block != 8
                or (n, k, topk, multiply) not in (
                    (2048, 4096, 8, False), (4096, 1024, 1, True),
                ) or (multiply and m % 8) or kwargs["use_atomic_add"]
                or not kwargs["use_fp32_reduce"] or not kwargs["is_k_full"]):
            raise ValueError("Small-row expert call is outside the verified CT policy")
        x = args[0]
        metadata = model_plugin._STORAGE[id(args[4])]
        assert metadata["shape"][0] == 288
        tile = model_plugin._plan(m, n, k, block, topk, 288, x.device.index)
        sms = torch.cuda.get_device_properties(x.device).multi_processor_count
        quantum = block_quantum(n, tile[1], sms, tile[2])
        routing, launch = bind_padding(tuple(args[11:14]), m * topk, block, quantum)
        launch()
        updated = (*args[:11], *routing, args[14])
        options = kwargs | {"thread_k": tile[0], "thread_n": tile[1], "blocks_per_sm": tile[2]}
        key = (m, n, k, block, topk, *tile, quantum, routing[0].numel())
        _COUNTS[key] = _COUNTS.get(key, 0) + 1
        return original(*updated, **options)

    ops.moe_wna16_marlin_gemm = gemm
    _INSTALLED = True


def snapshot() -> dict[str, Any]:
    """Report captured Python dispatches, which are not a count of GPU graph replays."""
    return {
        "installed": _INSTALLED, "policy": "bounded_padding_default_occupancy",
        "target_rows_max": 16, "prefill_above_limit": "native",
        "dispatches": [{"geometry": list(key), "python_calls": count}
                       for key, count in sorted(_COUNTS.items())],
        "limits": "Python counts establish capture coverage; actual graph replays need trace evidence.",
    }
