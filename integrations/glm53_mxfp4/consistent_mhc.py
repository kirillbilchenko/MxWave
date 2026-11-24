"""Keep the audited GLM mHC geometry on one source-pinned native reduction path."""

from __future__ import annotations

import functools
import hashlib
import importlib
import inspect
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

NATIVE_PINS = {
    "vllm.model_executor.kernels.mhc.tilelang": "c8ce81539c779436eb2b9fbba84738f3114bf3ac0a149f2b606f7d38bd1067ce",
    "vllm.model_executor.kernels.mhc.tilelang_kernels": "03aeb3f7c0eabfe8391006ff80f8d4cfedba7cf3d0a168e0dee2d678d3e3bd24",
    "vllm.model_executor.layers.mhc": "0aa41ebd29fea15b91969125712f900edda1f9af0451db65d7e85d473943fc6e",
}
FIXED_SPLITS = 48


def compose_post_pre(
    post: Callable[..., Any], pre: Callable[..., Any], arguments: Mapping[str, Any]
) -> tuple[Any, ...]:
    """Compose existing native operations while preserving caller defaults and return order."""
    current = post(
        **{name: arguments[name] for name in ("x", "residual", "post_layer_mix", "comb_res_mix")}
    )
    options = {
        name: value
        for name, value in arguments.items()
        if name not in {"self", "x", "post_layer_mix", "comb_res_mix", "tile_n"}
    }
    options.update(residual=current, n_splits=FIXED_SPLITS)
    return (current, *pre(**options))


def select_fixed_split(
    native: Callable[..., int], block_k: int, k: int | None, grid_size: int
) -> int:
    """Limit the fixed reduction policy to the component-tested geometry and row budget."""
    if block_k == 64 and k == 16384 and 1 <= grid_size <= 32:
        return FIXED_SPLITS
    return native(block_k, k, grid_size)


def consistent_mhc_policy() -> dict[str, Any]:
    """Describe experimental scope without implying full-model quality qualification."""
    return {
        "native_pins": dict(NATIVE_PINS),
        "fixed_prenorm_splits": FIXED_SPLITS,
        "residual_geometry": [4, 4096],
        "tested_rows": [1, 2048],
        "fused_dispatch": "existing native post followed by existing native pre",
        "installed_before_model_construction_and_graph_capture": True,
        "new_kernel_implementation": False,
        "native_fused_C1_bits_preserved": False,
        "full_model_qualified": False,
        "host_counters_include_graph_replays": False,
        "other_geometries": "delegate to original implementation",
    }


def install_consistent_mhc() -> Callable[[], dict[str, Any]]:
    """Patch native dispatch before model construction, checking every audited source hash."""
    modules = {name: importlib.import_module(name) for name in NATIVE_PINS}
    for name, module in modules.items():
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != NATIVE_PINS[name]:
            raise ValueError("Pinned mHC source changed: " + name)
    native = modules["vllm.model_executor.kernels.mhc.tilelang"]
    kernels = modules["vllm.model_executor.kernels.mhc.tilelang_kernels"]
    layer = modules["vllm.model_executor.layers.mhc"].MHCFusedPostPreOp
    prior = getattr(layer, "_mxwave_consistent_mhc_snapshot", None)
    if prior is not None:
        return prior
    original_split = kernels.compute_num_split
    original_forward = layer.forward_cuda
    signature = inspect.signature(original_forward)
    state: dict[str, Any] = {
        "native_small_row_splits_checked": False,
        "fixed_split_host_calls": 0,
        "other_split_host_calls": 0,
        "composed_host_calls_by_rows": {},
        "delegated_host_calls_by_shape": {},
    }

    def snapshot() -> dict[str, Any]:
        return {
            **consistent_mhc_policy(),
            **state,
            "composed_host_calls_by_rows": dict(state["composed_host_calls_by_rows"]),
            "delegated_host_calls_by_shape": dict(state["delegated_host_calls_by_shape"]),
        }

    @functools.wraps(original_split)
    def fixed_split(block_k: int, k: int | None, grid_size: int) -> int:
        if block_k == 64 and k == 16384 and 1 <= grid_size <= 32:
            if not state["native_small_row_splits_checked"]:
                if original_split(64, 16384, 1) != FIXED_SPLITS:
                    raise ValueError("Require the component-tested native 48-split device policy")
                state["native_small_row_splits_checked"] = True
            state["fixed_split_host_calls"] += 1
        else:
            state["other_split_host_calls"] += 1
        return select_fixed_split(original_split, block_k, k, grid_size)

    @functools.wraps(original_forward)
    def forward(instance: Any, *args: Any, **kwargs: Any) -> Any:
        bound = signature.bind(instance, *args, **kwargs)
        bound.apply_defaults()
        shape = tuple(bound.arguments["residual"].shape)
        if len(shape) != 3 or shape[1:] != (4, 4096) or not 1 <= shape[0] <= 2048:
            key = str(shape)
            counts = state["delegated_host_calls_by_shape"]
            counts[key] = counts.get(key, 0) + 1
            return original_forward(instance, *args, **kwargs)
        key = str(shape[0])
        counts = state["composed_host_calls_by_rows"]
        counts[key] = counts.get(key, 0) + 1
        return compose_post_pre(native.mhc_post_tilelang, native.mhc_pre_tilelang, bound.arguments)

    kernels.compute_num_split = fixed_split
    layer.forward_cuda = forward
    layer._mxwave_consistent_mhc_snapshot = staticmethod(snapshot)
    return snapshot
