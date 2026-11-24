"""Use the pinned native single-row compaction path for non-power-of-two MLA widths."""

from __future__ import annotations

import functools
import hashlib
import importlib
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional

NATIVE_SHA256 = "cd95caf494f1d2a946ac87abfb4951f738827fe7749aeefd6c0fd83d1cee072d"


def convert_stable(
    native: Callable[..., Any],
    request_ids: torch.Tensor,
    block_table: torch.Tensor,
    token_indices: torch.Tensor,
    **kwargs: Any,
) -> Any:
    """Pad invalid columns, call native compaction, and retain the original output width."""
    if not kwargs.get("return_valid_counts", False):
        return native(request_ids, block_table, token_indices, **kwargs)
    if token_indices.ndim != 2 or token_indices.shape[1] <= 0:
        raise ValueError("Require a nonempty sparse index row")
    width = token_indices.shape[1]
    if kwargs.get("NUM_TOPK_TOKENS", 2048) != width:
        raise ValueError("Declared sparse width differs from the input")
    padded_width = 1 << (width - 1).bit_length()
    if padded_width == width:
        return native(request_ids, block_table, token_indices, **kwargs)
    padded = functional.pad(token_indices, (0, padded_width - width), value=-1)
    options = {**kwargs, "NUM_TOPK_TOKENS": padded_width}
    slots, counts = native(request_ids, block_table, padded, **options)
    return slots[:, :width].contiguous(), counts


def install_stable_conversion() -> dict[str, Any]:
    """Wrap the canonical helper and all loaded vLLM aliases before graph capture."""
    helper = importlib.import_module("vllm.v1.attention.backends.mla.sparse_utils")
    native = helper.triton_convert_req_index_to_global_index
    if getattr(native, "_mxwave_stable_sparse", False):
        convert = native
        native = convert.__wrapped__
    else:
        source = Path(native.__code__.co_filename)
        if hashlib.sha256(source.read_bytes()).hexdigest() != NATIVE_SHA256:
            raise ValueError("Pinned sparse conversion implementation changed")

        @functools.wraps(native)
        def convert(
            request_ids: torch.Tensor,
            block_table: torch.Tensor,
            token_indices: torch.Tensor,
            **kwargs: Any,
        ) -> Any:
            return convert_stable(native, request_ids, block_table, token_indices, **kwargs)

        convert._mxwave_stable_sparse = True  # type: ignore[attr-defined]

    aliases = []
    for name, module in tuple(sys.modules.items()):
        if module is None or not name.startswith("vllm."):
            continue
        for alias, value in tuple(vars(module).items()):
            if value is native:
                setattr(module, alias, convert)
            if value is native or value is convert:
                aliases.append(name + "." + alias)
    if helper.triton_convert_req_index_to_global_index is not convert:
        raise RuntimeError("Canonical sparse conversion helper was not wrapped")
    return {**stable_conversion_policy(), "patched_aliases": sorted(aliases)}


def stable_conversion_policy() -> dict[str, Any]:
    """Describe the adapter's scope without claiming numerical or speed qualification."""
    return {
        "native_source_sha256": NATIVE_SHA256,
        "non_power_of_two_counted_rows": "pad invalid columns to the next power of two",
        "native_compaction": "one tile per row, without atomic allocation between tiles",
        "returned_width": "original width, contiguous",
        "unaccounted_valid_entries_added": 0,
        "new_native_kernel": False,
        "alias_scope": "canonical helper and all loaded vLLM references; future imports inherit it",
        "DCP_scope": "DCP size one only; multi-rank DCP compaction is outside this experiment",
        "production_service_modified": False,
        "qualification_attempted": False,
    }
