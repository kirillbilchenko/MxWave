"""Observe actual layer-three MLA operations with bounded copies and unchanged native calls."""

from __future__ import annotations

import functools
import re
from collections.abc import Callable
from typing import Any

import torch
from decode_boundary_capture import BoundaryCapture, tensor_leaves

MAX_STAGING_BYTES = 8 * 1024**2
MAX_TOTAL_BYTES = 32 * 1024**2
TRANSFER_ALLOWANCE_BYTES = 8 * 1024**2


def layout(value: torch.Tensor) -> dict[str, Any]:
    """Describe the original operand view, including empty NoPE tensors and aliases."""
    storage = value.untyped_storage()
    return {"shape": list(value.shape), "stride": list(value.stride()),
        "dtype": str(value.dtype), "storage_offset": value.storage_offset(),
        "storage_bytes": storage.nbytes(), "storage_address": storage.data_ptr()}


class MLAInternalCapture(BoundaryCapture):
    """Replace the broad boundary probe with one layer's internal capture, never stack both."""

    def __init__(self, capturing: Callable[[], bool] | None = None) -> None:
        """Keep at most24 MiB of retained copies plus8 MiB of conservatively counted staging."""
        super().__init__(capturing)
        self.staging_bytes = 0
        self.target: Any = None
        self.state: Any = None
        self.constants: dict[str, torch.Tensor] = {}
        self.captured_plan: list[Any] = []
        self.valid_counts: torch.Tensor | None = None
        self.topk_width = 0

    def record(self, name: str, category: str, value: Any,
               parameters: dict[str, Any] | None = None) -> None:
        """Record logical values while preserving original view and zero-width metadata."""
        params = dict(parameters or {})
        params["operand_layouts"] = {path: layout(t) for path, t in tensor_leaves(value)}
        size = sum(t.numel() * t.element_size() for _, t in tensor_leaves(value))
        if self.nbytes + self.staging_bytes + size + TRANSFER_ALLOWANCE_BYTES > MAX_TOTAL_BYTES:
            raise MemoryError("Internal MLA copies and staging exceed32 MiB")
        previous = len(self.events)
        super().record(name, category, value, params)
        if len(self.events) == previous:
            self.events.append({"name": name, "category": category,
                                "parameters": params, "tensors": []})

    def selected_cache(self, name: str, cache: torch.Tensor, indices: torch.Tensor,
                       counts: torch.Tensor | None = None) -> None:
        """Gather FP8 bytes in actual slot order; zero invalid tails without indexing FP8 values."""
        if (cache.dtype not in (torch.uint8, torch.float8_e4m3fn)
                or not cache.is_contiguous() or cache.shape[-1] != 512):
            raise ValueError("Require the pinned contiguous512-byte FP8 cache layout")
        if indices.ndim not in (1, 2) or indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("Require integral slot indices with one or two dimensions")
        if counts is not None and (indices.ndim != 2 or counts.numel() != indices.shape[0]):
            raise ValueError("Valid counts must correspond to the actual query rows")
        count = indices.numel()
        # Account conservatively for cast/clamp/mask/gather temporaries, not only saved clones.
        staging = count * (512 + 8 + 8 + 4)
        if counts is not None:
            staging += indices.shape[1] * 8 + count * 8
        if (self.staging_bytes + staging > MAX_STAGING_BYTES
                or self.nbytes + self.staging_bytes + staging + count * 513
                    + TRANSFER_ALLOWANCE_BYTES > MAX_TOTAL_BYTES):
            raise MemoryError("Internal MLA cache gather exceeds the combined staging/copy budget")
        self.staging_bytes += staging
        raw = cache.view(torch.uint8).view(-1, 512)
        ids = indices.reshape(-1).to(dtype=torch.int64)
        valid = (ids >= 0) & (ids < raw.shape[0])
        if counts is not None:
            columns = torch.arange(indices.shape[1], device=indices.device)
            valid &= (columns[None, :] < counts.reshape(-1, 1)).reshape(-1)
        selected = raw.index_select(0, ids.clamp(0, raw.shape[0] - 1))
        selected.masked_fill_(~valid[:, None], 0)
        self.record(name, "cache_bytes", {
            "bytes": selected.view(*indices.shape, 512), "valid": valid.view(indices.shape)},
            {"invalid_slots_and_tails_zeroed": True, "slot_order": "original_query_then_topk"})

    def _active(self) -> bool:
        return self.active_layer == 3 and self.capturing()

    def _scope(self, original: Any) -> Any:
        @functools.wraps(original)
        def call(*args: Any, **kwargs: Any) -> Any:
            q = args[0] if args else kwargs.get("q")
            if not self.capturing() or not isinstance(q, torch.Tensor) or q.shape[0] != 4:
                return original(*args, **kwargs)
            previous, self.active_layer = self.active_layer, 3
            try:
                self.record("mla/input", "operand", (args, kwargs))
                result = original(*args, **kwargs)
                self.record("mla/output", "result", result)
                return result
            finally:
                self.active_layer = previous
        return call

    def _bmm(self, original: Any) -> Any:
        @functools.wraps(original)
        def call(*args: Any, **kwargs: Any) -> Any:
            if not self._active():
                return original(*args, **kwargs)
            weight = args[1] if len(args) > 1 else kwargs.get("mat2")
            label = next((name for name, tensor in self.constants.items()
                          if weight is tensor), None)
            if label is None:
                return original(*args, **kwargs)
            operand = args[0] if args else kwargs["input"]
            self.record(label + "/input", "operand", operand,
                        {"constant_layout": layout(weight)})
            result = original(*args, **kwargs)
            self.record(label + "/output", "result", result)
            return result
        return call

    def _cache_update(self, original: Any) -> Any:
        @functools.wraps(original)
        def call(*args: Any, **kwargs: Any) -> Any:
            if not self._active():
                return original(*args, **kwargs)
            if len(args) != 6 or kwargs:
                raise ValueError("Pinned native MLA cache-write signature changed")
            latent, rope, cache, slots, dtype, scale = args
            self.record("kv_write/input", "operand",
                {"latent": latent, "rope": rope, "slots": slots, "scale": scale},
                {"cache_dtype": dtype, "cache_layout": layout(cache)})
            self.selected_cache("kv_write/before", cache, slots[:4])
            result = original(*args, **kwargs)
            self.selected_cache("kv_write/after", cache, slots[:4])
            return result
        return call

    def _convert(self, original: Any) -> Any:
        @functools.wraps(original)
        def call(*args: Any, **kwargs: Any) -> Any:
            if not self._active():
                return original(*args, **kwargs)
            if len(args) != 3 or not kwargs.get("return_valid_counts"):
                raise ValueError("Require the actual sparse slot conversion and valid counts")
            self.record("sparse_convert/input", "operand", args, kwargs)
            result = original(*args, **kwargs)
            slots, self.valid_counts = result
            self.topk_width = slots.shape[1]
            self.record("sparse_convert/output", "result",
                        {"slots": slots, "valid_counts": self.valid_counts})
            return result
        return call

    def _attention(self, original: Any) -> Any:
        @functools.wraps(original)
        def call(*args: Any, **kwargs: Any) -> Any:
            if not self._active():
                return original(*args, **kwargs)
            if len(args) != 4 or self.valid_counts is None or self.topk_width <= 0:
                raise ValueError("Require native query/cache operands after actual sparse conversion")
            q, q_pe, cache, k_pe = args
            if q.shape[0] != 4 or q_pe.numel() != 0 or k_pe.numel() != 0:
                raise ValueError("Require the actual four-row zero-RoPE MLA geometry")
            wrapper = self.state.wrapper
            self.captured_plan = list(wrapper._plan_info)
            indices = wrapper._kv_indices_buf[:4 * self.topk_width].view(4, self.topk_width)
            self.record("attention/input", "operand", {
                "q": q, "q_pe": q_pe, "indices": indices,
                "qo_indptr": wrapper._qo_indptr_buf, "kv_indptr": wrapper._kv_indptr_buf,
                "kv_lens": wrapper._kv_len_arr_buf, "valid_counts": self.valid_counts},
                {"scales": kwargs, "captured_plan_info": self.captured_plan,
                 "cache_layout": layout(cache), "k_pe_layout": layout(k_pe)})
            # A mismatched schedule may read clamped slot-zero tails. Preserve the planned
            # prefix, including those real reads; record conversion-valid counts separately.
            self.selected_cache("attention/selected_cache", cache, indices,
                                wrapper._kv_len_arr_buf[:4])
            result = original(*args, **kwargs)
            self.record("attention/output", "result", result)
            return result
        return call

    def install(self, model: Any, backend: Any) -> None:
        """Wrap native calls in layer3 only; never install the broad multi-layer boundary probe."""
        targets = [module for name, module in model.named_modules()
                   if re.search(r"(?:^|\.)layers\.3\.", name)
                   and all(hasattr(module, attr) for attr in ("W_UK_T", "W_UV", "impl"))]
        if len(targets) != 1:
            raise ValueError("Require one actual inner MLA module in layer3")
        self.target = targets[0]
        if (self.target.qk_rope_head_dim != 0 or self.target.kv_lora_rank != 512
                or self.target.num_heads != 32 or self.target.impl.dcp_world_size != 1):
            raise ValueError("Actual MLA geometry differs from the audited TP2 NoPE path")
        self.state = backend._get_sm90_state()
        if self.state is None:
            raise ValueError("Require the eagerly initialized native shared MLA wrapper")
        self.constants = {"query_bmm": self.target.W_UK_T, "value_bmm": self.target.W_UV}
        self.replace(self.target, "forward", self._scope)
        self.replace(torch, "bmm", self._bmm)
        self.replace(self.target.impl, "do_kv_cache_update", self._cache_update)
        self.replace(backend, "triton_convert_req_index_to_global_index", self._convert)
        self.replace(self.state.wrapper, "run", self._attention)

    def metadata(self) -> dict[str, Any]:
        """Include the complete copy/staging budget and native plan identity without tensor values."""
        return {**super().metadata(), "staging_bytes_conservative": self.staging_bytes,
            "combined_copy_staging_bytes": self.nbytes + self.staging_bytes,
            "maximum_combined_copy_staging_bytes": MAX_TOTAL_BYTES,
            "broad_boundary_probe_installed": False, "captured_plan_info": self.captured_plan}


def patch_manager(manager_type: Any, backend: Any) -> None:
    """Capture one real four-row MLA trace and remove every Python hook before serving."""
    original = manager_type.capture

    @functools.wraps(original)
    def capture(manager: Any, model: Any, *args: Any, **kwargs: Any) -> Any:
        trace = MLAInternalCapture()
        try:
            trace.install(model, backend)
            result = original(manager, model, *args, **kwargs)
        finally:
            trace.restore()
        names = [event["name"] for event in trace.events]
        expected = ["mla/input", "kv_write/input", "kv_write/before", "kv_write/after",
            "query_bmm/input", "query_bmm/output", "sparse_convert/input", "sparse_convert/output",
            "attention/input", "attention/selected_cache", "attention/output",
            "value_bmm/input", "value_bmm/output", "mla/output"]
        if names != expected or len([d for d in manager.graphs if d.num_tokens == 4]) != 1:
            raise ValueError("Require exactly one complete native four-row MLA operation chain")
        if trace.nbytes + trace.staging_bytes + 8 * 1024**2 > MAX_TOTAL_BYTES:
            raise MemoryError("MLA capture plus the reserved transfer allowance exceeds32 MiB")
        manager._mxwave_mla_internal_trace = trace
        return result

    manager_type.capture = capture
