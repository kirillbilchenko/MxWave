"""Retain the native sparse-indexer inputs, scoring and selection for CPU replay analysis."""

from __future__ import annotations

import functools
import importlib
import json
from pathlib import Path
from typing import Any

from .decode_boundary_capture import BoundaryCapture, tensor_leaves
from .drafter_context_probe import install as install_context


class SparsePrefillCapture(BoundaryCapture):
    """Observe only the first differing indexer, leaving its arithmetic and outputs untouched."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.saved: dict[str, Any] = {}
        super().__init__(self.enabled, rows=3973, layers=(3,), max_bytes=512 * 1024**2)

    def enabled(self) -> bool:
        """Capture only explicitly marked requests after startup and graph warm-up."""
        return bool(json.loads((self.root / "probe-marker.json").read_bytes())["active"])

    def wrapper(self, prefix: str, **options: Any) -> Any:
        """Assign independent storage to each selected layer-three prefill."""
        decorate = super().wrapper(prefix, **options)
        if options.get("layer") != 3:
            return decorate

        def prepare(original: Any) -> Any:
            wrapped = decorate(original)

            @functools.wraps(original)
            def call(*args: Any, **kwargs: Any) -> Any:
                positions = args[0] if args else kwargs.get("positions")
                if positions is not None and positions.numel() == self.rows and self.enabled():
                    self.events, self.counts, self.locks, self.saved = [], {}, [], {}
                    self.nbytes = 0
                return wrapped(*args, **kwargs)
            return call
        return prepare

    def record(
        self, name: str, category: str, value: Any, parameters: dict[str, Any] | None = None,
    ) -> None:
        """Retain full indexer, score and top-k values, including masks needed to reject unused data."""
        import torch

        if not (".indexer.indexer_op#" in name or name.startswith("sparse_prefill.")):
            return
        fields = []
        for path, tensor in tensor_leaves(value):
            if not tensor.numel():
                continue
            if tensor.untyped_storage().data_ptr() in self._topk_storages:
                tensor = tensor[:self.rows]
            size = tensor.numel() * tensor.element_size()
            if self.nbytes + size > self.max_bytes:
                raise MemoryError("Sparse-prefill observations exceed their storage budget")
            key = f"sparse_prefill_{len(self.saved):04d}"
            self.saved[key] = tensor.detach().clone(memory_format=torch.contiguous_format)
            fields.append({"path": path, "key": key, "shape": list(tensor.shape),
                           "dtype": str(tensor.dtype), "bytes": size})
            self.nbytes += size
        if fields:
            self.events.append({"name": name, "category": category,
                                "tensors": fields, "parameters": dict(parameters or {})})

    def install_scoring(self, module: Any, ops: Any) -> None:
        """Tap actual gathered score inputs and the in-place native top-k output."""
        self.replace(module, "fp8_fp4_mqa_logits", self.wrapper("sparse_prefill.mqa_logits"))

        def decorate(original: Any) -> Any:
            @functools.wraps(original)
            def topk(*args: Any, **kwargs: Any) -> Any:
                if self.active_layer != 3:
                    return original(*args, **kwargs)
                if len(args) != 8 or kwargs:
                    raise ValueError("Native prefill top-k signature changed")
                prefix = "sparse_prefill.topk"
                count = self.counts.get(prefix, 0)
                self.counts[prefix] = count + 1
                name = prefix + f"#{count}"
                parameters = dict(zip(("rows", "stride0", "stride1", "k"), args[4:], strict=True))
                self.record(name + "/input", "operand", args[:3], parameters)
                result = original(*args, **kwargs)
                self.record(name + "/output", "result", args[3], parameters)
                return result
            return topk
        self.replace(ops, "top_k_per_row_prefill", decorate)

    def report(self) -> dict[str, Any]:
        """Require complete score/selection observations before attaching a frame."""
        mqa = self.counts.get("sparse_prefill.mqa_logits", 0)
        topk = self.counts.get("sparse_prefill.topk", 0)
        if not mqa or mqa != topk:
            raise ValueError("Missing native score or top-k observations")
        selected_rows = sum(e["parameters"]["rows"] for e in self.events
                            if e["name"].startswith("sparse_prefill.topk") and e["category"] == "result")
        if selected_rows != self.rows:
            raise ValueError("Native selection observations do not cover the complete prefill")
        return {"rows": self.rows, "layer": 3, "events": self.events, "captured_bytes": self.nbytes,
                "maximum_capture_bytes": self.max_bytes,
                "unused_score_columns_retained_for_masked_review_only": True,
                "arithmetic_replaced": False, "per_operation_CPU_copy": False,
                "stream_ordered_GPU_copies_may_alter_scheduling": True,
                "timing_claim": False}


def install(worker: Any) -> dict[str, Any]:
    """Attach native indexer observations to the existing accepted-history capture."""
    import torch
    from vllm import _custom_ops as ops

    report = install_context(worker)
    runner = worker.model_runner
    capture = runner.sample.__self__
    trace = SparsePrefillCapture(Path("/traces"))
    trace.install(runner.model, ops)
    trace.install_scoring(importlib.import_module(
        "vllm.model_executor.layers.sparse_attn_indexer_kpool"), torch.ops._C)
    original_sample = runner.sample

    @functools.wraps(original_sample)
    def sample(hidden: Any, batch: Any, grammar: Any) -> Any:
        result = original_sample(hidden, batch, grammar)
        frame = capture.pending
        if frame is not None and frame["metadata"]["index"] == 0:
            if batch.num_tokens != trace.rows:
                raise ValueError("Unexpected observed sparse-prefill geometry")
            frame["metadata"]["sparse_prefill"] = trace.report()
            if frame["tensors"].keys() & trace.saved.keys():
                raise ValueError("Sparse observation would overwrite an existing field")
            frame["tensors"].update(trace.saved)
        return result

    runner.sample = sample
    report["sparse_prefill"] = {"rows": trace.rows, "layer": 3,
                                "purpose": "Distinguish score, selected-set and ordering drift"}
    return report
