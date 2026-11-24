"""Insert bounded diagnostic copies at actual captured GLM decode boundaries."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import torch

if __package__:
    from .decode_state_values import value_metrics
else:
    from decode_state_values import value_metrics

MAX_CAPTURE_BYTES = 24 * 1024**2
_INSTALLED = False


def tensor_leaves(value: Any, path: str = "") -> list[tuple[str, torch.Tensor]]:
    """Flatten explicit argument containers without traversing modules or model weights."""
    if isinstance(value, torch.Tensor):
        return [(path, value)]
    if isinstance(value, (tuple, list)):
        return [leaf for i, child in enumerate(value)
                for leaf in tensor_leaves(child, path + "/" + str(i))]
    if isinstance(value, dict):
        return [leaf for key, child in value.items()
                for leaf in tensor_leaves(child, path + "/" + str(key))]
    return []


def boundary_metrics(current: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    """Measure BF16/F32 errors; preserve byte evidence for routing and other formats."""
    if current.dtype in (torch.bfloat16, torch.float32):
        return {"metric_kind": "numerical", **value_metrics(current, reference)}
    a = current.contiguous().reshape(-1).view(torch.uint8)
    b = reference.contiguous().reshape(-1).view(torch.uint8)
    changed = a != b
    return {"metric_kind": "bytes", "bytes_exact": not bool(changed.any()),
            "changed_bytes": int(changed.sum()), "elements": current.numel(),
            "changed_elements_by_bits": int(changed.reshape(-1, current.element_size()).any(1).sum())}


class BoundaryCapture:
    """Retain graph-written copies, rather than views of later-overwritten activations."""

    def __init__(self, capturing: Callable[[], bool] | None = None, *,
                 rows: int = 4, layers: tuple[int, ...] = (2, 3, 4),
                 max_bytes: int = MAX_CAPTURE_BYTES) -> None:
        """Bound diagnostic allocation before any measured graph replay."""
        self.capturing = capturing or torch.cuda.is_current_stream_capturing
        self.rows, self.layers, self.max_bytes = rows, layers, max_bytes
        self.active_layer: int | None = None
        self.events: list[dict[str, Any]] = []
        self.nbytes = 0
        self.counts: dict[str, int] = {}
        self.methods: list[tuple[Any, str, Any, bool]] = []
        self.locks: list[torch.Tensor] = []
        self._topk_storages: set[int] = set()
        self._wrapped: set[tuple[int, str]] = set()

    def record(self, name: str, category: str, value: Any,
               parameters: dict[str, Any] | None = None) -> None:
        """Capture stream-ordered copies whose contents update on every graph replay."""
        leaves = []
        for path, tensor in tensor_leaves(value):
            if not tensor.numel():
                continue
            if (tensor.untyped_storage().data_ptr() in self._topk_storages
                    and tensor.shape[0] > self.rows):
                tensor = tensor[:self.rows]
            leaves.append((path, tensor))
        size = sum(t.numel() * t.element_size() for _, t in leaves)
        if self.nbytes + size > self.max_bytes:
            raise MemoryError(f"Boundary diagnostic exceeds its {self.max_bytes}-byte budget")
        if not leaves:
            return
        copies = [{"path": path, "shape": list(t.shape), "dtype": str(t.dtype),
                   "value": t.detach().clone(memory_format=torch.contiguous_format)}
                  for path, t in leaves]
        self.nbytes += size
        self.events.append({"name": name, "category": category, "tensors": copies,
                            "parameters": parameters or {}})

    def replace(self, owner: Any, name: str, factory: Callable[[Any], Any]) -> None:
        """Wrap a method once and retain its exact instance/class lookup behavior."""
        key = (id(owner), name)
        if key in self._wrapped:
            return
        self._wrapped.add(key)
        previous = getattr(owner, name)
        self.methods.append((owner, name, previous, name in vars(owner)))
        setattr(owner, name, factory(previous))

    def wrapper(self, prefix: str, *, first_four: bool = False, first_args: int | None = None,
                layer: int | None = None, extra: Any = None) -> Callable[[Any], Any]:
        """Record runtime inputs/results only during the selected graph capture."""
        def decorate(original: Any) -> Any:
            @functools.wraps(original)
            def call(*args: Any, **kwargs: Any) -> Any:
                previous = self.active_layer
                if layer is not None:
                    positions = args[0] if args else kwargs.get("positions")
                    if (not isinstance(positions, torch.Tensor) or positions.numel() != self.rows
                            or not self.capturing()):
                        return original(*args, **kwargs)
                    self.active_layer = layer
                if self.active_layer is None:
                    return original(*args, **kwargs)
                count = self.counts.get(prefix, 0)
                self.counts[prefix] = count + 1
                name = prefix + f"#{count}"
                selected = args[:4] if first_four else (args, kwargs)
                if first_args is not None:
                    selected = args[:first_args]
                try:
                    self.record(name + "/input", "operand", selected)
                    output = original(*args, **kwargs)
                    self.record(name + "/output", "result", output)
                    buffer = getattr(extra, "topk_indices_buffer", None)
                    if isinstance(buffer, torch.Tensor):
                        self.record(name + "/topk", "result", buffer[:self.rows])
                    return output
                finally:
                    if layer is not None:
                        self.active_layer = previous
            return call
        return decorate

    def install(self, model: Any, ops: Any) -> None:
        """Wrap selected layers and their real operations without replacing arithmetic."""
        layers = []
        for name, module in model.named_modules():
            index = getattr(module, "layer_idx", None)
            if index not in self.layers or not hasattr(module, "hc_fused_post_pre"):
                continue
            layers.append(index)
            self.replace(module, "forward", self.wrapper(name, layer=index))
            self.replace(module, "hc_fused_post_pre",
                         self.wrapper(name + ".hc_fused_post_pre", first_four=True))
            self.replace(module, "hc_pre", self.wrapper(name + ".hc_pre", first_args=1))
            # Child module copies distinguish projections/indexing from whole attention.
            for path, child in module.self_attn.named_modules():
                label = name + ".self_attn" + ("." + path if path else "")
                topk = getattr(child, "topk_indices_buffer", None)
                if isinstance(topk, torch.Tensor):
                    self._topk_storages.add(topk.untyped_storage().data_ptr())
                self.replace(child, "forward", self.wrapper(label, extra=child))
            self.replace(module.mlp, "forward", self.wrapper(name + ".mlp"))
            if index in (3, 4):
                self.replace(module.mlp.gate, "forward", self.wrapper(name + ".mlp.gate"))
                experts = module.mlp.experts
                for method in ("_apply_quant_method", "_maybe_reduce_final_output"):
                    self.replace(experts, method, self.wrapper(name + ".mlp." + method))
                self.replace(experts.router, "select_experts",
                             self.wrapper(name + ".mlp.router"))
        if sorted(layers) != sorted(self.layers):
            raise ValueError(f"Require exactly GLM decoder layers {self.layers}")

        def decorate_gemm(original: Any) -> Any:
            @functools.wraps(original)
            def gemm(*args: Any, **kwargs: Any) -> Any:
                if self.active_layer not in (3, 4):
                    return original(*args, **kwargs)
                if len(args) != 15:
                    raise ValueError("Pinned MoE GEMM tensor signature changed")
                prefix = f"layer{self.active_layer}.moe_gemm"
                count = self.counts.get(prefix, 0)
                self.counts[prefix] = count + 1
                name = prefix + f"#{count}"
                self.locks.append(args[10])
                params = {k: v if type(v) in (str, int, float, bool) or v is None else str(v)
                          for k, v in kwargs.items()}
                self.record(name + "/input", "operand", {str(i): args[i]
                            for i in (0, 5, 6, 11, 12, 13, 14)}, params)
                self.record(name + "/locks", "workspace", args[10])
                self.record(name + "/initial_output", "workspace", args[1])
                result = original(*args, **kwargs)
                self.record(name + "/output", "result", result, params)
                return result
            return gemm
        self.replace(ops, "moe_wna16_marlin_gemm", decorate_gemm)

    def restore(self) -> None:
        """Remove every Python wrapper after capture; graph copies remain owned by this trace."""
        for owner, name, previous, had_instance in reversed(self.methods):
            if had_instance:
                setattr(owner, name, previous)
            else:
                delattr(owner, name)
        self.methods.clear()

    def metadata(self) -> dict[str, Any]:
        """Expose only bounded plain metadata and exact graph-copy layout."""
        lock_storages = {(t.untyped_storage().data_ptr(), t.untyped_storage().nbytes())
                         for t in self.locks}
        return {"capture_bytes": self.nbytes, "maximum_capture_bytes": self.max_bytes,
                "rows": self.rows, "layers": list(self.layers),
                "python_wrappers_removed": not self.methods,
                "stock_marlin_lock_storage_bytes": sum(size for _, size in lock_storages),
                "stock_marlin_lock_storages": len(lock_storages),
                "events": [{k: v for k, v in event.items() if k != "tensors"} | {
                    "tensors": [{k: v for k, v in tensor.items() if k != "value"}
                                for tensor in event["tensors"]]} for event in self.events]}


def patch_manager(manager_type: Any, ops: Any) -> None:
    """Instrument initial capture, leaving normal graph dispatch and arithmetic intact."""
    original = manager_type.capture

    @functools.wraps(original)
    def capture(manager: Any, model: Any, *args: Any, **kwargs: Any) -> Any:
        trace = BoundaryCapture()
        try:
            trace.install(model, ops)
            result = original(manager, model, *args, **kwargs)
        finally:
            trace.restore()
        descriptors = [d for d in manager.graphs if d.num_tokens == 4]
        if len(descriptors) != 1 or not trace.events or not trace.locks:
            raise ValueError("Require one actual four-token graph with recorded MoE boundaries")
        manager._mxwave_boundary_trace = trace
        return result

    manager_type.capture = capture


def install() -> None:
    """Install the capture-only adapter in this disposable diagnostic worker process."""
    global _INSTALLED
    if _INSTALLED:
        return
    from vllm import _custom_ops as ops
    from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager

    patch_manager(ModelCudaGraphManager, ops)
    _INSTALLED = True
