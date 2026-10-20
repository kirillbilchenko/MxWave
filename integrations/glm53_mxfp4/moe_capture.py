"""Bounded real-operand capture after all latency, quality and profile measurements."""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch


class LayerCapture:
    """Copy one real invocation per declared shape, filtering by exact layer weight storage."""

    def __init__(self, kernel: type, weight_pointer: int,
                 capturing: Callable[[], bool] | None = None) -> None:
        """Install only outside graph capture; retain no weights or persistent model views."""
        self.kernel = kernel
        self.original = kernel.apply
        signature = inspect.signature(self.original)
        if not {"hidden_states", "w1", "topk_ids", "topk_weights"} <= set(signature.parameters):
            raise ValueError("Pinned MoE call signature changed")
        self.rows: dict[int, dict[str, Any]] = {}
        self.nbytes = 0
        is_capturing = capturing or torch.cuda.is_current_stream_capturing

        @functools.wraps(self.original)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            bound = signature.bind(*args, **kwargs)
            values = bound.arguments
            x = values["hidden_states"]
            count = x.shape[0]
            selected = (values["w1"].data_ptr() == weight_pointer
                        and count in (1, 8, 16, 256) and count not in self.rows)
            if not selected:
                return self.original(*args, **kwargs)
            if is_capturing():
                raise RuntimeError("Operand capture is restricted to post-measurement eager calls")
            if x.shape[1] != 4096 or values["topk_ids"].shape != (count, 8):
                raise ValueError("Unexpected real GLM layer geometry")
            tensors = {key: values[key].detach().clone() for key in
                       ("hidden_states", "topk_ids", "topk_weights")}
            result = self.original(*args, **kwargs)
            if not isinstance(result, torch.Tensor) or result.shape != x.shape:
                raise ValueError("Unexpected routed-expert result contract")
            tensors["output"] = result.detach().clone()
            self.nbytes += sum(t.numel() * t.element_size() for t in tensors.values())
            if self.nbytes > 8 * 1024**2:
                raise MemoryError("Real MoE operand capture exceeded its 8 MiB bound")
            self.rows[count] = {
                "tensors": tensors,
                "activation": str(values.get("activation")),
                "global_num_experts": values.get("global_num_experts"),
                "expert_map_is_none": values.get("expert_map") is None,
                "apply_router_weight_on_input": values.get("apply_router_weight_on_input"),
                "implementation": type(args[0]).__module__ + "." + type(args[0]).__name__,
            }
            return result

        self.wrapper = wrapped
        kernel.apply = wrapped

    def restore(self) -> None:
        """Remove the exact owned wrapper before writing evidence or reporting errors."""
        if self.kernel.apply is self.wrapper:
            self.kernel.apply = self.original
        elif self.kernel.apply is not self.original:
            raise RuntimeError("Another owner changed the captured MoE method")

    def retain(self, root: Path) -> dict[str, Any]:
        """Write raw tensor bytes and hashes without pickled tensors or retained GPU storage."""
        self.restore()
        root.mkdir(parents=True, exist_ok=False)
        rows = []
        for count, row in sorted(self.rows.items()):
            record = {k: v for k, v in row.items() if k != "tensors"}
            record.update(tokens=count, tensors={})
            for key, tensor in row["tensors"].items():
                cpu = tensor.detach().cpu().contiguous()
                raw = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
                name = f"m{count}-{key}.bin"
                (root / name).write_bytes(raw)
                record["tensors"][key] = {
                    "path": name, "shape": list(cpu.shape), "dtype": str(cpu.dtype),
                    "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                }
            rows.append(record)
        result = {"status": "complete" if set(self.rows) == {1, 8, 16, 256} else "incomplete",
                  "layer": 23, "rows": rows, "payload_bytes": self.nbytes,
                  "mode": "actual eager forwards after all measured work",
                  "graph_operands_claimed": False, "python_wrapper_removed": True}
        (root / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
        self.rows.clear()
        return result
