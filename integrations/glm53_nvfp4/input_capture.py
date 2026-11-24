"""Capture actual speculative target inputs and raw logits without altering graph dispatch."""
from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from repeat_capture import tensor_leaves


class InputCapture:
    """Clone prepared inputs, graph results and target logits; inspect only after requests."""

    def __init__(self, runner: Any, maximum_bytes: int = 256 * 1024**2) -> None:
        self.enabled = False
        self.label = ""
        self.step = -1
        self.used = 0
        self.maximum_bytes = maximum_bytes
        self.events: list[dict[str, Any]] = []
        self.tensors: list[torch.Tensor] = []
        self.batches: list[dict[str, Any]] = []
        prepare = runner.prepare_inputs

        @functools.wraps(prepare)
        def prepared(*args: Any, **kwargs: Any) -> Any:
            batch = prepare(*args, **kwargs)
            if self.enabled:
                self.step += 1
                if self.step >= 32:
                    raise RuntimeError("Target-input forward bound exceeded")
                self.batches.append({"step": self.step, "num_tokens": int(batch.num_tokens),
                                     "num_tokens_after_padding": int(batch.num_tokens_after_padding),
                                     "num_reqs": int(batch.num_reqs),
                                     "has_prefill": bool(batch.has_prefill)})
                for name in ("input_ids", "positions", "is_padding", "query_start_loc",
                             "seq_lens", "logits_indices", "cu_num_logits",
                             "num_draft_tokens_per_req", "expanded_idx_mapping",
                             "expanded_local_pos"):
                    value = getattr(batch, name, None)
                    if value is not None:
                        self.record("prepared/" + name, value)
            return batch

        runner.prepare_inputs = prepared
        self.observe(runner.model_state, "get_mm_embeddings", "embeddings", inputs=False)
        self.observe(runner.model_state, "prepare_inputs", "model-extra-inputs", inputs=False)
        self.observe(runner.cudagraph_manager, "run_fullgraph", "target-graph", inputs=False)
        self.observe(runner.model, "compute_logits", "target-logits", inputs=True)

    def observe(self, owner: Any, method: str, label: str, *, inputs: bool) -> None:
        """Wrap an existing call without changing its values or execution policy."""
        original = getattr(owner, method)

        @functools.wraps(original)
        def call(*args: Any, **kwargs: Any) -> Any:
            if self.enabled and inputs:
                self.record(label + "/input", (args, kwargs))
            result = original(*args, **kwargs)
            if self.enabled:
                self.record(label + "/output", result)
            return result

        setattr(owner, method, call)

    def record(self, label: str, value: Any) -> None:
        """Preserve exact producer-stream values with bounded allocations."""
        for path, tensor in tensor_leaves(value):
            size = tensor.numel() * tensor.element_size()
            if self.used + size > self.maximum_bytes:
                raise RuntimeError("Target-input snapshot allocation bound exceeded")
            self.tensors.append(tensor.detach().clone())
            self.events.append({"step": self.step, "label": label + path,
                                "shape": list(tensor.shape), "dtype": str(tensor.dtype),
                                "bytes": size})
            self.used += size

    def arm(self, label: str) -> dict[str, Any]:
        """Enable one bounded group between completed API requests."""
        if self.enabled or not label or not label.replace("-", "").isalnum():
            raise ValueError("Require a fresh safe capture label")
        self.label, self.enabled = label, True
        self.step, self.used = -1, 0
        self.events, self.tensors, self.batches = [], [], []
        return {"enabled": True, "label": label}

    def finish(self, directory: Path) -> dict[str, Any]:
        """Persist raw operands after all requests in the group finish."""
        if not self.enabled:
            raise ValueError("No input capture is armed")
        self.enabled = False
        directory.mkdir(parents=True, exist_ok=False)
        for i, (event, tensor) in enumerate(zip(self.events, self.tensors, strict=True)):
            raw = tensor.contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
            filename = f"{i:04d}.bin"
            (directory / filename).write_bytes(raw)
            event.update(file=filename, sha256=hashlib.sha256(raw).hexdigest())
        report = {"label": self.label, "events": self.events, "batches": self.batches,
                  "bytes": self.used, "CPU_sync_during_forward": False,
                  "intrusive_observer": True, "timing_qualification": False}
        (directory / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        self.tensors = []
        return report
