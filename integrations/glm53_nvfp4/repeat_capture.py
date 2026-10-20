"""Bounded, opt-in GPU snapshots for original-checkpoint prefill diagnostics."""
from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def tensor_leaves(value: Any, prefix: str = "") -> list[tuple[str, torch.Tensor]]:
    """Enumerate tensor leaves without reading device values."""
    if isinstance(value, torch.Tensor):
        return [(prefix, value)]
    if isinstance(value, (tuple, list)):
        return [item for i, child in enumerate(value)
                for item in tensor_leaves(child, f"{prefix}/{i}")]
    if isinstance(value, dict):
        return [item for key, child in value.items()
                for item in tensor_leaves(child, f"{prefix}/{key}")]
    return []


class PrefillCapture:
    """Clone producer-stream values; perform all CPU inspection after the request."""

    def __init__(self, runner: Any, maximum_bytes: int = 768 * 1024**2) -> None:
        self.runner = runner
        self.maximum_bytes = maximum_bytes
        self.enabled = False
        self.active = False
        self.pending_logits = False
        self.rows = 256
        self.focus = -1
        self.label = ""
        self.used = 0
        self.forwards = 0
        self.events: list[dict[str, Any]] = []
        self.tensors: list[torch.Tensor] = []
        self.calls: list[dict[str, Any]] = []
        language = getattr(runner.model, "language_model", runner.model)
        model = language.model
        original = model.forward

        @functools.wraps(original)
        def forward(*args: Any, **kwargs: Any) -> Any:
            ids = kwargs.get("input_ids", args[0] if args else None)
            embeds = kwargs.get("inputs_embeds", args[3] if len(args) > 3 else None)
            rows = (ids.numel() if isinstance(ids, torch.Tensor) else
                    embeds.shape[0] if isinstance(embeds, torch.Tensor) else None)
            if self.enabled:
                self.calls.append({"rows": rows, "precomputed_embeddings": embeds is not None,
                                   "has_token_ids": ids is not None})
            selected = self.enabled and rows == self.rows
            if not selected:
                return original(*args, **kwargs)
            if self.active or self.forwards:
                raise RuntimeError("Expected exactly one selected prefill per capture")
            self.active = True
            self.forwards += 1
            try:
                self.record("model/input", {"args": args, "kwargs": kwargs})
                result = original(*args, **kwargs)
                self.record("model/output", result)
                self.pending_logits = True
                return result
            finally:
                self.active = False

        model.forward = forward
        self.wrap(model.embed_tokens, "forward", "embedding")
        for index, layer in enumerate(model.layers):
            self.wrap(layer, "forward", f"layer-{index}")
            self.wrap(layer.self_attn, "forward", f"layer-{index}/attention", index)
            self.wrap(layer.mlp, "forward", f"layer-{index}/mlp", index)
            for method in ("hc_pre", "hc_post", "hc_fused_post_pre"):
                self.wrap(layer, method, f"layer-{index}/{method}", index)
            # Immediate projection/norm/attention children provide a second level
            # without cloning any weights or touching the operation's arguments.
            for name, child in layer.self_attn.named_children():
                self.wrap(child, "forward", f"layer-{index}/attention/{name}", index)
            for name in ("gate", "experts", "shared_experts"):
                child = getattr(layer.mlp, name, None)
                if child is not None:
                    self.wrap(child, "forward", f"layer-{index}/mlp/{name}", index)
        self.wrap(model.norm, "forward", "final-norm")
        original_logits = runner.model.compute_logits

        @functools.wraps(original_logits)
        def logits(*args: Any, **kwargs: Any) -> Any:
            selected = self.enabled and self.pending_logits
            if selected:
                self.record("logits/input", {"args": args, "kwargs": kwargs})
            result = original_logits(*args, **kwargs)
            if selected:
                self.record("logits/output", result)
                self.pending_logits = False
            return result

        runner.model.compute_logits = logits

    def wrap(self, owner: Any, method: str, label: str, focus: int = -1) -> None:
        """Observe outputs, and focused inputs, without changing call semantics."""
        original = getattr(owner, method)

        @functools.wraps(original)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            selected = self.active and (focus == -1 or focus == self.focus)
            # Multimodal-capable runners embed text before entering model.forward,
            # then pass input_ids=None and inputs_embeds. Observe that boundary too.
            if label == "embedding" and self.enabled:
                ids = args[0] if args else kwargs.get("input_ids")
                selected = isinstance(ids, torch.Tensor) and ids.numel() == self.rows
            if selected and (focus != -1 or label == "embedding"):
                self.record(label + "/input", {"args": args, "kwargs": kwargs})
            result = original(*args, **kwargs)
            if selected:
                self.record(label + "/output", result)
            return result

        setattr(owner, method, wrapped)

    def record(self, label: str, value: Any) -> None:
        """Retain bounded detached clones, never synchronizing to inspect values."""
        for path, tensor in tensor_leaves(value):
            size = tensor.numel() * tensor.element_size()
            if self.used + size > self.maximum_bytes:
                raise RuntimeError("Prefill snapshot allocation bound exceeded")
            self.tensors.append(tensor.detach().clone())
            self.events.append({"label": label + path, "shape": list(tensor.shape),
                                    "stride": list(tensor.stride()), "dtype": str(tensor.dtype),
                                    "bytes": size})
            self.used += size

    def arm(self, label: str, focus: int = -1, rows: int = 256) -> dict[str, Any]:
        """Start only between requests; require a safe fresh filename label."""
        if self.enabled or self.active or not label or not label.replace("-", "").isalnum():
            raise ValueError("Capture already armed, or invalid label")
        if focus < -1 or focus >= 45 or rows != 256:
            raise ValueError("This diagnostic is bounded to a single 256-token prefill")
        self.events, self.tensors = [], []
        self.calls = []
        self.used = self.forwards = 0
        self.label, self.focus, self.rows = label, focus, rows
        self.enabled = True
        self.pending_logits = False
        return {"enabled": True, "label": label, "focus": focus}

    def finish(self, directory: Path) -> dict[str, Any]:
        """Persist exact bytes and metadata after the completed request."""
        if not self.enabled or self.active:
            raise RuntimeError("Capture must finish outside the model forward")
        self.enabled = False
        directory.mkdir(parents=True, exist_ok=False)
        for i, (event, tensor) in enumerate(zip(self.events, self.tensors, strict=True)):
            raw = tensor.contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
            filename = f"{i:04d}.bin"
            (directory / filename).write_bytes(raw)
            event.update(file=filename, sha256=hashlib.sha256(raw).hexdigest())
        report = {"label": self.label, "focus": self.focus, "rows": self.rows,
                      "forwards": self.forwards, "bytes": self.used, "events": self.events,
                      "producer_stream_clones": True, "CPU_sync_during_forward": False,
                      "intrusive_observer": True, "timing_qualification": False,
                      "logits_captured": any(e["label"] == "logits/output" for e in self.events)}
        report["observed_model_calls"] = self.calls
        (directory / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        self.tensors = []
        return report
