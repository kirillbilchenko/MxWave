"""Hash exact early-layer prefill operands/results without changing model arithmetic."""

from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path
from typing import Any

from .decode_boundary_capture import BoundaryCapture, tensor_leaves
from .drafter_context_probe import cpu_copy
from .drafter_context_probe import install as install_context


class PrefillHashTrace(BoundaryCapture):
    """Bound one eager prefill trace; keep digests instead of resident activation copies."""

    def __init__(self, root: Path, rows: int) -> None:
        self.root = root
        self.layer_prefixes: dict[int, str] = {}
        super().__init__(self.enabled, rows=rows, layers=(0, 1, 2, 3, 4, 5),
                         max_bytes=64 * 1024**3)

    def enabled(self) -> bool:
        """Only instrument requests explicitly marked for capture, after startup."""
        return bool(json.loads((self.root / "probe-marker.json").read_bytes())["active"])

    def wrapper(self, prefix: str, **options: Any) -> Any:
        """Reset digest ownership when the selected request enters layer zero."""
        decorate = super().wrapper(prefix, **options)
        if options.get("layer") is not None:
            self.layer_prefixes[options["layer"]] = prefix
        if options.get("layer") != 0:
            return decorate

        def prepare(original: Any) -> Any:
            wrapped = decorate(original)

            @functools.wraps(original)
            def call(*args: Any, **kwargs: Any) -> Any:
                positions = args[0] if args else kwargs.get("positions")
                if positions is not None and positions.numel() == self.rows and self.enabled():
                    self.events, self.counts, self.locks = [], {}, []
                    self.nbytes = 0
                return wrapped(*args, **kwargs)
            return call
        return prepare

    def record(
        self, name: str, category: str, value: Any, parameters: dict[str, Any] | None = None,
    ) -> None:
        """Hash complete logical operands; exclude unused routing tails and scratch contents."""
        import torch

        if category == "workspace":
            return
        params = dict(parameters or {})
        if isinstance(value, dict) and {"11", "12", "13"} <= value.keys():
            valid = int(value["13"].item())
            block = int(params["moe_block_size"])
            if not 0 <= valid <= value["11"].numel() or valid % block:
                raise ValueError("Invalid consumed expert routing extent")
            value = {**value, "11": value["11"][:valid], "12": value["12"][:valid // block]}
            params["consumed_routing_tokens"] = valid
        fields = []
        for path, tensor in tensor_leaves(value):
            if not tensor.numel():
                continue
            if tensor.untyped_storage().data_ptr() in self._topk_storages:
                tensor = tensor[:self.rows]
            size = tensor.numel() * tensor.element_size()
            if self.nbytes + size > self.max_bytes:
                raise MemoryError("Prefill digest trace exceeded its declared byte budget")
            # A synchronous observation may alter overlap, but never replaces an operand.
            # Native before/after controls and this limitation are retained in the plan.
            copied = cpu_copy(tensor)
            raw = copied.reshape(-1).view(torch.uint8).numpy()
            sha = hashlib.sha256(memoryview(raw).cast("B")).hexdigest()
            fields.append({"path": path, "shape": list(tensor.shape), "dtype": str(tensor.dtype),
                           "bytes": size, "sha256": sha})
            self.nbytes += size
        if fields:
            self.events.append({"name": name, "category": category, "tensors": fields,
                                "parameters": params})

    def digest_report(self) -> dict[str, Any]:
        """Return complete per-operation digest metadata after one captured prefill."""
        if (set(self.layer_prefixes) != set(self.layers)
                or any(self.counts.get(name) != 1 for name in self.layer_prefixes.values())
                or not self.events):
            raise ValueError("Early prefill layers did not execute exactly once under observation")
        return {"logical_rows": self.rows, "layers": list(self.layers), "events": self.events,
                "hashed_bytes": self.nbytes, "maximum_hashed_bytes": self.max_bytes,
                "full_tensors_hashed": True, "unused_routing_tails_excluded": True,
                "scratch_contents_excluded": True, "CPU_observation_synchronizes": True,
                "timing_claim": False}


def install(worker: Any) -> dict[str, Any]:
    """Attach digest observations to the existing feature probe after native graph warm-up."""
    from vllm import _custom_ops as ops

    report = install_context(worker)
    runner = worker.model_runner
    capture = runner.sample.__self__
    plan = json.loads(Path("/trial/plan.json").read_bytes())
    trace = PrefillHashTrace(Path("/traces"), plan["prefill_boundary_rows"])
    trace.install(runner.model, ops)
    original_sample = runner.sample

    @functools.wraps(original_sample)
    def sample(hidden: Any, batch: Any, grammar: Any) -> Any:
        result = original_sample(hidden, batch, grammar)
        frame = capture.pending
        if frame is not None and frame["metadata"]["index"] == 0:
            if batch.num_tokens != trace.rows:
                raise ValueError("Unexpected captured prefill geometry")
            frame["metadata"]["prefill_boundaries"] = trace.digest_report()
        return result

    runner.sample = sample
    report["prefill_boundaries"] = {
        "logical_rows": trace.rows, "layers": list(trace.layers),
        "method": "SHA256 of complete CPU-copied operand/result bits; no arithmetic replacement",
        "scope": "Eager marked prefill only; decode graphs are unchanged",
    }
    return report
