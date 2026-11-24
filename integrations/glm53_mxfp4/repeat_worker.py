"""Bounded, opt-in snapshots of the first varying reference MoE and fixed-input replays.

These disposable-worker probes are intrusive. They neither change quality thresholds
nor provide throughput measurements. GPU clones defer CPU synchronization until disarm.
"""

from __future__ import annotations

import functools
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import torch
from quality_worker import QualityWorker, runtime_metadata


def tensor_leaves(value: Any, prefix: str = "") -> list[tuple[str, torch.Tensor]]:
    """Find runtime tensors in containers without walking modules or parameter objects."""
    if isinstance(value, torch.Tensor):
        return [(prefix, value)]
    if isinstance(value, (tuple, list)):
        return [leaf for i, child in enumerate(value)
                for leaf in tensor_leaves(child, f"{prefix}/{i}")]
    if isinstance(value, dict):
        return [leaf for key, child in value.items()
                for leaf in tensor_leaves(child, f"{prefix}/{key}")]
    return []


class Snapshots:
    """Bound deferred activation clones; report overflow instead of killing model inference."""

    def __init__(self, maximum: int = 512 * 1024**2) -> None:
        self.maximum = maximum
        self.nbytes = 0
        self.events: list[dict[str, Any]] = []
        self.error: str | None = None

    def record(self, name: str, value: Any) -> None:
        """Clone values on their producing stream without synchronizing the CPU."""
        if self.error is not None:
            return
        leaves = tensor_leaves(value)
        size = sum(t.numel() * t.element_size() for _, t in leaves)
        if self.nbytes + size > self.maximum or len(self.events) >= 300:
            self.error = "Snapshot memory/event bound exceeded; evidence is incomplete"
            return
        values = [
            {"path": path, "shape": list(t.shape), "dtype": str(t.dtype),
             "snapshot": t.detach().clone(memory_format=torch.contiguous_format)}
            for path, t in leaves
        ]
        self.nbytes += size
        self.events.append({"name": name, "tensors": values})

    def dump(self, root: Path) -> dict[str, Any]:
        """Write exact bytes and hashes to a fresh owned directory after GPU synchronization."""
        root.mkdir(parents=True, exist_ok=False)
        events = []
        for i, event in enumerate(self.events):
            values = []
            for j, original in enumerate(event["tensors"]):
                item = dict(original)
                tensor = item.pop("snapshot")
                raw = tensor.reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
                name = f"event-{i:03d}-tensor-{j:02d}.bin"
                (root / name).write_bytes(raw)
                item.update(file=name, nbytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
                values.append(item)
            events.append({"name": event["name"], "tensors": values})
        report = {"status": "failed" if self.error else "complete", "error": self.error,
                  "retained_gpu_bytes": self.nbytes, "events": events,
                  "intrusive": True, "cpu_sync_during_forward": False}
        (root / "snapshots.json").write_text(json.dumps(report, indent=2) + "\n")
        return report


def difference(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    """Measure byte and numerical differences on CPU, including non-finite values."""
    a, b = left.detach().contiguous().cpu(), right.detach().contiguous().cpu()
    if a.shape != b.shape or a.dtype != b.dtype:
        raise ValueError("Comparison shape/dtype changed")
    byte_count = int(torch.count_nonzero(a.view(torch.uint8) != b.view(torch.uint8)))
    delta = a.float() - b.float()
    return {"bits_exact": byte_count == 0, "differing_bytes": byte_count,
            "different_values": int(torch.count_nonzero(a != b)),
            "all_finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
            "max_abs": float(delta.abs().max()) if delta.numel() else 0.0,
            "rmse": float(delta.square().mean().sqrt()) if delta.numel() else 0.0}


class RepeatWorker(QualityWorker):
    """Inspect only layers 3–5 and replay actual captured Marlin inputs in the same load."""

    def mxwave_repeat_policy(self, enabled: bool) -> dict[str, Any]:
        """Switch PyTorch's strict policy; native/custom kernels may not implement this policy."""
        if type(enabled) is not bool:
            raise ValueError("Require an explicit boolean policy")
        torch.use_deterministic_algorithms(enabled)
        return {"status": "complete", **runtime_metadata(
            self.model_runner.cudagraph_manager._graphs_captured)}

    def mxwave_repeat_capture(self, label: str, enabled: bool) -> dict[str, Any]:
        """Arm bounded hooks, or remove every hook before retaining exact activation fixtures."""
        from vllm import _custom_ops as ops
        from vllm.distributed import get_tensor_model_parallel_rank

        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,40}", label):
            raise ValueError("Use a bounded artifact label")
        rank = get_tensor_model_parallel_rank()
        if not enabled:
            if label != getattr(self, "_repeat_label", None):
                raise ValueError("Disarm label does not match the active capture")
            for handle in self._repeat_handles:
                handle.remove()
            self._repeat_handles = []
            for owner, name, previous, had_instance_value in reversed(self._repeat_methods):
                if had_instance_value:
                    setattr(owner, name, previous)
                else:
                    delattr(owner, name)
            self._repeat_methods = []
            torch.cuda.synchronize()
            root = Path("/traces") / f"repeat-{label}-rank{rank}"
            report = self._repeat_snapshots.dump(root)
            self._repeat_snapshots = None
            self._repeat_label = None
            return {"status": report["status"], "rank": rank, "root": str(root),
                    "events": len(report["events"]), "bytes": report["retained_gpu_bytes"],
                    "replay_cases": len(self._repeat_replays), "error": report["error"]}
        if getattr(self, "_repeat_label", None) is not None:
            raise ValueError("Remove the previous capture before arming another")
        if self.model_runner.cudagraph_manager._graphs_captured:
            raise ValueError("Snapshot hooks require actual eager dispatch")
        self._repeat_label = label
        self._repeat_snapshots = Snapshots()
        self._repeat_handles: list[Any] = []
        self._repeat_methods: list[tuple[Any, str, Any, bool]] = []
        self._repeat_replays: list[dict[str, Any]] = []
        self._repeat_replay_bytes = 0
        self._repeat_active: str | None = None
        snapshots = self._repeat_snapshots

        def prefill(value: Any) -> bool:
            return any(t.ndim > 0 and t.shape[0] == 258 for _, t in tensor_leaves(value))

        def replace(owner: Any, name: str, function: Any) -> None:
            previous = getattr(owner, name)
            self._repeat_methods.append((owner, name, previous, name in vars(owner)))
            setattr(owner, name, function(previous))

        def wrapper(name: str, *, first_four: bool = False, active: bool = False) -> Any:
            def decorate(original: Any) -> Any:
                @functools.wraps(original)
                def call(*args: Any, **kwargs: Any) -> Any:
                    selected = args[:4] if first_four else (args, kwargs)
                    if not prefill(selected):
                        return original(*args, **kwargs)
                    previous = self._repeat_active
                    if active:
                        self._repeat_active = name
                    try:
                        snapshots.record(name + "/input", selected)
                        output = original(*args, **kwargs)
                        snapshots.record(name + "/output", output)
                        return output
                    finally:
                        self._repeat_active = previous
                return call
            return decorate

        selected_layers = []
        for name, layer in self.model_runner.model.named_modules():
            match = re.search(r"\.layers\.([345])$", name)
            if match is None:
                continue
            selected_layers.append(int(match[1]))
            replace(layer, "forward", wrapper(name))
            replace(layer.self_attn, "forward", wrapper(name + ".self_attn"))
            replace(layer, "hc_fused_post_pre", wrapper(name + ".hc_fused_post_pre",
                                                        first_four=True))
            mlp = layer.mlp
            replace(mlp, "forward", wrapper(name + ".mlp", active=True))
            replace(mlp.gate, "forward", wrapper(name + ".mlp.gate"))
            runner = mlp.experts
            replace(runner.router, "select_experts", wrapper(name + ".mlp.router"))
            replace(runner, "_apply_quant_method", wrapper(name + ".mlp.quant_method"))
            replace(runner, "_maybe_reduce_final_output", wrapper(name + ".mlp.final_reduce"))
            if mlp.shared_experts is not None:
                for child, module in mlp.shared_experts.named_modules():
                    replace(module, "forward", wrapper(name + ".mlp.shared_experts" +
                                                       ("." + child if child else "")))
        if sorted(selected_layers) != [3, 4, 5]:
            raise RuntimeError("Expected the three bounded target MoE layers")

        def capture_gemm(original: Any) -> Any:
            @functools.wraps(original)
            def gemm(*args: Any, **kwargs: Any) -> Any:
                if self._repeat_active is None:
                    return original(*args, **kwargs)
                index = len(self._repeat_replays)
                if index >= 6:
                    snapshots.error = "Unexpected additional GEMM; replay bound exceeded"
                    return original(*args, **kwargs)
                name = self._repeat_active + f"/gemm{index}"
                # Weight/scales stay owned by the loaded model; snapshot runtime inputs only.
                selected = {str(i): args[i] for i in (0, 10, 11, 12, 13, 14)}
                snapshots.record(name + "/input", selected)
                extra = sum(t.numel() * t.element_size() for _, t in tensor_leaves(selected))
                extra += args[1].numel() * args[1].element_size()
                if self._repeat_replay_bytes + extra > 128 * 1024**2:
                    snapshots.error = "Replay fixture memory bound exceeded"
                    return original(*args, **kwargs)
                self._repeat_replay_bytes += extra
                saved = list(args)
                for i in (0, 10, 11, 12, 13, 14):
                    if isinstance(saved[i], torch.Tensor):
                        saved[i] = saved[i].detach().clone()
                output = original(*args, **kwargs)
                snapshots.record(name + "/output", output)
                self._repeat_replays.append({"name": name, "args": saved, "kwargs": kwargs,
                                             "output": output.detach().clone(),
                                             "original": original})
                return output
            return gemm
        replace(ops, "moe_wna16_marlin_gemm", capture_gemm)
        return {"status": "armed", "rank": rank, "layers": selected_layers,
                "maximum_snapshot_bytes": snapshots.maximum,
                "maximum_replay_bytes": 128 * 1024**2,
                "intrusive": True, "cpu_sync_during_forward": False}

    def mxwave_repeat_replay(self) -> dict[str, Any]:
        """Replay six captured GEMMs with fixed inputs under both deterministic policies."""
        from vllm.distributed import get_tensor_model_parallel_rank

        cases = getattr(self, "_repeat_replays", [])
        if len(cases) != 6 or getattr(self, "_repeat_label", None) is not None:
            raise ValueError("Disarm a complete three-layer/two-GEMM capture before replay")
        previous = torch.are_deterministic_algorithms_enabled()
        previous_warn = torch.is_deterministic_algorithms_warn_only_enabled()
        rows = []
        try:
            with torch.inference_mode():
                for enabled in (False, True):
                    torch.use_deterministic_algorithms(enabled)
                    for case in cases:
                        reference = case["output"].cpu()
                        values = []
                        for _ in range(5):
                            args = list(case["args"])
                            args[1] = torch.empty_like(case["output"])
                            # Locks are mutable even on the non-atomic FP32-reduce path.
                            args[10] = case["args"][10].clone()
                            try:
                                result = case["original"](*args, **case["kwargs"])
                                values.append(difference(reference, result))
                            except Exception as exc:  # noqa: BLE001 -- retain unsupported probes.
                                values.append({"error": repr(exc)})
                                break
                        rows.append({"name": case["name"], "deterministic": enabled,
                                     "parameters": {k: v if type(v) in (str, bool, int, float)
                                                    else str(v) for k, v in case["kwargs"].items()},
                                     "repeats": values})
        finally:
            torch.use_deterministic_algorithms(previous, warn_only=previous_warn)
            self._repeat_replays = []
        return {"status": "complete", "rank": get_tensor_model_parallel_rank(),
                "cases": rows, "input_scope": "Identical captured activations, routing, padding, "
                "locks and loaded weights; no forward context or TP reduction in GEMM replay"}
