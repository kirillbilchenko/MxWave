"""Bounded internal MoE snapshots and fixed-operand replays for disposable workers."""
from __future__ import annotations

import functools
import inspect
import json
from pathlib import Path
from typing import Any

import torch
from repeat_capture import PrefillCapture, tensor_leaves


def clone_tree(value: Any) -> Any:
    """Freeze runtime operands without traversing model objects."""
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(clone_tree(x) for x in value)
    if isinstance(value, list):
        return [clone_tree(x) for x in value]
    if isinstance(value, dict):
        return {k: clone_tree(v) for k, v in value.items()}
    return value


def compare(left: Any, right: Any) -> list[dict[str, Any]]:
    """Compare immutable outputs after GPU work completes."""
    a, b = tensor_leaves(left), tensor_leaves(right)
    if len(a) != len(b):
        raise ValueError("Replay changed its output structure")
    rows = []
    for (path, x), (other, y) in zip(a, b, strict=True):
        if path != other or x.shape != y.shape or x.dtype != y.dtype:
            raise ValueError("Replay changed its output contract")
        x, y = x.contiguous().cpu(), y.contiguous().cpu()
        delta = x.double() - y.double()
        rows.append({"path": path, "shape": list(x.shape), "dtype": str(x.dtype),
                     "bits_exact": bool(torch.equal(x.view(torch.uint8), y.view(torch.uint8))),
                     "different_values": int(torch.count_nonzero(x != y)),
                     "max_abs": float(delta.abs().max()) if delta.numel() else 0.0,
                     "rmse": float(delta.square().mean().sqrt()) if delta.numel() else 0.0})
    return rows


class MoeCapture(PrefillCapture):
    """Observe layers 3–5; retain runtime fixtures while keeping weights model-owned."""

    def __init__(self, runner: Any) -> None:
        super().__init__(runner, maximum_bytes=512 * 1024**2)
        from vllm import _custom_ops as ops
        from vllm.model_executor.layers.fused_moe.experts import marlin_moe

        self.current_layer: int | None = None
        self.cases: list[dict[str, Any]] = []
        self.replay_bytes = 0
        self.counts: dict[str, int] = {}
        language = getattr(runner.model, "language_model", runner.model)
        for index in (3, 4, 5):
            layer = language.model.layers[index]
            moe = layer.mlp.experts
            original = moe._apply_quant_method

            def context(original: Any, index: int) -> Any:
                @functools.wraps(original)
                def call(*args: Any, **kwargs: Any) -> Any:
                    if not self.active:
                        return original(*args, **kwargs)
                    previous, self.current_layer = self.current_layer, index
                    try:
                        self.record(f"moe-{index}/quant/input", (args, kwargs))
                        output = original(*args, **kwargs)
                        self.record(f"moe-{index}/quant/output", output)
                        return output
                    finally:
                        self.current_layer = previous
                return call

            moe._apply_quant_method = context(original, index)
            self.observe(moe.router, "select_experts", "routing", layer=index)
            self.observe(moe, "_maybe_reduce_final_output", "final-reduce", layer=index)
        self.observe(marlin_moe, "moe_align_block_size", "alignment")
        self.observe(ops, "moe_wna16_marlin_gemm", "gemm", positional=(0, 10, 11, 12, 13, 14))
        self.observe(ops, "moe_sum", "sum", positional=(0, 2, 3), output_position=1)
        self.observe(marlin_moe, "fused_marlin_moe", "fixed-routing",
                     named=("hidden_states", "topk_weights", "topk_ids", "workspace"))

    def record(self, label: str, value: Any) -> None:
        """Keep early layer outputs and logits, omitting redundant later snapshots."""
        if label.startswith("layer-") and int(label.split("/")[0].split("-")[1]) > 5:
            return
        super().record(label, value)

    def arm(self, label: str, focus: int = -1, rows: int = 256) -> dict[str, Any]:
        """Release previous fixtures before arming a fresh bounded capture."""
        result = super().arm(label, focus, rows)
        self.cases, self.counts = [], {}
        self.replay_bytes = 0
        return result

    def observe(self, owner: Any, method: str, kind: str, *, layer: int | None = None,
                positional: tuple[int, ...] | None = None,
                named: tuple[str, ...] | None = None,
                output_position: int | None = None) -> None:
        """Snapshot runtime arguments and outputs without changing their order or values."""
        original = getattr(owner, method)

        @functools.wraps(original)
        def call(*args: Any, **kwargs: Any) -> Any:
            index = self.current_layer if layer is None else layer
            if not self.active or index is None:
                return original(*args, **kwargs)
            key = f"moe-{index}/{kind}"
            count = self.counts.get(key, 0)
            self.counts[key] = count + 1
            name = f"{key}-{count}"
            if len(self.cases) >= 36:
                raise RuntimeError("MoE replay case bound exceeded")
            if named is not None:
                bound = inspect.signature(original).bind(*args, **kwargs)
                bound.apply_defaults()
                saved_kwargs = dict(bound.arguments)
                selected = {k: saved_kwargs[k] for k in named}
                for k, v in selected.items():
                    saved_kwargs[k] = clone_tree(v)
                # These buffers are mutable scratch, not model parameters.
                for k in ("intermediate_cache13", "intermediate_cache2", "output"):
                    value = saved_kwargs[k]
                    if value is not None:
                        saved_kwargs[k] = torch.empty_like(value)
                saved_args = []
            elif positional is not None:
                selected = {str(i): args[i] for i in positional if i < len(args)}
                saved_args, saved_kwargs = list(args), dict(kwargs)
                for i in positional:
                    if i < len(args):
                        saved_args[i] = clone_tree(args[i])
                if kind == "gemm":
                    saved_args[1] = torch.empty_like(args[1])
                elif output_position is not None:
                    saved_args[output_position] = torch.empty_like(args[output_position])
            else:
                selected = (args, kwargs)
                saved_args, saved_kwargs = clone_tree(args), clone_tree(kwargs)
            self.record(name + "/input", selected)
            # Count only newly owned runtime allocations, never loaded weights.
            extra = sum(t.numel() * t.element_size() for _, t in tensor_leaves(selected))
            if named is not None:
                extra += sum(t.numel() * t.element_size() for k in
                             ("intermediate_cache13", "intermediate_cache2", "output")
                             for _, t in tensor_leaves(saved_kwargs[k]))
            elif positional is not None:
                extra += args[1].numel() * args[1].element_size()
            output = original(*args, **kwargs)
            observed = args[output_position] if output_position is not None else output
            if kind == "fixed-routing" and output is None:
                observed = kwargs["output"]
            self.record(name + "/output", observed)
            expected = clone_tree(observed)
            extra += sum(t.numel() * t.element_size() for _, t in tensor_leaves(expected))
            self.replay_bytes += extra
            if self.replay_bytes > 384 * 1024**2:
                raise RuntimeError("MoE replay fixture allocation bound exceeded")
            self.cases.append({"name": name, "kind": kind, "args": saved_args,
                               "kwargs": saved_kwargs, "expected": expected,
                               "original": original, "output_position": output_position})
            return output

        setattr(owner, method, call)

    def replay(self, directory: Path, repeats: int = 20) -> dict[str, Any]:
        """Replay fixed routing, each GEMM and collective with immutable runtime inputs."""
        if self.enabled or self.active or not self.cases or not 1 <= repeats <= 32:
            raise ValueError("Replay requires a completed capture and 1–32 repeats")
        results = []
        with torch.inference_mode():
            for case in self.cases:
                values = []
                for _ in range(repeats):
                    args, kwargs = list(case["args"]), dict(case["kwargs"])
                    kind = case["kind"]
                    if kind == "gemm":
                        args[1], args[10] = torch.empty_like(args[1]), args[10].clone()
                    elif kind == "fixed-routing":
                        for key in ("workspace", "hidden_states", "topk_weights", "topk_ids"):
                            kwargs[key] = clone_tree(kwargs[key])
                        for key in ("intermediate_cache13", "intermediate_cache2", "output"):
                            if kwargs[key] is not None:
                                kwargs[key] = torch.empty_like(kwargs[key])
                    else:
                        args, kwargs = clone_tree(args), clone_tree(kwargs)
                    output = case["original"](*args, **kwargs)
                    if case["output_position"] is not None:
                        output = args[case["output_position"]]
                    elif kind == "fixed-routing" and output is None:
                        output = kwargs["output"]
                    values.append(compare(case["expected"], output))
                results.append({"name": case["name"], "kind": case["kind"], "repeats": values})
        report = {"label": self.label, "cases": results, "fixture_bytes": self.replay_bytes,
                  "timing_qualification": False, "scope": "fixed captured operands; loaded weights"}
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "replay.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
