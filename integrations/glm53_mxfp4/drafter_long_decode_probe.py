"""Observe the first equal-input decode graph after repeatable canonical prefill."""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

from .drafter_component_probe import ComponentTrace, graph_metadata, patch_manager, selected_graph
from .drafter_context_probe import cpu_copy
from .drafter_context_probe import install as install_context


class LongDecodeTrace(ComponentTrace):
    """Retain early boundaries and native decode selection, excluding scratch contents."""

    def record(self, name: str, category: str, value: Any,
               parameters: dict[str, Any] | None = None) -> None:
        """Keep operation operands/results; workspace bytes cannot establish a causal drift."""
        if category != "workspace":
            super().record(name, category, value, parameters)

    def install(self, model: Any, ops: Any) -> None:
        """Add score/bounds/selection copies inside the existing one-request eight-row graph."""
        import torch

        super().install(model, ops)

        def decorate(native: Any) -> Any:
            @functools.wraps(native)
            def topk(*args: Any, **kwargs: Any) -> Any:
                if self.active_layer != 3:
                    return native(*args, **kwargs)
                if len(args) != 8 or kwargs:
                    raise ValueError("Pinned native decode top-k signature changed")
                prefix = "layer3.native_decode_topk"
                count = self.counts.get(prefix, 0)
                self.counts[prefix] = count + 1
                name = prefix + f"#{count}"
                params = {"next_n": args[1], "rows": args[4], "stride0": args[5],
                          "stride1": args[6], "k": args[7]}
                self.record(name + "/input", "operand", {"scores": args[0], "seq_lens": args[2]}, params)
                result = native(*args, **kwargs)
                self.record(name + "/output", "result", args[3], params)
                return result
            return topk
        self.replace(torch.ops._C, "top_k_per_row_decode", decorate)


def arm_capture(worker: Any) -> None:
    """Arm bounded graph copies before native target graph capture; replace no arithmetic."""
    from vllm import _custom_ops as ops
    from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager

    if worker.model_runner.cudagraph_manager is not None:
        raise ValueError("Decode observation must be armed before graph capture")
    patch_manager(ModelCudaGraphManager, ops, LongDecodeTrace)


def install(worker: Any) -> dict[str, Any]:
    """Join first-decode graph-written operands to the existing native sample/proposal frame."""
    report = install_context(worker)
    runner = worker.model_runner
    capture = runner.sample.__self__
    manager = runner.cudagraph_manager
    trace = manager._mxwave_component_trace
    if trace.methods or not any(e["name"] == "layer3.native_decode_topk#0/output" for e in trace.events):
        raise ValueError("Missing native decode selection or unrestored capture wrappers")
    plan = json.loads(Path("/trial/plan.json").read_bytes())
    original_replay, original_sample = manager.run_fullgraph, runner.sample
    last_descriptor: list[Any] = [None]

    @functools.wraps(original_replay)
    def replay(descriptor: Any) -> Any:
        result = original_replay(descriptor)
        last_descriptor[0] = descriptor
        return result

    @functools.wraps(original_sample)
    def sample(hidden: Any, batch: Any, grammar: Any) -> Any:
        result = original_sample(hidden, batch, grammar)
        frame = capture.pending
        if frame is None or frame["metadata"]["index"] != 1:
            return result
        positions = frame["tensors"]["target_positions"].reshape(-1).tolist()
        if (positions != list(range(plan["decode_start_position"], plan["decode_start_position"] + 8))
                or batch.num_tokens_after_padding != 8 or not selected_graph(last_descriptor[0])
                or last_descriptor[0] != manager._mxwave_component_descriptor):
            raise ValueError("First decode differs from the independently reviewed input geometry")
        meta = trace.metadata()
        meta["graph_descriptor"] = graph_metadata(last_descriptor[0])
        for index, event in enumerate(trace.events):
            for leaf, tensor in enumerate(event["tensors"]):
                key = f"decode_component_{index:04d}_{leaf:02d}"
                frame["tensors"][key] = cpu_copy(tensor["value"])
                meta["events"][index]["tensors"][leaf]["payload_key"] = key
        frame["metadata"]["decode_components"] = meta
        return result

    manager.run_fullgraph, runner.sample = replay, sample
    report["decode_components"] = {"rows": 8, "layers": [0, 1, 2, 3, 4],
                                    "first_position": plan["decode_start_position"],
                                    "maximum_capture_bytes": trace.max_bytes,
                                    "graph_copies_present_in_all_controls": True,
                                    "arithmetic_replaced": False, "timing_claim": False}
    return report
