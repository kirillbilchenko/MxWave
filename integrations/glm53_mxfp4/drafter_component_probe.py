"""Locate the first changed target operation in a reproduced eight-row verification."""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

from .decode_boundary_capture import BoundaryCapture
from .drafter_context_probe import cpu_copy
from .drafter_verification_probe import VerificationCapture
from .drafter_verification_probe import install as install_verification

_INSTALLED = False


class ComponentTrace(BoundaryCapture):
    """Record early-layer activations, routing and GEMMs under native graph replay."""

    def __init__(self, capturing: Any = None) -> None:
        super().__init__(capturing, rows=8, layers=(0, 1, 2, 3, 4), max_bytes=64 * 1024**2)

    def install(self, model: Any, ops: Any) -> None:
        """Extend existing attention/mHC/expert boundaries to dense MLP children."""
        super().install(model, ops)
        for name, module in model.named_modules():
            if (getattr(module, "layer_idx", None) not in self.layers
                    or not hasattr(module, "hc_fused_post_pre")):
                continue
            if module.layer_idx < 3:
                for path, child in module.mlp.named_modules():
                    if path:
                        self.replace(child, "forward", self.wrapper(name + ".mlp." + path))
            else:
                experts = module.mlp.experts
                self.replace(experts.routed_experts, "forward_modular",
                             self.wrapper(name + ".mlp.routed_experts"))


def patch_manager(manager_type: Any, ops: Any) -> None:
    """Insert graph copies once, restoring all Python methods even if capture fails."""
    original = manager_type.capture

    @functools.wraps(original)
    def capture(manager: Any, model: Any, *args: Any, **kwargs: Any) -> Any:
        trace = ComponentTrace()
        try:
            trace.install(model, ops)
            result = original(manager, model, *args, **kwargs)
        finally:
            trace.restore()
        descriptors = [d for d in manager.graphs if d.num_tokens == 8]
        if len(descriptors) != 1 or not trace.events or len(trace.locks) != 4:
            raise ValueError("Require one eight-row graph and four captured expert GEMMs")
        # Each selected layer is entered once in this one graph.
        layer_calls = [v for k, v in trace.counts.items()
                       if k.rsplit(".", 1)[-1] in {"0", "1", "2", "3", "4"}]
        if layer_calls != [1] * 5:
            raise ValueError(f"Selected layers were not captured exactly once: {layer_calls}")
        manager._mxwave_component_trace = trace
        return result

    manager_type.capture = capture


def arm_capture(worker: Any) -> None:
    """Validate the installed source and arm only this disposable worker's capture."""
    global _INSTALLED
    if _INSTALLED:
        return
    from drafter_head_probe import validate_sources
    from vllm import _custom_ops as ops
    from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager

    plan = json.loads(Path("/trial/plan.json").read_bytes())
    validate_sources(plan)
    assert worker.model_runner.cudagraph_manager is None
    patch_manager(ModelCudaGraphManager, ops)
    _INSTALLED = True


class ComponentCapture(VerificationCapture):
    """Join graph-written component copies to the actual eager sampler boundary."""

    def __init__(self, runner: Any, root: Path, plan: dict[str, Any], rank: int) -> None:
        super().__init__(runner, root, plan, rank)
        self.component_trace = runner.cudagraph_manager._mxwave_component_trace
        assert not self.component_trace.methods

    def sample(self, hidden: Any, batch: Any, grammar: Any) -> Any:
        """Retain component payloads only at the prior and first divergent forwards."""
        result = super().sample(hidden, batch, grammar)
        frame = self.pending
        if frame is None:
            return result
        positions = frame["tensors"]["target_positions"]
        selected = int(positions[0]) in self.plan["component_start_positions"]
        frame["metadata"]["components_retained"] = selected
        if not selected:
            return result
        assert len(positions) == 8 and batch.num_tokens_after_padding == 8
        assert frame["metadata"]["target_graph_mode"].endswith("FULL")
        assert positions.tolist() == list(range(int(positions[0]), int(positions[0]) + 8))
        meta = self.component_trace.metadata()
        for index, event in enumerate(self.component_trace.events):
            for leaf, tensor in enumerate(event["tensors"]):
                key = f"component_{index:04d}_{leaf:02d}"
                frame["tensors"][key] = cpu_copy(tensor["value"])
                meta["events"][index]["tensors"][leaf]["payload_key"] = key
        frame["metadata"]["components"] = meta
        return result


def install(worker: Any) -> dict[str, Any]:
    """Install the joined probe after startup, preserving native sampling and proposals."""
    return install_verification(worker, ComponentCapture)
