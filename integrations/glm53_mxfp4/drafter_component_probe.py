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


def selected_graph(descriptor: Any) -> bool:
    """Select one request with eight tokens, not every graph with eight total tokens."""
    return (
        str(descriptor.cg_mode).split(".")[-1] == "FULL"
        and descriptor.num_tokens == 8 and descriptor.num_reqs == 1
        and descriptor.uniform_token_count == 8 and descriptor.num_active_loras == 0
    )


def graph_metadata(descriptor: Any) -> dict[str, Any]:
    """Retain the complete dispatch identity in plain metadata."""
    return {key: str(value) if key == "cg_mode" else value
            for key, value in vars(descriptor).items()}


class ComponentTrace(BoundaryCapture):
    """Record early-layer activations, routing and GEMMs under native graph replay."""

    def __init__(self, capturing: Any = None) -> None:
        import torch

        self.selected_forward = False
        if capturing is None:
            capturing = lambda: self.selected_forward and torch.cuda.is_current_stream_capturing()
        super().__init__(capturing, rows=8, layers=(0, 1, 2, 3, 4), max_bytes=64 * 1024**2)

    def forward_factory(self, original: Any) -> Any:
        """Gate copies using the descriptor supplied to the native forward factory."""
        @functools.wraps(original)
        def build(*args: Any, **kwargs: Any) -> Any:
            factory = original(*args, **kwargs)

            def create(descriptor: Any, warmup: bool) -> Any:
                forward = factory(descriptor, warmup=warmup)

                @functools.wraps(forward)
                def run(*args: Any, **kwargs: Any) -> Any:
                    previous = self.selected_forward
                    self.selected_forward = selected_graph(descriptor) and not warmup
                    try:
                        return forward(*args, **kwargs)
                    finally:
                        self.selected_forward = previous
                return run
            return create
        return build

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


def patch_manager(
    manager_type: Any, ops: Any, trace_type: type[ComponentTrace] | None = None,
) -> None:
    """Insert graph copies once, restoring all Python methods even if capture fails."""
    original = manager_type.capture

    @functools.wraps(original)
    def capture(manager: Any, model: Any, *args: Any, **kwargs: Any) -> Any:
        trace = (trace_type or ComponentTrace)()
        try:
            trace.replace(manager, "_build_forward_fn_factory", trace.forward_factory)
            trace.install(model, ops)
            result = original(manager, model, *args, **kwargs)
        finally:
            trace.restore()
        descriptors = [d for d in manager.graphs if selected_graph(d)]
        if len(descriptors) != 1 or not trace.events or len(trace.locks) != 4:
            raise ValueError({
                "error": "Require one one-request/eight-token graph and four expert GEMMs",
                "descriptors": [graph_metadata(d) for d in manager.graphs],
                "events": len(trace.events), "gemms": len(trace.locks),
                "layer_calls": trace.counts,
            })
        # Each selected layer is entered once in this one graph.
        layer_calls = [v for k, v in trace.counts.items()
                       if k.rsplit(".", 1)[-1] in {"0", "1", "2", "3", "4"}]
        if layer_calls != [1] * 5:
            raise ValueError(f"Selected layers were not captured exactly once: {layer_calls}")
        manager._mxwave_component_trace = trace
        manager._mxwave_component_descriptor = descriptors[0]
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
        self.original_target_replay = runner.cudagraph_manager.run_fullgraph
        self.last_target_descriptor: Any = None
        assert not self.component_trace.methods

    def attach(self) -> None:
        """Observe actual target dispatch without changing its descriptor or output."""
        super().attach()
        self.runner.cudagraph_manager.run_fullgraph = self.target_replay

    def target_replay(self, descriptor: Any) -> Any:
        """Retain the descriptor of the actual native target replay."""
        result = self.original_target_replay(descriptor)
        self.last_target_descriptor = descriptor
        return result

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
        assert selected_graph(self.last_target_descriptor)
        assert self.last_target_descriptor == (
            self.runner.cudagraph_manager._mxwave_component_descriptor)
        assert positions.tolist() == list(range(int(positions[0]), int(positions[0]) + 8))
        meta = self.component_trace.metadata()
        meta["graph_descriptor"] = graph_metadata(self.last_target_descriptor)
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


def validate_graph_candidates() -> dict[str, Any]:
    """Exercise the installed CPU candidate builder with this frozen dynamic-draft policy."""
    from types import SimpleNamespace

    from vllm.config import CUDAGraphMode
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager

    manager = object.__new__(CudaGraphManager)
    manager.compilation_config = SimpleNamespace(
        cudagraph_capture_sizes=[1, 2, 4, 8, 16], max_cudagraph_capture_size=16)
    manager.vllm_config = SimpleNamespace(
        num_speculative_tokens=7,
        speculative_config=SimpleNamespace(
            uses_dynamic_speculative_decoding=lambda: True,
            num_speculative_tokens_per_batch_size=[[1, 1, 7], [2, 2, 5], [3, 16, 3]],
        ),
    )
    manager.max_num_reqs = 16
    manager.decode_query_len = 8
    manager.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
    manager.varlen_decode = False
    manager.lora_capture_cases = [0]
    manager._candidates, manager._capture_descs = {}, {}
    manager._init_candidates()
    descriptors = [d for group in manager._capture_descs.values() for d in group]
    assert len(descriptors) == 7
    assert len([d for d in descriptors if d.num_tokens == 8]) == 2
    assert len([d for d in descriptors if selected_graph(d)]) == 1
    return {"status": "passed", "descriptors": [graph_metadata(d) for d in descriptors]}
