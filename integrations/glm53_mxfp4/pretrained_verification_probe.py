"""Observe native decisions for different pretrained drafter architectures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .drafter_verification_probe import VerificationCapture


class PretrainedVerificationCapture(VerificationCapture):
    """Keep actual graph identities and enforce a per-rank trace byte bound."""

    def __init__(self, runner: Any, root: Path, plan: dict[str, Any], rank: int) -> None:
        super().__init__(runner, root, plan, rank)
        self.original_target_replay = runner.cudagraph_manager.run_fullgraph
        self.descriptor: dict[str, Any] | None = None

    def attach(self) -> None:
        """Install eager observers after graph capture without replacing graph math."""
        super().attach()
        self.runner.cudagraph_manager.run_fullgraph = self.target_replay

    def target_replay(self, descriptor: Any) -> Any:
        """Retain the actual dispatch descriptor and return the original result."""
        self.descriptor = {
            key: str(value) if key == "cg_mode" else value
            for key, value in vars(descriptor).items()
        }
        return self.original_target_replay(descriptor)

    def prepare_attn(self, *args: Any, **kwargs: Any) -> Any:
        """Clear the prior descriptor before observing the next native forward."""
        self.descriptor = None
        return super().prepare_attn(*args, **kwargs)

    def sample(self, hidden: Any, batch: Any, grammar: Any) -> Any:
        """Join graph identity to the original decision, including eager prefill."""
        result = super().sample(hidden, batch, grammar)
        if self.pending is not None:
            self.pending["metadata"]["target_graph_descriptor"] = self.descriptor
        return result

    def propose(self, *args: Any, **kwargs: Any) -> Any:
        """Stop observation before a request could exceed its trace disk budget."""
        if self.pending is not None:
            used = sum(p.stat().st_size for p in self.root.glob("context-*.safetensors"))
            reserve = self.plan["max_state_bytes_per_frame"] + 64 * 1024**2
            if used + reserve > self.plan["max_capture_bytes_per_rank"]:
                raise ValueError("Passive trace exceeded the per-rank byte bound")
        return super().propose(*args, **kwargs)


def install(worker: Any) -> dict[str, Any]:
    """Validate each native drafter and observe the same target with its actual taps."""
    import os

    import pretrained_startup
    from drafter_head_probe import validate_sources, write
    from vllm.distributed import get_tensor_model_parallel_rank
    from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
        get_eagle3_aux_layers_from_config,
    )

    report = pretrained_startup.install(worker)
    plan = json.loads(Path("/trial/plan.json").read_bytes())
    sources = validate_sources(plan)
    variant = os.environ["MXWAVE_DRAFTER_VARIANT"]
    layers = plan["variant_feature_layers"][variant]
    actual = get_eagle3_aux_layers_from_config(worker.vllm_config.speculative_config)
    if list(actual) != [i + 1 for i in layers]:
        raise ValueError("Native feature taps differ from the pinned drafter configuration")
    plan["target_feature_layers"] = layers
    runner = worker.model_runner
    spec = runner.speculator
    assert (1, 8) in spec._mxwave_head_taps["graphs"]
    assert spec.num_speculative_steps == 7 and spec.selector_top_k == 16
    rank = get_tensor_model_parallel_rank()
    capture = PretrainedVerificationCapture(runner, Path("/traces"), plan, rank)
    capture.attach()
    report.update(
        sources=sources, target_feature_layers=layers, timing_claim=False,
        selected_cache_layers=list(capture.layers),
        observation="Passive pre/post-sampler logits, native proposals and pre-forward state",
    )
    write(Path("/traces") / f"probe-installed-r{rank}.json", report)
    return report
