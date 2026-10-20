"""Retain native target verification and selected pre-forward recurrent state."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

from .drafter_context_probe import ContextCapture


def state_slots(indices: Any) -> list[int]:
    """Preserve first occurrence order and omit invalid physical cache slots."""
    return list(dict.fromkeys(int(x) for x in indices.reshape(-1).tolist() if int(x) >= 0))


class VerificationCapture(ContextCapture):
    """Observe the original forward and sampler without substituting either output."""

    def __init__(self, runner: Any, root: Path, plan: dict[str, Any], rank: int) -> None:
        super().__init__(runner, root, plan, rank)
        self.original_attn = runner.model_state.prepare_attn
        self.attn_signature = inspect.signature(self.original_attn)
        self.before: dict[str, Any] | None = None
        self.live_logits: Any = None
        self.layers = {
            module.prefix: module for name, module in runner.model.named_modules()
            if type(module).__name__ == "Glm5NextLinearAttention"
            and any(f"layers.{i}.self_attn" in name for i in plan["state_capture_layers"])
        }
        assert len(self.layers) == len(plan["state_capture_layers"]), list(self.layers)

    def attach(self) -> None:
        """Install only eager boundary wrappers after native graph capture."""
        super().attach()
        self.runner.model_state.prepare_attn = self.prepare_attn

    def prepare_attn(self, *args: Any, **kwargs: Any) -> Any:
        """Copy metadata and selected cache slots after preparation, before execution."""
        import torch
        from drafter_context_probe import cpu_copy

        result = self.original_attn(*args, **kwargs)
        marker = json.loads((self.root / "probe-marker.json").read_bytes())
        if not marker["active"]:
            self.before = None
            return result
        call = self.attn_signature.bind(*args, **kwargs).arguments
        assert not call.get("for_capture", False)
        batch = call["input_batch"]
        assert batch.num_reqs == 1 and self.before is None
        slot = int(batch.idx_mapping[0].item())
        tensors = {
            "before_num_computed": cpu_copy(
                self.runner.req_states.num_computed_tokens.gpu[slot:slot + 1]),
            "before_last_sampled": cpu_copy(
                self.runner.req_states.last_sampled_tokens[slot:slot + 1]),
            "before_num_accepted": cpu_copy(
                self.runner.model_state.num_accepted_tokens_gpu[slot:slot + 1]),
        }
        metadata: dict[str, Any] = {
            "target_graph_mode": str(call["cudagraph_mode"]), "state_layers": {},
        }
        cache_bytes = 0
        for prefix, module in self.layers.items():
            attn = result[prefix]
            layer = prefix.split(".layers.")[1].split(".")[0]
            record: dict[str, Any] = {
                "prefix": prefix, "num_spec_decodes": int(attn.num_spec_decodes),
                "cache_shapes": [list(t.shape) for t in module.kv_cache],
            }
            for field in (
                "num_accepted_tokens", "spec_state_indices_tensor", "spec_query_start_loc",
                "non_spec_state_indices_tensor", "non_spec_query_start_loc",
                "spec_token_indx", "non_spec_token_indx", "has_initial_state",
            ):
                value = getattr(attn, field)
                if value is not None:
                    assert isinstance(value, torch.Tensor) and value.numel() <= 16384
                    tensors[f"before_layer_{layer}_meta_{field}"] = cpu_copy(value)
            if attn.num_spec_decodes:
                indices = cpu_copy(attn.spec_state_indices_tensor[:1])
                slots = state_slots(indices)
                assert len(slots) <= 8 and slots
                record["physical_state_slots"] = slots
                for index, cache in enumerate(module.kv_cache):
                    # Keep all speculative columns; analysis must select the state
                    # actually read, rather than treating unused slots as meaningful.
                    selected = torch.tensor(slots, dtype=torch.long, device=cache.device)
                    value = cpu_copy(cache.index_select(0, selected))
                    cache_bytes += value.numel() * value.element_size()
                    assert cache_bytes <= self.plan["max_state_bytes_per_frame"]
                    tensors[f"before_layer_{layer}_cache_{index}"] = value
            metadata["state_layers"][layer] = record
        self.before = {"tensors": tensors, "metadata": metadata}
        return result

    def logits(self, hidden: Any, *args: Any, **kwargs: Any) -> Any:
        """Keep the original pre-sampler logits and their actual mutable output."""
        result = super().logits(hidden, *args, **kwargs)
        if self.pending is not None:
            self.live_logits = result
        return result

    def sample(self, hidden: Any, batch: Any, grammar: Any) -> Any:
        """Join pre-forward state and post-sampler logits to the unmodified decision."""
        from drafter_context_probe import cpu_copy

        result = super().sample(hidden, batch, grammar)
        if self.pending is not None:
            assert self.before is not None and self.live_logits is not None
            self.pending["tensors"].update(self.before["tensors"])
            self.pending["metadata"].update(self.before["metadata"])
            self.pending["tensors"]["target_logits_after_sampler"] = cpu_copy(self.live_logits)
        self.before = None
        self.live_logits = None
        return result


def install(worker: Any) -> dict[str, Any]:
    """Verify all effective drafter tensors and install a bounded passive trace."""
    import os

    from drafter_context_probe import draft_weight_inventory
    from drafter_head_probe import validate_sources, write
    from vllm.distributed import get_tensor_model_parallel_rank

    plan = json.loads(Path("/trial/plan.json").read_bytes())
    sources = validate_sources(plan)
    runner = worker.model_runner
    spec = runner.speculator
    assert (1, 8) in spec._mxwave_head_taps["graphs"]
    assert spec.num_speculative_steps == 7 and spec.selector_top_k == 16
    rank = get_tensor_model_parallel_rank()
    inventory = draft_weight_inventory(spec.model)
    expected = json.loads((Path("/trial") / f"expected-runtime-r{rank}.json").read_bytes())
    variant = os.environ["MXWAVE_DRAFTER_VARIANT"]
    expected["model.fc.weight"]["sha256"] = plan["context_payloads"][variant]
    assert inventory == expected
    root = Path("/traces")
    write(root / f"draft-runtime-weights-r{rank}.json", inventory)
    probe = VerificationCapture(runner, root, plan, rank)
    probe.attach()
    report = {
        "status": "installed", "rank": rank, "sources": sources,
        "runtime_weights_verified": True, "selected_cache_layers": list(probe.layers),
        "timing_claim": False, "observation": "Copies only; native forward and sampler retained",
    }
    write(root / f"probe-installed-r{rank}.json", report)
    return report
