"""Observe target features, verification and native draft graphs without replacing outputs."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any


def cpu_copy(tensor: Any) -> Any:
    """Make an independent CPU copy, even when the source already resides on CPU."""
    return tensor.detach().to(device="cpu", copy=True).contiguous()


def retain_frame(path: Path, tensors: dict[str, Any], metadata: dict[str, Any]) -> None:
    """Retain exact tensor bits and a digest, with readable nonexecutable permissions."""
    import torch
    from drafter_head_probe import write
    from safetensors.torch import save_file

    assert not path.exists()
    stored = {}
    metadata["tensor_dtypes"] = {}
    for name, tensor in tensors.items():
        value = cpu_copy(tensor)
        metadata["tensor_dtypes"][name] = str(value.dtype)
        stored[name] = value.view(torch.int16) if value.dtype == torch.bfloat16 else value
    save_file(stored, str(path))
    path.chmod(0o644)
    metadata.update(
        payload_bytes=path.stat().st_size,
        payload_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    write(path.with_suffix(".json"), metadata)


def draft_weight_inventory(draft: Any) -> dict[str, Any]:
    """Hash effective learned draft parameters and its derived context projection."""
    from drafter_head_probe import tensor_bytes

    tensors = {
        name: value for name, value in draft.named_parameters()
        if name not in ("lm_head.weight", "model.embed_tokens.weight")
    }
    for name in ("_fused_kv_weight", "_fused_kv_bias"):
        value = getattr(draft.model, name, None)
        if value is not None:
            tensors["model." + name] = value
    result = {}
    for name, value in sorted(tensors.items()):
        digest = hashlib.sha256()
        flat = value.detach().reshape(-1)
        for chunk in flat.split(max(1, 16 * 1024**2 // value.element_size())):
            digest.update(tensor_bytes(chunk))
        result[name] = {
            "shape": list(value.shape), "dtype": str(value.dtype),
            "bytes": value.numel() * value.element_size(), "sha256": digest.hexdigest(),
        }
    assert result and "model._fused_kv_weight" in result
    return result


class ContextCapture:
    """Join sampling and proposal observations for one serial request at a time."""

    def __init__(self, runner: Any, root: Path, plan: dict[str, Any], rank: int) -> None:
        self.runner = runner
        self.spec = runner.speculator
        self.root, self.plan, self.rank = root, plan, rank
        self.pending: dict[str, Any] | None = None
        self.counts: dict[str, int] = {}
        self.original_sample = runner.sample
        self.original_logits = runner.model.compute_logits
        self.original_propose = self.spec.propose
        self.propose_signature = inspect.signature(self.original_propose)
        self.original_replay = self.spec.query_cudagraph_manager.run_fullgraph

    def attach(self) -> None:
        """Wrap eager boundaries while leaving captured GPU operations intact."""
        self.runner.sample = self.sample
        self.runner.model.compute_logits = self.logits
        self.spec.propose = self.propose
        self.spec.query_cudagraph_manager.run_fullgraph = self.replay

    def sample(self, hidden: Any, batch: Any, grammar: Any) -> Any:
        """Capture target inputs and the actual sampler's decision."""
        marker = json.loads((self.root / "probe-marker.json").read_bytes())
        assert self.pending is None
        if not marker["active"]:
            return self.original_sample(hidden, batch, grammar)
        assert batch.num_reqs == 1 and grammar is None
        case = marker["case"]
        index = self.counts.get(case, 0)
        assert index < self.plan["max_frames_per_case"]
        n = batch.num_tokens
        tensors = {
            "target_input_ids": cpu_copy(batch.input_ids[:n]),
            "target_positions": cpu_copy(batch.positions[:n]),
            "target_idx_mapping": cpu_copy(batch.idx_mapping[:1]),
            "target_query_start_loc": cpu_copy(batch.query_start_loc[:2]),
            "target_logits_indices": cpu_copy(batch.logits_indices),
            "target_final_hidden": cpu_copy(hidden[:n]),
        }
        self.pending = {
            "tensors": tensors,
            "metadata": {
                "schema": "mxwave-draft-context-frame-v1",
                "case": case, "index": index, "rank": self.rank,
                "target_tokens": n, "target_padded_tokens": batch.num_tokens_after_padding,
                "target_num_draft_tokens": batch.num_draft_tokens,
                "full_target_logits_retained": index < self.plan["full_logits_frames"],
            },
        }
        try:
            result = self.original_sample(hidden, batch, grammar)
            output, sampled, rejected = result
            tensors.update(
                target_sampled_tokens=cpu_copy(output.sampled_token_ids),
                target_num_sampled=cpu_copy(sampled),
                target_num_rejected=cpu_copy(rejected),
            )
            return result
        except BaseException:
            self.pending = None
            raise

    def logits(self, hidden: Any, *args: Any, **kwargs: Any) -> Any:
        """Retain target logits before sampling, returning the original tensor."""
        result = self.original_logits(hidden, *args, **kwargs)
        if self.pending is not None:
            tensors = self.pending["tensors"]
            assert "target_top_ids" not in tensors
            values, ids = result.topk(32, dim=-1)
            tensors.update(target_top_ids=cpu_copy(ids), target_top_values=cpu_copy(values))
            if self.pending["metadata"]["full_target_logits_retained"]:
                tensors["target_logits"] = cpu_copy(result)
        return result

    def replay(self, descriptor: Any) -> Any:
        """Copy actual outputs after native replay, without shadow computation."""
        result = self.original_replay(descriptor)
        if self.pending is not None:
            assert (descriptor.num_reqs, descriptor.num_tokens) == (1, 8)
            graph = self.spec._mxwave_head_taps["graphs"][(1, 8)]
            tensors = self.pending["tensors"]
            assert "draft_hidden" not in tensors
            for name in ("hidden", "ids", "values", "scores"):
                tensors["draft_" + name] = cpu_copy(graph[name])
            tensors.update(
                draft_proposal=cpu_copy(self.spec.draft_tokens[:1]),
                draft_realized_scores=cpu_copy(self.spec._selector_scores[:1]),
                sample_positions=cpu_copy(self.spec.sample_pos[:7]),
                draft_query_ids=cpu_copy(self.spec.input_buffers.input_ids[:8]),
                draft_query_positions=cpu_copy(self.spec.input_buffers.positions[:8]),
            )
            self.pending["metadata"]["draft_graph"] = [1, 8]
        return result

    def propose(self, *args: Any, **kwargs: Any) -> Any:
        """Capture the five target boundaries and join them to the next draft graph."""
        frame = self.pending
        if frame is None:
            return self.original_propose(*args, **kwargs)
        call = self.propose_signature.bind(*args, **kwargs).arguments
        batch = call["input_batch"]
        aux = call["aux_hidden_states"]
        assert aux is not None and len(aux) == 5 and batch.num_reqs == 1
        n = batch.num_tokens
        tensors = frame["tensors"]
        for layer, value in zip(self.plan["target_feature_layers"], aux, strict=True):
            tensors[f"aux_layer_{layer}"] = cpu_copy(value[:n])
        slot = int(batch.idx_mapping[0].item())
        tensors.update(
            anchor_before_propose=cpu_copy(call["last_sampled"][slot:slot + 1]),
            request_temperature=cpu_copy(call["temperature"][slot:slot + 1]),
            request_seed=cpu_copy(call["seeds"][slot:slot + 1]),
        )
        try:
            result = self.original_propose(*args, **kwargs)
            assert "draft_hidden" in tensors, "Expected a native FULL draft replay"
            tensors["combined_target_hidden"] = cpu_copy(self.spec.hidden_states[:n])
            tensors["context_positions"] = cpu_copy(self.spec.context_positions[:n])
            for index, slots in enumerate(self.spec._context_slot_mappings):
                tensors[f"context_slot_mapping_{index}"] = cpu_copy(slots[:n])
            tensors["returned_proposal"] = cpu_copy(result)
            meta = frame["metadata"]
            stem = f"context-{meta['case']}-{meta['index']:04d}-r{self.rank}"
            retain_frame(self.root / (stem + ".safetensors"), tensors, meta)
            self.counts[meta["case"]] = meta["index"] + 1
            from drafter_head_probe import write

            write(self.root / f"probe-progress-r{self.rank}.json", {"counts": self.counts})
            return result
        finally:
            self.pending = None


def install(worker: Any) -> dict[str, Any]:
    """Install the passive probe after capture and record effective drafter weights."""
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
    root = Path("/traces")
    write(root / f"draft-runtime-weights-r{rank}.json", inventory)
    capture = ContextCapture(runner, root, plan, rank)
    capture.attach()
    report = {
        "status": "installed", "rank": rank, "sources": sources,
        "captured_graphs": [list(key) for key in sorted(spec._mxwave_head_taps["graphs"])],
        "target_feature_layers": plan["target_feature_layers"],
        "runtime_weight_tensors": len(inventory),
        "runtime_weight_bytes": sum(row["bytes"] for row in inventory.values()),
        "observation": "Passive copies; original target, sampling and draft outputs returned",
        "timing_claim": False,
    }
    write(root / f"probe-installed-r{rank}.json", report)
    return report
