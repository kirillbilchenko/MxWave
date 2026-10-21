"""Bounded observations of native DFlash proposals and target verification."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

from profile_worker import ProfileWorker


def tensor_report(value: Any, limit: int = 128) -> dict[str, Any]:
    """Copy a bounded preview and whole-tensor numerical health to CPU."""
    import torch

    value = value.detach()
    flat = value.reshape(-1)
    result = {"shape": list(value.shape), "dtype": str(value.dtype),
              "preview": flat[:limit].float().cpu().tolist()}
    if flat.numel():
        floats = flat.float()
        finite = torch.isfinite(floats)
        result.update(finite=int(finite.sum().item()), count=flat.numel())
        if bool(finite.all().item()):
            result.update(min=float(floats.min().item()), max=float(floats.max().item()),
                          rms=float(floats.square().mean().sqrt().item()))
    return result


class DraftTrace:
    """Observe eager boundaries, with a separately labelled draft-only eager control."""

    def __init__(self, runner: Any, path: Path, mode: str, frames: int) -> None:
        self.runner, self.spec, self.path = runner, runner.speculator, path
        self.mode, self.frames, self.index = mode, frames, 0
        self.original_sample = runner.sample
        self.original_propose = self.spec.propose
        self.original_candidates = self.spec.model.compute_candidates
        self.signature = inspect.signature(self.original_propose)
        self.candidates: dict[str, Any] | None = None

    def write(self, event: dict[str, Any]) -> None:
        """Append durable observations without replacing any model output."""
        with self.path.open("a") as output:
            output.write(json.dumps({"mode": self.mode, "index": self.index, **event}) + "\n")

    def attach(self) -> None:
        """Attach only after startup and CUDA graph capture have completed."""
        self.runner.sample = self.sample
        self.spec.propose = self.propose
        self.spec.model.compute_candidates = self.compute_candidates

    def detach(self) -> None:
        """Restore the exact bound methods installed before the observer."""
        self.runner.sample = self.original_sample
        self.spec.propose = self.original_propose
        self.spec.model.compute_candidates = self.original_candidates

    def sample(self, hidden: Any, batch: Any, grammar: Any) -> Any:
        """Retain target input/proposal alignment and actual rejection results."""
        if self.index >= self.frames:
            return self.original_sample(hidden, batch, grammar)
        before = {
            "event": "sample", "requests": batch.num_reqs,
            "num_draft_tokens": batch.num_draft_tokens,
            "input_ids": tensor_report(batch.input_ids[:batch.num_tokens]),
            "positions": tensor_report(batch.positions[:batch.num_tokens]),
            "idx_mapping": tensor_report(batch.idx_mapping[:batch.num_reqs]),
            "logits_indices": tensor_report(batch.logits_indices),
            "hidden": tensor_report(hidden[:batch.num_tokens], 8),
        }
        result = self.original_sample(hidden, batch, grammar)
        output, sampled, rejected = result
        before.update(sampled=tensor_report(output.sampled_token_ids),
                      num_sampled=tensor_report(sampled), num_rejected=tensor_report(rejected))
        self.write(before)
        return result

    def compute_candidates(self, hidden: Any) -> Any:
        """Observe actual eager candidates; captured graphs retain native buffers instead."""
        result = self.original_candidates(hidden)
        if self.index < self.frames:
            ids, logits = result
            self.candidates = {"hidden": tensor_report(hidden, 8),
                               "ids": tensor_report(ids), "logits": tensor_report(logits)}
        return result

    def propose(self, *args: Any, **kwargs: Any) -> Any:
        """Capture actual context and proposals, optionally forcing only draft eager execution."""
        call = self.signature.bind(*args, **kwargs)
        if call.arguments.get("dummy_run"):
            return self.original_propose(*args, **kwargs)
        if self.mode == "draft_eager":
            call.arguments["is_profile"] = True
        if self.index >= self.frames:
            return self.original_propose(*call.args, **call.kwargs)
        batch = call.arguments["input_batch"]
        aux = call.arguments["aux_hidden_states"]
        event: dict[str, Any] = {
            "event": "propose", "requests": batch.num_reqs, "tokens": batch.num_tokens,
            "aux": [tensor_report(x[:batch.num_tokens], 8) for x in (aux or [])],
        }
        self.candidates = None
        result = self.original_propose(*call.args, **call.kwargs)
        n = batch.num_reqs
        q = n * self.spec.num_query_per_req
        steps = n * self.spec.num_speculative_steps
        for name, value in {
            "proposal": result,
            "query_ids": self.spec.input_buffers.input_ids[:q],
            "query_positions": self.spec.input_buffers.positions[:q],
            "sample_indices": self.spec.sample_indices[:steps],
            "sample_positions": self.spec.sample_pos[:steps],
            "sample_mapping": self.spec.sample_idx_mapping[:steps],
            "context_positions": self.spec.context_positions[:batch.num_tokens],
            "context_slots": self.spec._context_slot_mappings[:, :batch.num_tokens],
            "combined_hidden": self.spec.hidden_states[:batch.num_tokens],
            "candidate_ids": self.spec.candidate_sampler.cached_candidate_ids[:n],
            "candidate_scores": self.spec.candidate_sampler.scores[:n],
        }.items():
            event[name] = tensor_report(value, 8 if name == "combined_hidden" else 128)
        event["eager_candidates"] = self.candidates
        self.write(event)
        self.index += 1
        return result


class DraftTraceWorker(ProfileWorker):
    """Expose private diagnostic RPCs without modifying installed vLLM source."""

    def mxwave_draft_trace_start(self, label: str, mode: str, frames: int = 16) -> dict[str, Any]:
        """Start one serial diagnostic; all captured timings are unqualified."""
        from vllm.distributed import get_tensor_model_parallel_rank

        if mode not in ("native", "draft_eager") or not 1 <= frames <= 64:
            raise ValueError("Invalid bounded trace mode")
        if not label.replace("-", "").isalnum() or hasattr(self, "_mxwave_draft_trace"):
            raise ValueError("Invalid trace label or observer already attached")
        runner = self.model_runner
        spec = runner.speculator
        if type(spec).__name__ != "DFlash2Speculator":
            raise RuntimeError("Expected native DFlash2")
        rank = get_tensor_model_parallel_rank()
        path = Path("/reports") / f"draft-trace-{label}-r{rank}.jsonl"
        path.touch(exist_ok=False)
        tracer = DraftTrace(runner, path, mode, frames)
        tracer.write({"event": "start", "rank": rank,
                      "target_layers": list(runner.model.language_model.model.aux_hidden_state_layers),
                      "fc_weight": tensor_report(spec.model.model.fc.weight, 8),
                      "draft_logits_present": spec.draft_logits is not None,
                      "batch_sharder": type(runner.batch_sharder).__name__,
                      "draft_kv_dtypes": [str(x.self_attn.attn.kv_cache.dtype)
                                          for x in spec.model.model.layers]})
        tracer.attach()
        self._mxwave_draft_trace = tracer
        return {"path": str(path), "rank": rank, "mode": mode, "frames": frames}

    def mxwave_draft_trace_stop(self) -> dict[str, Any]:
        """Detach the observer before the next request or restoration."""
        tracer = self._mxwave_draft_trace
        tracer.detach()
        del self._mxwave_draft_trace
        return {"frames": tracer.index, "mode": tracer.mode}
