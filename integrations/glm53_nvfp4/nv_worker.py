"""Read-only qualification RPCs for the disposable original-checkpoint test."""
from __future__ import annotations

import os
from typing import Any

import nv_adapter


class NvWorker:
    """Expose allocation, model ownership, graph inventory and optional host traces."""

    def nv_snapshot(self, sources: bool = False) -> dict[str, Any]:
        """Record the actual loaded stack and ensure all target scale families are covered."""
        from vllm.distributed import get_tensor_model_parallel_rank

        runner = self.model_runner
        language = getattr(runner.model, 'language_model', runner.model)
        speculator = runner.speculator
        result = nv_adapter.report()
        if result['layers_seen'] != 42 or result['source_bytes'] != 9512681472:
            raise RuntimeError('Incomplete ModelOpt expert-scale coverage')
        if result['mode'] == 'packed' and result['raw_scale_owners_alive']:
            raise RuntimeError('Raw scale owners remain alive after replacement')
        if not sources:
            result.pop('layers')
        draft = speculator.model
        result.update(
            rank=get_tensor_model_parallel_rank(),
            target_class=type(runner.model).__name__, draft_class=type(draft).__name__,
            shared_head=draft.lm_head is language.lm_head,
            shared_embedding=draft.model.embed_tokens is language.model.embed_tokens,
            head_dtype=str(language.lm_head.weight.dtype),
            draft_context_dtype=str(draft.model._fused_kv_weight.dtype),
            graph_token_counts=sorted({d.num_tokens for d in runner.cudagraph_manager.graphs}),
            max_model_len=runner.vllm_config.model_config.max_model_len,
            max_num_seqs=runner.vllm_config.scheduler_config.max_num_seqs,
            max_num_batched_tokens=runner.vllm_config.scheduler_config.max_num_batched_tokens,
            prefix_caching=runner.vllm_config.cache_config.enable_prefix_caching,
            kv_cache_memory_bytes=runner.vllm_config.cache_config.kv_cache_memory_bytes,
            production_flags={k: v for k, v in os.environ.items() if k.startswith('GLM_')},
            full_model_qualified=False,
        )
        if (not result['shared_head'] or not result['shared_embedding']
                or result['draft_context_dtype'] != 'torch.bfloat16'):
            raise RuntimeError('Production target/draft sharing changed')
        return result

    def nv_trace(self, enabled: bool, label: str = '') -> dict[str, Any]:
        """Observe existing CPU batch metadata only in explicitly labelled quality groups."""
        from c8_forward_trace import ForwardTrace
        from vllm.distributed import get_tensor_model_parallel_rank

        if not hasattr(self, '_nv_trace'):
            self._nv_trace = ForwardTrace(self.model_runner)
        return {'rank': get_tensor_model_parallel_rank(),
                **self._nv_trace.configure(enabled, label)}
