"""Read-only allocation and drafter evidence for a private CT profile check."""

from __future__ import annotations

from typing import Any


def storage_inventory(module: Any) -> dict[tuple[str, int], int]:
    """Count unique tensor backing stores, including aliased parameters and buffers."""
    result = {}
    for tensor in (*module.parameters(), *module.buffers()):
        if tensor.device.type != "cuda" or not tensor.numel():
            continue
        storage = tensor.untyped_storage()
        result[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
    return result


class ProfileWorker:
    """Expose measurements without altering weights, dispatch or model behavior."""

    def mxwave_profile_report(self) -> dict[str, Any]:
        """Report the loaded drafter, shared vocabulary and actual post-startup allocation."""
        from mxwave_glm53.worker import snapshot

        runner = self.model_runner
        target = runner.model.language_model
        draft = runner.speculator.model
        spec = self.vllm_config.speculative_config
        if spec is None or type(draft).__name__ != "DFlash2Qwen3ForCausalLM":
            raise RuntimeError("Expected the selected native DFlash2 drafter")
        target_stores = storage_inventory(runner.model)
        draft_stores = storage_inventory(draft)
        result = snapshot(self)
        result["profile"] = {
            "runner_class": type(runner).__module__ + "." + type(runner).__name__,
            "speculator_class": type(runner.speculator).__name__,
            "draft_class": type(draft).__name__,
            "shared_head": draft.lm_head is target.lm_head,
            "shared_embedding": draft.model.embed_tokens is target.model.embed_tokens,
            "draft_head_method": type(draft.lm_head.quant_method).__name__,
            "draft_quantization": spec.draft_model_config.quantization,
            "draft_kv_cache_dtype": spec.kv_cache_dtype,
            "draft_depth": spec.num_speculative_tokens,
            "draft_policy": spec.num_speculative_tokens_per_batch_size,
            "max_num_seqs": self.vllm_config.scheduler_config.max_num_seqs,
            "max_num_batched_tokens": self.vllm_config.scheduler_config.max_num_batched_tokens,
            "prefix_caching": self.vllm_config.cache_config.enable_prefix_caching,
            "target_registered_storage_bytes": sum(target_stores.values()),
            "draft_exclusive_registered_storage_bytes": sum(
                size for key, size in draft_stores.items() if key not in target_stores),
            "target_draft_shared_registered_storage_bytes": sum(
                size for key, size in draft_stores.items() if key in target_stores),
            "storage_scope": "Registered CUDA tensors only; excludes unregistered pools/graphs/KV",
        }
        if not result["profile"]["shared_head"] or not result["profile"]["shared_embedding"]:
            raise RuntimeError("Drafter vocabulary is not shared with the target")
        return result
