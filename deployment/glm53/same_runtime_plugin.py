"""Use existing CT codecs on unchanged production overlays; retain startup evidence."""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
from typing import Any


def install() -> None:
    """Register the existing model adapters and the same observation hooks in both arms."""
    from vllm.v1.worker.gpu_worker import Worker

    if getattr(Worker.load_model, "_mxwave_same_runtime", False):
        return
    if os.environ["MXWAVE_COMPARE_ARM"] == "B":
        import model_plugin
        import vocab_adapter

        # The recipe explicitly retains a dense head for production's head hooks.
        # Restrict our own vocabulary selector; do not modify a production overlay.
        targets = vocab_adapter.load_targets
        vocab_adapter.load_targets = lambda path: {
            k: v for k, v in targets(path).items() if v.kind == "embedding"}
        model_plugin.install()
    original_load = Worker.load_model

    @functools.wraps(original_load)
    def loaded(worker: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_load(worker, *args, **kwargs)
        if os.environ["MXWAVE_COMPARE_ARM"] == "B":
            import kda_mxfp8_runtime
            import torch
            from vllm.distributed import get_tensor_model_parallel_rank

            before = torch.get_default_dtype()
            try:
                torch.set_default_dtype(torch.bfloat16)
                rank = get_tensor_model_parallel_rank()
                report = kda_mxfp8_runtime.install(
                    worker.model_runner.model, Path("/kda-runtime/rounding-reference.json"),
                    Path("/reports") / f"kda-r{rank}.json")
            finally:
                torch.set_default_dtype(before)
            worker.model_runner.model_memory_usage += (
                report["cuda_allocated_after"] - report["cuda_allocated_before"])
            worker._mxwave_same_runtime_kda = report
        return result

    loaded._mxwave_same_runtime = True
    Worker.load_model = loaded
    original_warm = Worker.compile_or_warm_up_model

    @functools.wraps(original_warm)
    def warmed(worker: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_warm(worker, *args, **kwargs)
        report = SameRuntimeWorker.mxwave_same_runtime_report(worker)
        target = Path("/reports") / f"startup-r{report['rank']}.json"
        target.write_text(json.dumps(report, indent=2) + "\n")
        print("MxWave same-runtime startup verified: " + str(target), flush=True)
        return result

    Worker.compile_or_warm_up_model = warmed


class SameRuntimeWorker:
    """Read-only evidence for the actual loaded model, graph and adapter allocation."""

    def mxwave_same_runtime_report(self) -> dict[str, Any]:
        """Fail before timing if required CT components or production capacity are absent."""
        import torch
        from vllm.distributed import get_tensor_model_parallel_rank

        runner = self.model_runner
        target = runner.model
        language = getattr(target, "language_model", target)
        head = language.lm_head
        config = self.vllm_config
        report: dict[str, Any] = {
            "arm": os.environ["MXWAVE_COMPARE_ARM"],
            "rank": get_tensor_model_parallel_rank(),
            "max_model_len": config.model_config.max_model_len,
            "max_num_seqs": config.scheduler_config.max_num_seqs,
            "max_num_batched_tokens": config.scheduler_config.max_num_batched_tokens,
            "kv_cache_memory_bytes": config.cache_config.kv_cache_memory_bytes,
            "enable_prefix_caching": config.cache_config.enable_prefix_caching,
            "head_weight_dtype": str(head.weight.dtype),
            "head_quant_method": type(head.quant_method).__name__,
            "cuda_allocated_bytes": torch.cuda.memory_allocated(),
            "cuda_reserved_bytes": torch.cuda.memory_reserved(),
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "model_memory_usage": runner.model_memory_usage,
            "speculation": config.speculative_config is not None,
            "physical_graph_token_counts": sorted({d.num_tokens for d in
                                                   runner.cudagraph_manager.graphs}),
        }
        if (report["max_model_len"] != 500000 or report["max_num_seqs"] != 32
                or report["max_num_batched_tokens"] != 6919
                or report["kv_cache_memory_bytes"] != 6 * 1024**3
                or not report["enable_prefix_caching"] or head.weight.dtype != torch.bfloat16):
            raise RuntimeError("Required production capacity/dense-head configuration did not load")
        if report["speculation"]:
            draft = runner.speculator.model
            report.update(shared_head=draft.lm_head is head,
                          draft_layers=len(draft.model.layers),
                          draft_context_dtype=str(draft.model._fused_kv_weight.dtype))
            if not report["shared_head"] or report["draft_layers"] != 5:
                raise RuntimeError("Require the same five-layer Inco drafter sharing the dense head")
        if report["arm"] == "B":
            import model_plugin
            import vocab_runtime

            report["experts"] = model_plugin.report()
            report["vocabulary"] = vocab_runtime.report()
            report["kda"] = self._mxwave_same_runtime_kda
            if (report["experts"]["layers_seen"] != 42
                    or report["vocabulary"]["resident_layers"] != 1
                    or language.model.embed_tokens.weight is not None
                    or len(report["kda"]["layers"]) != 68
                    or report["kda"]["original_BF16_parameters_alive"] != 0):
                raise RuntimeError("The selected CT compression/MXFP8 recipe did not load completely")
        # Record hook activation separately from nominal flags; shape eligibility can differ.
        import glm_dense_fast
        import glm_marlin_tune

        report["dense_fast_report"] = glm_dense_fast.S.get("report")
        report["marlin_tune_stats"] = {
            key: (str(value) if isinstance(value, dict) else value)
            for key, value in glm_marlin_tune.stats().items()}
        report["overlay_source_hashes"] = {
            module.__name__: hashlib.sha256(Path(inspect.getfile(module)).read_bytes()).hexdigest()
            for module in (glm_dense_fast, glm_marlin_tune)}
        report["existing_KL_gate_passed"] = False
        return report
