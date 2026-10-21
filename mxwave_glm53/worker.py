"""Convert audited KDA weights before graph capture and report actual loaded storage."""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path
from typing import Any


def snapshot(worker: Any) -> dict[str, Any]:
    """Return measured allocations and complete compression coverage for this worker."""
    import torch
    from vllm.distributed import get_tensor_model_parallel_rank

    from ._runtime import model_plugin, vocab_runtime

    return {
        "rank": get_tensor_model_parallel_rank(), "vllm": "0.32.0",
        "cuda_allocated_bytes": torch.cuda.memory_allocated(),
        "cuda_reserved_bytes": torch.cuda.memory_reserved(),
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "experts": model_plugin.report(), "vocabulary": vocab_runtime.report(),
        "kda": worker._mxwave_kda_report,
        "max_model_len": worker.vllm_config.model_config.max_model_len,
        "kv_cache_memory_bytes": worker.vllm_config.cache_config.kv_cache_memory_bytes,
        "full_model_quality_qualified": False,
        "upstream_serving_performance_qualified": False,
    }


def install_worker() -> None:
    """Attach startup preparation without altering files inside the vLLM installation."""
    import torch
    from vllm.distributed import get_tensor_model_parallel_rank
    from vllm.v1.worker.gpu_worker import Worker

    from ._runtime.kda_mxfp8_runtime import install

    if getattr(Worker.load_model, "_mxwave_glm53", False):
        return
    original_load = Worker.load_model

    @functools.wraps(original_load)
    def load(worker: Any, *args: Any, **kwargs: Any) -> Any:
        if (worker.vllm_config.parallel_config.tensor_parallel_size != 2
                or worker.vllm_config.model_config.dtype != torch.bfloat16
                or kwargs.get("load_dummy_weights", False)):
            raise ValueError("This GLM profile requires real BF16-activation TP2 loading")
        if torch.cuda.get_device_capability() != (12, 1):
            raise ValueError("This kernel bundle requires SM121")
        result = original_load(worker, *args, **kwargs)
        root = Path(os.environ["MXWAVE_GLM53_BUNDLE"])
        reports = Path(os.environ["MXWAVE_GLM53_REPORTS"])
        reports.mkdir(parents=True, exist_ok=True)
        report_path = reports / f"kda-rank{get_tensor_model_parallel_rank()}.json"
        before = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            report = install(worker.model_runner.model, root / "rounding-reference.json", report_path)
        finally:
            torch.set_default_dtype(before)
        worker.model_runner.model_memory_usage += (
            report["cuda_allocated_after"] - report["cuda_allocated_before"])
        worker._mxwave_kda_report = report
        return result

    load._mxwave_glm53 = True  # type: ignore[attr-defined]
    Worker.load_model = load
    original_warmup = Worker.compile_or_warm_up_model

    @functools.wraps(original_warmup)
    def warmup(worker: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_warmup(worker, *args, **kwargs)
        report = snapshot(worker)
        experts, vocab = report["experts"], report["vocabulary"]
        scales = [s for layer in experts["layers"] for s in layer["scales"]]
        if (experts["layers_seen"] != 42 or len(scales) != 84
                or any(not s.get("gpu_bytes_exact") for s in scales)
                or vocab["resident_layers"] != 2 or vocab["dense_parameter_references_alive"]):
            raise RuntimeError("Incomplete compression coverage or retained dense vocabulary")
        target = Path(os.environ["MXWAVE_GLM53_REPORTS"]) / f"startup-rank{report['rank']}.json"
        target.write_text(json.dumps(report, indent=2) + "\n")
        print(f"MxWave GLM startup verified on rank {report['rank']}: {target}", flush=True)
        return result

    Worker.compile_or_warm_up_model = warmup


class WorkerExtension:
    """Read-only worker RPC for recording final memory with the actual selected drafter."""

    def mxwave_memory_report(self) -> dict[str, Any]:
        """Measure current allocation after model, drafter, KV and graph initialization."""
        return snapshot(self)
