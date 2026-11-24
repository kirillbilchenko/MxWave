"""Construct the pinned native speculative model configuration without loading weights or CUDA."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    """Resolve actual target/draft metadata and require inherited CT precision and TP2."""
    import torch
    from mtp_mapping import install
    from vllm import platforms
    from vllm.platforms.cpu import CpuPlatform

    platforms.current_platform = CpuPlatform()
    from vllm.config import ModelConfig, ParallelConfig, SpeculativeConfig

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("Preserve the first native MTP configuration preflight")
    install()
    target = ModelConfig(model=str(args.source), dtype="bfloat16", max_model_len=8192,
                         seed=42, trust_remote_code=False)
    parallel = ParallelConfig(tensor_parallel_size=2, distributed_executor_backend="mp")
    spec = SpeculativeConfig(method="mtp", num_speculative_tokens=2,
        use_local_argmax_reduction=True, target_model_config=target,
        target_parallel_config=parallel, kv_cache_dtype="fp8_e4m3", moe_backend="marlin")
    draft = spec.draft_model_config
    assert draft is not None and spec.draft_parallel_config is not None
    from vllm.model_executor.model_loader.utils import get_model_architecture
    from vllm.models.glm5next.nvidia.mtp import Glm5NextMTP

    resolved, architecture = get_model_architecture(draft)
    args.report.with_suffix(".incomplete.json").write_text(json.dumps({
        "draft_architectures": draft.hf_config.architectures,
        "resolved_architecture": architecture,
        "resolved_class": resolved.__module__ + "." + resolved.__name__,
        "cuda_initialized": torch.cuda.is_initialized()}, indent=2) + "\n")
    assert draft.quantization == target.quantization == "compressed-tensors"
    assert draft.dtype == target.dtype == torch.bfloat16
    assert draft.max_model_len == target.max_model_len == 8192
    assert spec.draft_parallel_config.tensor_parallel_size == 2
    assert draft.hf_config.architectures == ["Glm5NextMTPModel"] and resolved is Glm5NextMTP
    assert getattr(draft.hf_config, "index_share_for_mtp_iteration", False)
    assert not torch.cuda.is_initialized()
    result = {"status": "passed", "cuda_initialized": False, "model_loaded": False,
        "target_architectures": target.hf_config.architectures,
        "draft_architectures": draft.hf_config.architectures, "method": str(spec.method),
        "resolved_class": resolved.__module__ + "." + resolved.__name__,
        "num_speculative_tokens": spec.num_speculative_tokens,
        "use_local_argmax_reduction": spec.use_local_argmax_reduction,
        "kv_cache_dtype": spec.kv_cache_dtype, "moe_backend": str(spec.moe_backend),
        "draft_quantization": draft.quantization, "draft_dtype": str(draft.dtype),
        "draft_tp": spec.draft_parallel_config.tensor_parallel_size,
        "draft_max_model_len": draft.max_model_len,
        "source_config_sha256": hashlib.sha256((args.source / "config.json").read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
