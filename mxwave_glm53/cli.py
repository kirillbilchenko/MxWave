"""Prepare assets, inspect installation and launch CT without experiment scripts."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

from .bundle import (
    DRAFTER_FILES,
    MODEL_FILES,
    VLLM_VERSION,
    prepare_bundle,
    verify_bundle,
    verify_files,
)


def doctor(bundle: Path | None = None, *, gpu: bool = False) -> dict[str, Any]:
    """Inspect dependencies and assets without loading a model or changing production."""
    versions: dict[str, str | None] = {}
    for name in ("mxwave", "vllm", "torch", "transformers", "compressed-tensors"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    errors = []
    if versions["vllm"] != VLLM_VERSION:
        errors.append(f"Require vllm=={VLLM_VERSION}; found {versions['vllm']}")
    if platform.system() != "Linux" or platform.machine() not in ("aarch64", "arm64"):
        errors.append("This native bundle targets Linux ARM64 on DGX Spark (SM121)")
    upstream_checked = False
    if versions["vllm"] == VLLM_VERSION:
        from .plugin import check_upstream

        try:
            check_upstream()
            upstream_checked = True
        except (RuntimeError, OSError) as exc:
            errors.append(str(exc))
    manifest = verify_bundle(bundle) if bundle is not None else None
    device: dict[str, Any] | None = None
    if gpu:
        import torch

        if not torch.cuda.is_available():
            errors.append("CUDA is unavailable")
        else:
            capability = list(torch.cuda.get_device_capability())
            device = {"name": torch.cuda.get_device_name(), "capability": capability}
            if capability != [12, 1]:
                errors.append("The supplied native kernels require SM121")
    return {
        "status": "failed" if errors else "passed", "versions": versions,
        "bundle_verified": manifest is not None, "device": device, "errors": errors,
        "upstream_contract_verified": upstream_checked,
        "scope": "installation preflight; no full-model speed or quality qualification",
    }


def launch_command(args: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    """Build explicit upstream CLI arguments and relocatable worker configuration."""
    if args.node_rank not in (0, 1) or args.kv_cache_gib <= 0:
        raise ValueError("Require rank 0/1 and a positive KV budget")
    if args.max_model_len < 1 or not 1 <= args.port <= 65535:
        raise ValueError("Require a positive context limit and a valid API port")
    max_seqs = getattr(args, "max_num_seqs", 16)
    prefill_budget = getattr(args, "max_num_batched_tokens", 4096)
    graph_sizes = getattr(args, "graph_sizes", [1, 2, 4, 8, 16])
    prefix_caching = getattr(args, "prefix_caching", False)
    vocab_mode = getattr(args, "vocab_mode", "all")
    utilization = getattr(args, "gpu_memory_utilization", 0.85)
    block_size = getattr(args, "block_size", 2304)
    load_format = getattr(args, "load_format", "auto")
    loader_memory_mib = getattr(args, "loader_memory_mib", 2048)
    loader_concurrency = getattr(args, "loader_concurrency", 8)
    draft_attention_backend = getattr(args, "draft_attention_backend", "auto")
    if (not 1 <= max_seqs <= 32 or prefill_budget < max_seqs
            or not graph_sizes or sorted(set(graph_sizes)) != graph_sizes
            or graph_sizes[0] < 1 or graph_sizes[-1] > 32
            or vocab_mode not in ("all", "embedding")
            or not 0 < utilization <= 1 or block_size < 1
            or load_format not in ("auto", "runai_streamer")
            or draft_attention_backend not in ("auto", "TRITON_ATTN", "FLASHINFER")
            or loader_memory_mib < 1 or loader_concurrency < 1):
        raise ValueError("Invalid serving capacity, graph sizes or vocabulary mode")
    verify_bundle(args.bundle)
    verify_files(args.model, MODEL_FILES)
    verify_files(args.drafter, DRAFTER_FILES)
    shards = json.loads((args.model / "model.safetensors.index.json").read_bytes())["weight_map"]
    for name in set(shards.values()):
        if Path(name).name != name or not (args.model / name).is_file():
            raise ValueError(f"Missing local checkpoint shard: {name}")
    spec = {
        "method": "dflash", "model": str(args.drafter.resolve()),
        "num_speculative_tokens": 5, "kv_cache_dtype": "fp8_e4m3",
        "num_speculative_tokens_per_batch_size": [
            [lo, min(hi, max_seqs), depth]
            for lo, hi, depth in ((1, 1, 5), (2, 2, 5), (3, 32, 3)) if lo <= max_seqs
        ],
        "rejection_sample_method": "standard",
    }
    if draft_attention_backend != "auto":
        spec["attention_backend"] = draft_attention_backend
    compile_config = {
        "mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": graph_sizes, "max_cudagraph_capture_size": graph_sizes[-1],
    }
    command = [
        sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", str(args.model.resolve()),
        "--served-model-name", "glm-5.3-flash", "--dtype", "bfloat16",
        "--tensor-parallel-size", "2", "--nnodes", "2", "--node-rank", str(args.node_rank),
        "--master-addr", args.master_addr, "--master-port", str(args.master_port),
        "--distributed-executor-backend", "mp", "--max-model-len", str(args.max_model_len),
        "--kv-cache-dtype", "fp8_e4m3",
        "--kv-cache-memory-bytes", str(int(args.kv_cache_gib * 1024**3)),
        "--gpu-memory-utilization", str(utilization), "--block-size", str(block_size),
        "--max-num-seqs", str(max_seqs), "--max-num-batched-tokens", str(prefill_budget),
        "--moe-backend", "marlin", "--linear-backend", "marlin",
        "--enable-prefix-caching" if prefix_caching else "--no-enable-prefix-caching",
        "--enable-chunked-prefill",
        "--no-enable-flashinfer-autotune",
        "--disable-custom-all-reduce", "--seed", "42",
        "--compilation-config", json.dumps(compile_config),
        "--speculative-config", json.dumps(spec),
        "--limit-mm-per-prompt", '{"image":0,"video":0}',
        "--reasoning-parser", "glm45", "--tool-call-parser", "glm47",
        "--enable-auto-tool-choice", "--generation-config", "vllm",
        "--default-chat-template-kwargs", '{"reasoning_effort":"low"}',
        "--host", args.host, "--port", str(args.port),
        "--worker-extension-cls", "mxwave_glm53.worker.WorkerExtension",
    ]
    if args.node_rank == 1:
        command.append("--headless")
    command += ["--load-format", load_format]
    if load_format == "runai_streamer":
        command += ["--model-loader-extra-config", json.dumps({
            "memory_limit": loader_memory_mib * 1024**2, "concurrency": loader_concurrency,
        })]
    env = {
        "VLLM_PLUGINS": "mxwave_glm53", "MXWAVE_GLM53_ENABLE": "1",
        "MXWAVE_GLM53_BUNDLE": str(args.bundle.resolve()),
        "MXWAVE_GLM53_MODEL": str(args.model.resolve()),
        "MXWAVE_GLM53_DRAFTER": str(args.drafter.resolve()),
        "MXWAVE_GLM53_REPORTS": str(args.reports.resolve()),
        "MXWAVE_GLM53_VOCAB_MODE": vocab_mode,
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }
    return command, env


def main() -> None:
    """Run preparation or launch a version-checked upstream vLLM process."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    prepare = commands.add_parser("prepare-bundle", help="Export existing verified runtime assets")
    prepare.add_argument("--from-experiment", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("doctor", help="Check installation; GPU inspection is optional")
    check.add_argument("--bundle", type=Path)
    check.add_argument("--gpu", action="store_true")
    serve = commands.add_parser("serve", help="Launch the selected five-draft CT profile")
    for name in ("bundle", "model", "drafter", "reports"):
        serve.add_argument("--" + name, type=Path, required=True)
    serve.add_argument("--node-rank", type=int, required=True)
    serve.add_argument("--master-addr", required=True)
    serve.add_argument("--master-port", type=int, default=29719)
    serve.add_argument("--port", type=int, default=8046)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--max-model-len", type=int, default=8192)
    serve.add_argument("--kv-cache-gib", type=float, default=6)
    serve.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    serve.add_argument("--block-size", type=int, default=2304)
    serve.add_argument("--load-format", choices=("auto", "runai_streamer"), default="auto")
    serve.add_argument("--loader-memory-mib", type=int, default=2048)
    serve.add_argument("--loader-concurrency", type=int, default=8)
    serve.add_argument("--draft-attention-backend", default="auto",
                       choices=("auto", "TRITON_ATTN", "FLASHINFER"),
                       help="Native drafter backend; Triton avoids small-page hybrid cache waste")
    serve.add_argument("--max-num-seqs", type=int, default=16)
    serve.add_argument("--max-num-batched-tokens", type=int, default=4096)
    serve.add_argument("--graph-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    serve.add_argument("--prefix-caching", action=argparse.BooleanOptionalAction, default=False)
    serve.add_argument("--vocab-mode", choices=("all", "embedding"), default="all",
                       help="Compress both vocabulary matrices, or keep the LM head dense")
    serve.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        if args.action == "prepare-bundle":
            result = prepare_bundle(args.from_experiment, args.output)
        elif args.action == "doctor":
            result = doctor(args.bundle, gpu=args.gpu)
        else:
            command, updates = launch_command(args)
            if args.dry_run:
                result = {"command": command, "environment": updates,
                          "note": "validated assets; server not started"}
            else:
                result = doctor(args.bundle, gpu=True)
                if result["status"] != "passed":
                    raise ValueError("; ".join(result["errors"]))
                args.reports.mkdir(parents=True, exist_ok=True)
                os.execvpe(command[0], command, {**os.environ, **updates})
        print(json.dumps(result, indent=2))
        if result.get("status") == "failed":
            raise SystemExit(1)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"mxwave-glm53: {exc}\n")


if __name__ == "__main__":
    main()
