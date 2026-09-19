"""CLI for bounded exact-symmetry MXFP4 layout probes."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

from .calibration import load_calibration_data
from .calibration_cli import (
    _checkpoint_files,
    _load_corpus,
    _load_transformers,
    _model_class,
    _peak_rss_bytes,
    _sequence_sha256,
    _sha256_file,
    _tokenize_sequences,
)
from .counteraction_probe import CounteractionProbeResult
from .engine import QuantizeConfig, plan_model
from .exact_layout import EXACT_LAYOUT_CANDIDATES, probe_mlp_exact_layout
from .format import detect_input_format, load_config_json
from .mxfp4_checkpoint import checkpoint_tensor_files
from .runtime_adapters import resolve_runtime_graph


def build_parser() -> argparse.ArgumentParser:
    """Build the exact-layout probe parser."""
    parser = argparse.ArgumentParser(prog="mxwave-exact-layout-probe")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--baseline-model-dir", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--activation-stats", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layers", default="16,42,62")
    parser.add_argument(
        "--policy",
        default="qwen3.8-27b-compatible",
        choices=("qwen3.8-27b-mlp", "qwen3.8-27b-compatible", "all-linear"),
    )
    parser.add_argument("--num-sequences", type=int, default=8)
    parser.add_argument("--sequence-offset", type=int, default=136)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--logit-positions", type=int, default=8)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--scale-percentile", type=float, default=99.5)
    parser.add_argument("--mse-clip-depth", type=int, default=4)
    parser.add_argument("--tensor-row-chunk-size", type=int, default=256)
    parser.add_argument("--max-elapsed-minutes", type=float, default=40.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument(
        "--attention-implementation", choices=("eager", "sdpa"), default="sdpa"
    )
    parser.add_argument("--source-repository", default=None)
    parser.add_argument("--source-revision", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def _parse_layers(raw: str) -> tuple[int, ...]:
    values: list[int] = []
    for item in raw.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        try:
            layer = int(stripped)
        except ValueError as exc:
            raise ValueError(f"Invalid layer index: {stripped!r}") from exc
        if layer <= 0:
            raise ValueError("Exact-layout layers must be positive")
        values.append(layer)
    layers = tuple(dict.fromkeys(values))
    if not layers or len(layers) != len(values):
        raise ValueError("layers must be non-empty and contain no duplicates")
    return layers


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _optional_sha256(path: Path) -> str | None:
    return _sha256_file(path) if path.is_file() else None


def _report(
    args: argparse.Namespace,
    *,
    model_dir: Path,
    baseline_model_dir: Path,
    corpus_path: Path,
    graph: Any,
    stats_sha256: str,
    sequences: list[list[int]],
    results: list[CounteractionProbeResult],
    started: float,
    status: str,
    peak_accelerator: int,
) -> dict[str, Any]:
    return {
        "format": "mxwave-exact-layout-probe-v1",
        "status": status,
        "hypothesis": (
            "An exact gated-MLP channel permutation improves MXFP4 down-projection block "
            "layout without runtime metadata or custom kernels."
        ),
        "source": {
            "model_dir": str(model_dir),
            "repository": args.source_repository,
            "revision": args.source_revision,
            "config_sha256": _sha256_file(model_dir / "config.json"),
            "index_sha256": _optional_sha256(model_dir / "model.safetensors.index.json"),
        },
        "execution": {
            "baseline_model_dir": str(baseline_model_dir),
            "baseline_config_sha256": _sha256_file(baseline_model_dir / "config.json"),
            "baseline_index_sha256": _optional_sha256(
                baseline_model_dir / "model.safetensors.index.json"
            ),
            "expected_baseline_candidate": "block-hessian",
            "attention_implementation": args.attention_implementation,
            "dtype": args.dtype,
        },
        "runtime_graph": graph.as_dict(),
        "calibration": {"objective": "block-hessian", "file_sha256": stats_sha256},
        "heldout": {
            "corpus_sha256": _sha256_file(corpus_path),
            "token_ids_sha256": _sequence_sha256(sequences),
            "num_sequences": len(sequences),
            "sequence_offset": args.sequence_offset,
            "sequence_length": args.sequence_length,
            "logit_positions_per_sequence": args.logit_positions,
        },
        "candidate_specs": {
            "names": list(EXACT_LAYOUT_CANDIDATES),
            "scale_percentile": args.scale_percentile,
            "mse_clip_depth": args.mse_clip_depth,
            "sort": "stable-ascending",
        },
        "requested_layers": list(_parse_layers(args.layers)),
        "layers": [result.as_dict() for result in results],
        "resources": {
            "elapsed_seconds": time.monotonic() - started,
            "max_elapsed_minutes": args.max_elapsed_minutes,
            "peak_process_rss_bytes": _peak_rss_bytes(),
            "peak_accelerator_memory_allocated_bytes": peak_accelerator,
        },
    }


def run(args: argparse.Namespace) -> Path:
    """Execute the bounded exact-layout probe with atomic per-layer reports."""
    layers = _parse_layers(args.layers)
    if args.num_sequences <= 0 or args.sequence_length <= 0:
        raise ValueError("num_sequences and sequence_length must be positive")
    if args.sequence_offset < 0:
        raise ValueError("sequence_offset must be non-negative")
    if not 0 < args.logit_positions <= args.sequence_length:
        raise ValueError("logit_positions must be in [1, sequence_length]")
    if args.tensor_row_chunk_size <= 0 or args.max_elapsed_minutes <= 0.0:
        raise ValueError("row chunk size and elapsed-minute bound must be positive")

    model_dir = Path(args.model_dir)
    baseline_model_dir = Path(args.baseline_model_dir)
    corpus_path = Path(args.corpus)
    stats_path = Path(args.activation_stats)
    output_path = Path(args.output)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    if detect_input_format(baseline_model_dir).kind != "mxfp4":
        raise ValueError("baseline-model-dir must declare MXFP4 in config.json")

    plan = plan_model(
        QuantizeConfig(
            model_dir=model_dir,
            device=device,
            policy=args.policy,
            method="mse",
            gamma_proxy=False,
            verbose=False,
        )
    )
    graph = resolve_runtime_graph(
        load_config_json(model_dir), [item.info.name for item in plan.tensors]
    )
    target_widths = {
        item.info.name: item.info.shape[-1] for item in plan.tensors if item.quantized
    }
    calibration = load_calibration_data(
        stats_path,
        "block-hessian",
        target_widths,
        expected_policy=plan.policy.name,
        expected_source_repository=args.source_repository,
        expected_source_revision=args.source_revision,
    )
    transformers = _load_transformers()
    model_config = transformers.AutoConfig.from_pretrained(
        model_dir, trust_remote_code=args.trust_remote_code
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=args.trust_remote_code
    )
    sequences = _tokenize_sequences(
        tokenizer,
        _load_corpus(corpus_path, args.text_field),
        num_sequences=args.num_sequences,
        sequence_length=args.sequence_length,
        sequence_offset=args.sequence_offset,
    )
    try:
        accelerate = importlib.import_module("accelerate")
    except ImportError as exc:
        raise RuntimeError(
            'Exact-layout dependencies are missing; install with pip install -e ".[calibrate]"'
        ) from exc
    with accelerate.init_empty_weights(include_buffers=True):
        empty_model = _model_class(transformers, model_config).from_config(
            model_config, attn_implementation=args.attention_implementation
        )

    baseline_checkpoint_files = checkpoint_tensor_files(baseline_model_dir)
    source_checkpoint_files = _checkpoint_files(plan)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    results: list[CounteractionProbeResult] = []
    status = "in-progress"
    for layer in layers:
        if results and time.monotonic() - started >= args.max_elapsed_minutes * 60.0:
            status = "bounded-stop"
            break
        print(f"[mxwave] exact-layout layer {layer} start", flush=True)

        def report_progress(completed: int, total: int, selected: int = layer) -> None:
            print(f"[mxwave] exact-layout layer {selected}: decoder {completed}/{total}", flush=True)

        result = probe_mlp_exact_layout(
            empty_model,
            graph,
            calibration,
            layer,
            sequences,
            source_checkpoint_files,
            baseline_checkpoint_files,
            device=device,
            dtype=dtype,
            row_chunk_size=args.tensor_row_chunk_size,
            logit_positions_per_sequence=args.logit_positions,
            scale_percentile=args.scale_percentile,
            mse_clip_depth=args.mse_clip_depth,
            progress=report_progress,
        )
        baseline = next(
            candidate for candidate in result.candidates if candidate.candidate == "block-hessian"
        )
        if baseline.baseline_weight_nmse > 1e-12:
            raise ValueError(
                "Unchanged block-Hessian candidate does not reproduce the packed baseline"
            )
        if max(item.max_recurrence_relative_residual for item in result.candidates) > 1e-10:
            raise ValueError("Counteraction recurrence failed its numerical invariant")
        for name, metadata in result.candidate_metadata.items():
            if not all(
                bool(metadata.get(key))
                for key in (
                    "permutation_bijective",
                    "inverse_layout_exact",
                    "unquantized_output_equivalent",
                )
            ):
                raise ValueError(f"Exact-layout invariant failed for {name!r}")
        results.append(result)
        peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        _atomic_write_json(
            output_path,
            _report(
                args,
                model_dir=model_dir,
                baseline_model_dir=baseline_model_dir,
                corpus_path=corpus_path,
                graph=graph,
                stats_sha256=calibration.file_sha256,
                sequences=sequences,
                results=results,
                started=started,
                status="in-progress",
                peak_accelerator=peak,
            ),
        )

    if status == "in-progress":
        status = "complete"
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    report = _report(
        args,
        model_dir=model_dir,
        baseline_model_dir=baseline_model_dir,
        corpus_path=corpus_path,
        graph=graph,
        stats_sha256=calibration.file_sha256,
        sequences=sequences,
        results=results,
        started=started,
        status=status,
        peak_accelerator=peak,
    )
    _atomic_write_json(output_path, report)
    del empty_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(
        f"[mxwave] wrote {len(results)} layer(s) to {output_path}; "
        f"status={status}, elapsed={report['resources']['elapsed_seconds']:.2f}s",
        flush=True,
    )
    return output_path


def main(argv: list[str] | None = None) -> int:
    """Run the exact-layout probe and return a process status."""
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except (FileNotFoundError, ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"mxwave: exact-layout probe error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
