"""Command-line entry point for safe, resumable MXFP4 conversion."""

from __future__ import annotations

import argparse
import json
import sys

from .engine import QuantizeConfig, plan_model, quantize_model


def build_parser() -> argparse.ArgumentParser:
    """Build the ``mxstream-quantize`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="mxstream-quantize",
        description="Stream a BF16/FP16 checkpoint into vLLM MXFP4 shards.",
    )
    parser.add_argument("--model-dir", "--model_dir", required=True, help="Input model")
    parser.add_argument("--output-dir", "--output_dir", required=True, help="Output model")
    parser.add_argument(
        "--policy",
        choices=(
            "auto",
            "qwen3.8-27b-mlp",
            "qwen3.8-27b-compatible",
            "all-linear",
        ),
        default="auto",
        help="Architecture-aware tensor selection (auto is conservative)",
    )
    parser.add_argument(
        "--method",
        choices=("rtn", "mse"),
        default="mse",
        help="Reference compressed-tensors RTN or mxstream MSE scale search",
    )
    parser.add_argument(
        "--scale-percentile",
        "--scale_percentile",
        type=float,
        default=99.5,
        help="MSE anchor percentile; ignored by RTN",
    )
    parser.add_argument(
        "--mse-clip-depth",
        type=int,
        default=1,
        help="Exponent steps below the MSE anchor to evaluate (0-8)",
    )
    parser.add_argument(
        "--hessian-rounding-sweeps",
        type=int,
        default=0,
        help="Block-Hessian coordinate-refinement passes after scale selection (0-4)",
    )
    parser.add_argument(
        "--hessian-error-feedback",
        action="store_true",
        help="Experimental block-local GPTQ-style feedback on a fixed MXFP4 grid",
    )
    parser.add_argument(
        "--hessian-feedback-damp-percent",
        type=float,
        default=1.0,
        help="Average-Hessian-diagonal damping percentage for error feedback",
    )
    parser.add_argument(
        "--no-hessian-feedback-activation-order",
        action="store_false",
        dest="hessian_feedback_activation_order",
        help="Use original block-column order instead of static activation ordering",
    )
    parser.add_argument(
        "--hessian-feedback-max-mse-ratio",
        type=float,
        default=None,
        help="Optional 1-4x ordinary-MSE trust region for accepting feedback blocks",
    )
    parser.add_argument(
        "--feedback-selection-stats",
        default=None,
        help="Independent block-Hessian stats used only to accept feedback blocks",
    )
    parser.add_argument(
        "--tensor-row-chunk-size",
        type=int,
        default=1024,
        help="Maximum output rows of one weight tensor processed on device at once",
    )
    parser.add_argument(
        "--activation-stats",
        default=None,
        help="Module-keyed calibration safetensors from mxstream-calibrate",
    )
    parser.add_argument(
        "--calibration-objective",
        choices=("mean-abs", "rms", "block-hessian"),
        default="mean-abs",
        help="Objective to load from --activation-stats",
    )
    parser.add_argument(
        "--no-gamma-proxy",
        action="store_false",
        dest="gamma_proxy",
        help="Disable architecture-aware Qwen RMSNorm weighting for MSE",
    )
    parser.add_argument("--device", default="cuda", help="Quantization device")
    parser.add_argument(
        "--source-repository",
        default=None,
        help="Optional source repository recorded in the manifest",
    )
    parser.add_argument(
        "--source-revision",
        default=None,
        help="Optional immutable source revision recorded in the manifest",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Validate and reuse complete output shards from an interrupted run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect headers and print the plan without creating output",
    )
    parser.add_argument(
        "--verify-sqnr",
        action="store_true",
        help="Record bounded per-tensor SQNR samples in the manifest",
    )
    parser.add_argument(
        "--sqnr-rows",
        type=int,
        default=16,
        help="Leading output rows sampled per target for SQNR",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the converter and return a process exit status."""
    args = build_parser().parse_args(argv)
    config = QuantizeConfig(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        device=args.device,
        policy=args.policy,
        method=args.method,
        scale_percentile=args.scale_percentile,
        mse_clip_depth=args.mse_clip_depth,
        hessian_rounding_sweeps=args.hessian_rounding_sweeps,
        hessian_error_feedback=args.hessian_error_feedback,
        hessian_feedback_damp_percent=args.hessian_feedback_damp_percent,
        hessian_feedback_activation_order=args.hessian_feedback_activation_order,
        hessian_feedback_max_mse_ratio=args.hessian_feedback_max_mse_ratio,
        feedback_selection_stats=args.feedback_selection_stats,
        tensor_row_chunk_size=args.tensor_row_chunk_size,
        activation_stats=args.activation_stats,
        calibration_objective=args.calibration_objective,
        gamma_proxy=args.gamma_proxy,
        resume=args.resume,
        verify_sqnr=args.verify_sqnr,
        sqnr_rows=args.sqnr_rows,
        source_repository=args.source_repository,
        source_revision=args.source_revision,
    )
    try:
        if args.dry_run:
            print(json.dumps(plan_model(config).summary(), indent=2))
            return 0
        processed = quantize_model(config)
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, ValueError) as exc:
        print(f"mxstream: error: {exc}", file=sys.stderr)
        return 1
    print(f"[mxstream] complete: {processed} shard(s) -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
