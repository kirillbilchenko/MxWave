"""CLI for adapter-driven routed-expert MXFP4 conversion."""

from __future__ import annotations

import argparse
import json
import sys

from .expert_engine import ExpertQuantizationConfig, plan_expert_model, quantize_expert_model


def build_parser() -> argparse.ArgumentParser:
    """Build the experimental expert MXFP4 command parser."""
    parser = argparse.ArgumentParser(
        prog="mxwave-quantize-experts",
        description=(
            "Experimentally quantize routed experts from a supported, validated "
            "fused-MoE layout with weight-only MXFP4 using RTN or unweighted-MSE "
            "scale search."
        ),
    )
    parser.add_argument("--model-dir", required=True, help="Plain BF16/FP16 source model")
    parser.add_argument("--output-dir", required=True, help="Output model directory")
    parser.add_argument("--device", default="cuda", help="Quantization device")
    parser.add_argument(
        "--method",
        choices=("rtn", "mse"),
        default="rtn",
        help="Reference RTN (default) or unweighted-MSE scale search",
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
        default=4,
        help="Exponent steps below the MSE anchor to evaluate (0-8); ignored by RTN",
    )
    parser.add_argument(
        "--tensor-row-chunk-size",
        type=int,
        default=2048,
        help="Maximum logical expert rows processed on device at once",
    )
    parser.add_argument(
        "--host-tensor-cap-mib",
        type=int,
        default=1024,
        help=(
            "Hard cap in MiB for materialized tensor payload per output shard; "
            "serializer and filesystem-cache overhead are additional"
        ),
    )
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
        help="Validate and reuse complete cap-bounded output shards",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate headers and print the exact emission plan only",
    )
    parser.add_argument(
        "--verify-sqnr",
        action="store_true",
        help="Record bounded reconstruction SQNR for every logical expert matrix",
    )
    parser.add_argument(
        "--sqnr-rows",
        type=int,
        default=16,
        help="Leading logical output rows sampled per expert matrix for SQNR",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the routed-expert converter and return a process exit status."""
    args = build_parser().parse_args(argv)
    config = ExpertQuantizationConfig(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        device=args.device,
        method=args.method,
        scale_percentile=args.scale_percentile,
        mse_clip_depth=args.mse_clip_depth,
        tensor_row_chunk_size=args.tensor_row_chunk_size,
        host_tensor_cap_bytes=args.host_tensor_cap_mib * 1024**2,
        resume=args.resume,
        verify_sqnr=args.verify_sqnr,
        sqnr_rows=args.sqnr_rows,
        source_repository=args.source_repository,
        source_revision=args.source_revision,
    )
    try:
        if args.dry_run:
            print(json.dumps(plan_expert_model(config).summary(), indent=2))
            return 0
        processed = quantize_expert_model(config)
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"mxwave: error: {exc}", file=sys.stderr)
        return 1
    method = args.method.upper()
    print(f"[mxwave] complete: {processed} expert {method} shard(s) -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
