"""Experimental CLI for measured-precision planning and checkpoint composition."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .mixed_precision import build_fp8_composition_plan, compose_fp8_checkpoint
from .precision_budget import build_precision_budget_plan, load_bucket_modules


def build_parser() -> argparse.ArgumentParser:
    """Build the measured-precision command parser."""
    parser = argparse.ArgumentParser(
        prog="mxwave-precision-budget",
        description=(
            "Experimentally allocate a bounded FP8 budget over an MXFP4 checkpoint"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="write the fixed intervention plan")
    plan.add_argument("--primary-model", required=True)
    plan.add_argument("--dense-donor", required=True)
    plan.add_argument("--output", required=True)
    plan.add_argument("--bands", type=int, default=3)
    plan.add_argument("--layers-per-bucket", type=int, default=4)
    plan.add_argument("--max-candidates", type=int, default=12)
    plan.add_argument("--max-premium-bytes", type=int, default=1024**3)

    compose = subparsers.add_parser("compose", help="compose one named FP8 bucket")
    compose.add_argument("--plan", required=True)
    compose.add_argument("--bucket", required=True)
    compose.add_argument("--output", required=True)
    compose.add_argument("--quant-device", default="cpu")
    compose.add_argument("--dry-run", action="store_true")
    compose.add_argument("--quiet", action="store_true")
    return parser


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _run_plan(args: argparse.Namespace) -> None:
    plan = build_precision_budget_plan(
        args.primary_model,
        args.dense_donor,
        bands=args.bands,
        layers_per_bucket=args.layers_per_bucket,
        max_candidates=args.max_candidates,
        max_premium_bytes=args.max_premium_bytes,
    )
    document = plan.as_dict()
    output = Path(args.output)
    _write_json_atomic(output, document)
    print(
        json.dumps(
            {
                "output": str(output),
                "plan_sha256": document["plan_sha256"],
                "candidates": len(plan.buckets),
                "maximum_bucket_premium_bytes": max(
                    bucket.premium_bytes for bucket in plan.buckets
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _run_compose(args: argparse.Namespace) -> None:
    plan_path = Path(args.plan)
    raw_plan = json.loads(plan_path.read_text())
    if not isinstance(raw_plan, dict):
        raise TypeError("Precision-budget plan must be a JSON object")
    primary_model = raw_plan.get("primary_model")
    dense_donor = raw_plan.get("dense_donor")
    if not isinstance(primary_model, str) or not isinstance(dense_donor, str):
        raise TypeError("Precision-budget plan is missing checkpoint paths")
    modules = load_bucket_modules(plan_path, args.bucket)
    composition = build_fp8_composition_plan(primary_model, dense_donor, modules)
    print(json.dumps(composition.summary(), indent=2, sort_keys=True))
    if args.dry_run:
        return
    manifest = compose_fp8_checkpoint(
        composition,
        args.output,
        quant_device=args.quant_device,
        verbose=not args.quiet,
    )
    print(
        json.dumps(
            {
                "output": str(Path(args.output)),
                "mxfp4_target_tensors": manifest["mxfp4_target_tensors"],
                "fp8_target_tensors": manifest["fp8_target_tensors"],
                "actual_output_bytes": manifest["actual_output_bytes"],
                "global_compression_ratio": manifest["global_compression_ratio"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run measured-precision planning or composition."""
    try:
        args = build_parser().parse_args(argv)
        if args.command == "plan":
            _run_plan(args)
        else:
            _run_compose(args)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"mxwave: precision-budget error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
