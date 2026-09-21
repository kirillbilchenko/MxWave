"""Command-line entry point for reproducible model qualification."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .qualification import run_qualification, verify_checkpoint


def build_parser() -> argparse.ArgumentParser:
    """Build the ``mxwave-qualify`` parser."""
    parser = argparse.ArgumentParser(
        prog="mxwave-qualify",
        description=(
            "Verify a checkpoint, collect versioned evaluation evidence, and apply "
            "predeclared qualification gates."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect = subparsers.add_parser(
        "inspect", help="run whole-checkpoint structural verification only"
    )
    inspect.add_argument("--model-dir", type=Path, required=True)
    inspect.add_argument("--hash-shards", action="store_true")

    run = subparsers.add_parser(
        "run", help="execute one immutable qualification specification"
    )
    run.add_argument("--spec", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument(
        "--resume",
        action="store_true",
        help="reuse existing evidence only when the specification hash matches",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run qualification and return 0=qualified, 2=rejected/incomplete, 1=error."""
    args = build_parser().parse_args(argv)
    try:
        if args.command == "inspect":
            report = verify_checkpoint(args.model_dir, hash_shards=args.hash_shards)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        if args.command == "run":
            report, path = run_qualification(
                args.spec,
                args.output_dir,
                resume=args.resume,
            )
            decision = report["decision"]
            print(f"[mxwave] qualification: {decision} -> {path}")
            return 0 if decision in {"qualified-default", "qualified-optional"} else 2
        raise AssertionError(f"Unhandled command: {args.command}")
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"mxwave-qualify: error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
