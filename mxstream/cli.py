"""Command-line entry point for mxstream.

The CLI is intentionally thin: it delegates to the streaming engine (WIP) and
exposes the quality-oriented flags that differentiate this project
(activation-aware, rotation, verification).
"""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mxstream-quantize",
        description="GPU-streaming, calibration-aware MXFP4 quantization.",
    )
    p.add_argument("--model_dir", required=True, help="Path to input model")
    p.add_argument("--output_dir", required=True, help="Path to output model")
    p.add_argument("--workers", type=int, default=4, help="Parallel shard workers")
    p.add_argument(
        "--scale_percentile",
        type=float,
        default=99.5,
        help="Percentile anchoring block_max (100 = true amax)",
    )
    p.add_argument(
        "--calibration_stats",
        default=None,
        help="Path to activation stats JSON (activates gamma-weighted MSE)",
    )
    p.add_argument(
        "--rotation",
        choices=["hadamard", "none"],
        default="none",
        help="Rotation-based outlier suppression (QuaRot-style)",
    )
    p.add_argument(
        "--verify",
        action="store_true",
        help="Run SQNR + config-coverage verification before finishing",
    )
    p.add_argument("--device", default="cuda", help="Device for quantization kernel")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"[mxstream] model_dir={args.model_dir}")
    print(f"[mxstream] output_dir={args.output_dir}")
    print("[mxstream] The streaming engine is under construction; this CLI is a scaffold.")
    print(
        f"[mxstream] requested: workers={args.workers}, scale_percentile={args.scale_percentile}, "
        f"rotation={args.rotation}, verify={args.verify}, device={args.device}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
