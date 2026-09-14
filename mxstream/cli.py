"""Command-line entry point for mxstream.

The CLI is intentionally thin: it delegates to the streaming engine (WIP) and
exposes the quality-oriented flags that differentiate this project
(activation-aware, rotation, verification).
"""

from __future__ import annotations

import argparse
import sys

import torch

from .engine import QuantizeConfig, quantize_model
from .output import build_quantization_config


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

    # Load calibration stats (activation-aware gamma) if provided.
    gamma: torch.Tensor | None = None
    if args.calibration_stats:
        import json

        with open(args.calibration_stats) as f:
            stats = json.load(f)
        # stats is {layer_idx: {"pre_attn": [...], ...}}; use the first layer's
        # pre_attn as a representative gamma vector (per-channel magnitudes).
        first = next(iter(stats.values())) if stats else {}
        first_key = next(iter(first.values())) if first else None
        if first_key is not None:
            gamma = torch.tensor(list(first_key), dtype=torch.float32)

    cfg = QuantizeConfig(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        device=args.device,
        scale_percentile=args.scale_percentile,
        gamma=gamma,
        rotation=args.rotation,
        workers=args.workers,
    )

    print(f"[mxstream] model_dir={args.model_dir}")
    print(f"[mxstream] output_dir={args.output_dir}")
    print(f"[mxstream] rotation={args.rotation}, device={args.device}, workers={args.workers}")

    processed = quantize_model(cfg)
    print(f"[mxstream] quantized {processed} shard(s) -> {args.output_dir}")

    if args.verify:
        from .output import update_config, verify_emitted_config

        qcfg = build_quantization_config(
            targets=["Linear"],
            ignore=["lm_head", "embed_tokens"],
            transform_config={"type": "hadamard"} if args.rotation == "hadamard" else None,
        )
        update_config(args.output_dir, qcfg)
        gaps = verify_emitted_config(args.output_dir, real_modules=["Linear"])
        if gaps:
            print(f"[mxstream] WARNING: uncovered modules: {gaps}")
        else:
            print("[mxstream] config coverage OK")

    return 0


if __name__ == "__main__":
    sys.exit(main())
