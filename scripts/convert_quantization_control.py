"""Convert one predeclared corrected control and record its resource measurements."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import torch

from mxwave.checkpoint import _atomic_write_text, _file_integrity
from mxwave.core import QUANTIZATION_REVISION
from mxwave.engine import QuantizeConfig, quantize_model


def main() -> None:
    """Run a fixed RTN, unweighted MSE, norm fallback, or H64 control."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("rtn", "mse", "norm", "h64"), required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = Path(args.report)
    if report.exists():
        raise FileExistsError(f"Refusing to overwrite resource measurements: {report}")
    torch.cuda.reset_peak_memory_stats()
    config = QuantizeConfig(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        policy="qwen3.8-27b-compatible",
        method="rtn" if args.case == "rtn" else "mse",
        mse_clip_depth=4,
        scale_percentile=99.5,
        gamma_proxy=args.case == "norm",
        activation_stats=args.calibration if args.case == "h64" else None,
        calibration_objective="block-hessian",
        verify_sqnr=True,
        resume=args.resume,
        source_repository="Qwen/Qwen3.8-27B",
        source_revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    )
    started = time.monotonic()
    count = quantize_model(config)
    measurement = {
        "case": args.case,
        "quantization_revision": QUANTIZATION_REVISION,
        "converter_script_sha256": _file_integrity(Path(__file__)).sha256,
        "elapsed_seconds": time.monotonic() - started,
        "resumed": args.resume,
        "scope": (
            "resume validation and remaining emission; excludes the earlier interrupted attempt"
            if args.resume
            else "complete conversion including source hashes, fsync, shard hashes and SQNR"
        ),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "shard_count": count,
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "manifest_sha256": _file_integrity(Path(args.output_dir) / "mxwave-manifest.json").sha256,
    }
    _atomic_write_text(report, json.dumps(measurement, indent=2, sort_keys=True) + "\n")
    print(json.dumps(measurement), flush=True)


if __name__ == "__main__":
    main()
