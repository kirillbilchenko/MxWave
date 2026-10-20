"""Qualify composed storage and actual decode while reusing unchanged stock timing."""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path

from benchmark_full_model import allocations, write
from decode_quality import collect


def main() -> None:
    """Keep the loaded-source/chat gate and measure the new candidate exactly once."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("stock", "kda"), required=True)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--vocabulary", required=True, type=int)
    parser.add_argument("--fused-max-tokens", type=int, choices=(1, 8), default=8)
    parser.add_argument("--reuse-timing", action="store_true")
    args = parser.parse_args()
    if args.reuse_timing and args.mode != "stock":
        raise ValueError("Only unchanged stock timing can be reused")
    source = Path(__file__).parent
    if args.reuse_timing:
        diagnostic = args.root / "combined-diagnostic"
        diagnostic.mkdir(exist_ok=False)
        sys.argv = [str(source / "check_full_model.py"), "--root", str(diagnostic),
            "--protocol", str(args.root / "protocol.json"), "--vocabulary", str(args.vocabulary)]
        runpy.run_path(sys.argv[0], run_name="__main__")
        result = json.loads((diagnostic / "model-diagnostic.json").read_bytes())
        if result["status"] != "complete" or not all(p["sanity_passed"] for p in result["phases"]):
            raise RuntimeError("The fresh stock load did not pass correctness checks")
    else:
        if args.mode == "kda":
            ownership = allocations("http://127.0.0.1:18784")
            write(args.root / "combined-ownership.json", ownership)
            for rank, row in enumerate(ownership["results"]):
                vocabulary = row["vocabulary"]
                scales = [record for layer in row["layers"] for record in layer["scales"]]
                if (vocabulary["resident_layers"] != 2
                        or vocabulary["dense_parameter_references_alive"]
                        or {r["kind"] for r in vocabulary["layers"]} != {"head", "embedding"}
                        or row["layers_seen"] != 42
                        or sum(r["compressed"] for r in scales) < 80):
                    raise RuntimeError("Combined vocabulary/expert-scale ownership gate failed: "
                        f"rank={rank}, vocabulary_layers={vocabulary['resident_layers']}, "
                        f"dense_references={vocabulary['dense_parameter_references_alive']}, "
                        f"kinds={[r['kind'] for r in vocabulary['layers']]}, "
                        f"expert_layers={row['layers_seen']}, "
                        f"compressed_scales={sum(r['compressed'] for r in scales)}")
        sys.argv = [str(source / "measure_kda_arm.py"), "--mode", args.mode,
            "--root", str(args.root), "--vocabulary", str(args.vocabulary),
            "--fused-max-tokens", str(args.fused_max_tokens)]
        runpy.run_path(sys.argv[0], run_name="__main__")
    proto = json.loads((args.root / "protocol.json").read_bytes())
    prompts = [proto.get("quality_prefix_token_ids", []) + ids for ids in proto["quality"]]
    collect("http://127.0.0.1:18784", prompts, args.root, args.mode, args.vocabulary)
    if args.mode == "stock":
        collect("http://127.0.0.1:18784", prompts, args.root, "stock-repeat", args.vocabulary)


if __name__ == "__main__":
    main()
