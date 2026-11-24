"""Fail at loaded-weight/chat correctness before measuring a matched selective KDA arm."""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path


def main() -> None:
    """Use existing qualified diagnostic and benchmark clients with a frozen protocol."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("stock", "kda"), required=True)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--vocabulary", required=True, type=int)
    parser.add_argument("--fused-max-tokens", type=int, choices=(1, 8), default=1)
    args = parser.parse_args()
    source = Path(__file__).parent
    diagnostic = args.root / "diagnostic"
    diagnostic.mkdir(exist_ok=False)
    sys.argv = [str(source / "check_full_model.py"), "--root", str(diagnostic),
                "--protocol", str(args.root / "protocol.json"), "--vocabulary", str(args.vocabulary)]
    runpy.run_path(sys.argv[0], run_name="__main__")
    result = json.loads((diagnostic / "model-diagnostic.json").read_bytes())
    if result["status"] != "complete" or not all(p["sanity_passed"] for p in result["phases"]):
        raise RuntimeError("Loaded-weight/final-answer gate failed; no throughput sweep")
    if args.mode == "kda":
        from benchmark_full_model import allocations

        for row in allocations("http://127.0.0.1:18784")["results"]:
            state = row["kda"]
            if (state["resident_layers"] != 68 or state["dense_parameter_references_alive"]
                    or state["pool_bytes"] != 206569472
                    or state.get("fused_max_tokens", 1) != args.fused_max_tokens):
                raise RuntimeError("Incomplete selective KDA ownership or unexpected scratch")
    sys.argv = [str(source / "benchmark_full_model.py"), "--mode", args.mode,
                "--root", str(args.root), "--vocabulary", str(args.vocabulary)]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
