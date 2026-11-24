"""Collect the fixed native MTP arm, then compare it with retained no-drafter raw evidence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    """Keep the original timing/quality protocol and preserve the failed loader attempt."""
    import analyze_mtp_pilot
    import measure_mtp_pilot

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--vocabulary", type=int, required=True)
    args = parser.parse_args()
    sys.argv = ["measure_mtp_pilot.py", "--root", str(args.root), "--vocabulary", str(args.vocabulary),
        "--label", "candidate", "--data", "/audit/full-model-v1/quality-best-v1-data",
        "--cpu-audit", "/audit/full-model-v1/scale-palette-all-v1/results/screen.json", "--profile"]
    measure_mtp_pilot.main()
    sys.argv = ["analyze_mtp_pilot.py", "--root", str(args.root)]
    analyze_mtp_pilot.main()


if __name__ == "__main__":
    main()
