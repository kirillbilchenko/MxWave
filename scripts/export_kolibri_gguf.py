"""Export selected Kolibri MXFP4 experts and the FP8 backbone to native MXFP4 GGUF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mxwave.kolibri_gguf import export_kolibri_gguf


def main() -> None:
    """Validate the complete model or perform bounded, verified GGUF conversion."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(export_kolibri_gguf(args.model, args.output, dry_run=args.dry_run)), flush=True
    )


if __name__ == "__main__":
    main()
