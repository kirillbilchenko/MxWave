"""Execute a frozen candidate command; no installed vLLM source is replaced."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> None:
    """Bind the selected quality/timing mode and the stage's report destination."""
    mode, rank, stage = sys.argv[1:]
    if mode not in ("timing", "gsm8k") or rank not in ("0", "1"):
        raise ValueError("Unexpected frozen launch")
    plan = json.loads((Path("/trial") / f"launch-{mode}-rank{rank}.json").read_bytes())
    if ("--speculative-config" in plan["command"]) != (mode == "timing"):
        raise RuntimeError("Speculation must be disabled for GSM8K")
    os.environ.update(plan["environment"])
    os.environ["MXWAVE_GLM53_REPORTS"] = "/reports/" + stage
    os.execvpe(plan["command"][0], plan["command"], os.environ)


if __name__ == "__main__":
    main()
