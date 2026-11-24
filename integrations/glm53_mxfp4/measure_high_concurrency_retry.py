"""Run the unchanged compressed timing grid and compare with explicitly reused stock evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import measure_high_concurrency
from analyze_high_concurrency import review
from benchmark_full_model import write


def main() -> None:
    """Retain all timing, quality and separate profiles, including any failed frozen gates."""
    root_index = sys.argv.index("--root") + 1
    root = Path(sys.argv[root_index])
    plan = json.loads((root / "retry-plan.json").read_bytes())
    if plan["revision"] != "c816retryv2" or not plan["stock_control_reused"]:
        raise ValueError("Missing frozen declaration of reused stock control")
    sys.argv.extend(["--label", "candidate", "--data",
        "/audit/full-model-v1/quality-best-v1-data", "--cpu-audit",
        str(root / "source-audit.json"), "--profile"])
    measure_high_concurrency.main()
    result = review(root)
    result.update(stock_control_reused=True, stock_control_revision="c816v1",
                  comparison_has_time_gap=True, fresh_paired_stock_load=False,
                  retry_plan=plan, throughput_issue_declared_fixed=False)
    write(root / "quality-model-review.json", result)


if __name__ == "__main__":
    main()
