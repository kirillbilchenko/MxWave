"""Retain the entire frozen speed/quality grid and compare the bounded routing intervention."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any

import measure_high_concurrency as measurement
from analyze_high_concurrency import review
from benchmark_full_model import write

native_snapshot = measurement.snapshot


def snapshot() -> list[dict[str, Any]]:
    """Reject missing live-route attestations outside ordinary timed requests."""
    rows = native_snapshot()
    expected = {"enabled": True, "global_fused_limit": 8, "output_exact_rows": [16],
        "input_exception_weights": 0, "output_exception_weights": 34,
        "intermediate_rows_qualified": False, "resident_BF16_cache_bytes": 0}
    if any(row.get("output_row16_policy") != expected for row in rows):
        raise ValueError("The live route differs from the output-only component")
    return rows


def parent_comparison(root: Path) -> list[dict[str, Any]]:
    """Recompute descriptive ratios against retained compressed timing with its time gap explicit."""
    reports = [json.loads((root / name).read_bytes()) for name in
               ("prior-compressed-high-concurrency.json", "candidate-high-concurrency.json")]
    if any(report["status"] != "complete" or len(report["trials"]) != 72 for report in reports):
        raise ValueError("Require every prior and current compressed trial")
    rows = []
    for case, _, _ in measurement.CASES:
        for window in measurement.WINDOWS:
            values = [[trial["measurement"]["aggregate_tokens_per_second"]
                       for trial in report["trials"]
                       if (trial["case"], trial["window_ms"]) == (case, window)]
                      for report in reports]
            if any(len(value) != 6 for value in values):
                raise ValueError("Incomplete prior-compressed comparison")
            medians = [statistics.median(value) for value in values]
            rows.append({"case": case, "window_ms": window,
                "parent_tps": medians[0], "output16_tps": medians[1],
                "tps_change_percent": 100 * (medians[1] / medians[0] - 1),
                "comparison_has_time_gap": True})
    return rows


def main() -> None:
    """Keep both original model gates; do not turn a narrower component pass into qualification."""
    root = Path(sys.argv[sys.argv.index("--root") + 1])
    plan = json.loads((root / "output-row16-plan.json").read_bytes())
    if plan["revision"] != "output16v1" or plan["output_exact_rows"] != [16]:
        raise ValueError("Missing frozen output-only follow-up plan")
    measurement.snapshot = snapshot
    sys.argv.extend(["--label", "candidate", "--data",
        "/audit/full-model-v1/quality-best-v1-data", "--cpu-audit",
        str(root / "source-audit.json"), "--profile"])
    measurement.main()
    result = review(root)
    result.update(stock_control_reused=True, stock_control_revision="c816v1",
        comparison_has_time_gap=True, fresh_paired_stock_load=False,
        output_row16_plan=plan, prior_compressed_revision="c816retryv2",
        prior_compressed_comparisons=parent_comparison(root),
        throughput_issue_declared_fixed=False)
    write(root / "quality-model-review.json", result)


if __name__ == "__main__":
    main()
