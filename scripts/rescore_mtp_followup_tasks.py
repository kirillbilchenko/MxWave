"""Correct diagnostic builtin availability without changing or rerunning frozen generations."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

from evaluate_mtp_followup import _load, _task_compare
from evaluate_quantization_controls import _write_json
from run_mtp_followup import _source_digest


def main() -> None:
    """Bind corrected task scores to the unmodified raw reports and original comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--original-comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite corrected scores")
    protocol, digest = _load(args.protocol)
    original_bytes = args.original_comparison.read_bytes()
    original = json.loads(original_bytes)
    if not original["complete"] or original["protocol_sha256"] != digest:
        raise ValueError("Original comparison protocol mismatch")
    reports, hashes = {}, {}
    for mtp in (0, 2):
        name = f"h64-graphs-mtp{mtp}-diagnostics"
        raw = gzip.decompress((args.measurements / f"{name}.json.gz").read_bytes())
        sha = hashlib.sha256(raw).hexdigest()
        report = json.loads(raw)
        if (
            sha != original["report_sha256"][name]
            or not report["complete"]
            or report["protocol_sha256"] != digest
            or report["executed_source_sha256"] != protocol["executed_source_sha256"]
        ):
            raise ValueError("Raw generation identity mismatch")
        reports[mtp], hashes[name] = report, sha
    scoring_sha = hashlib.sha256(
        _source_digest().encode() + Path(__file__).read_bytes()
    ).hexdigest()
    result = {
        "format": "mxwave-mtp-followup-corrected-task-scores-v1",
        "complete": True,
        "protocol_sha256": digest,
        "collection_executed_source_sha256": protocol["executed_source_sha256"],
        "scoring_source_sha256": scoring_sha,
        "scoring_python_version": sys.version,
        "original_comparison_sha256": hashlib.sha256(original_bytes).hexdigest(),
        "report_sha256": hashes,
        "correction": "Allow the standard safe builtin isinstance. No prompt, answer, unit case, "
        "generation, timing or acceptance counter changed; no new inference.",
        "original_score_groups": original["tasks"]["groups"],
        "corrected_tasks": _task_compare(protocol, reports[0], reports[2]),
    }
    _write_json(args.output, result)
    print(json.dumps(result["corrected_tasks"]["groups"], indent=2))


if __name__ == "__main__":
    main()
