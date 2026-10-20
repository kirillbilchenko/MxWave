"""Isolate each N8 mechanism in a fresh child process within one bounded GPU session."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from build_bf16_n8 import VARIANTS


def main() -> None:
    """Keep per-candidate evidence and reverse candidate order in the confirmation session."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--confirmation", action="store_true")
    parser.add_argument("--candidates", nargs="+", choices=VARIANTS, required=True)
    args = parser.parse_args()
    if (args.report.exists() or "control" in args.candidates
            or len(set(args.candidates)) != len(args.candidates)):
        raise ValueError("Require a fresh report and distinct non-control candidates")
    order = list(reversed(args.candidates)) if args.confirmation else args.candidates
    started = time.monotonic()
    result = {"status": "running", "confirmation": args.confirmation, "order": order,
              "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "candidate_reports": {}}
    for candidate in order:
        output = args.report.with_name(args.report.stem + "-" + candidate + ".json")
        command = [sys.executable, "-u", str(Path(__file__).with_name("benchmark_bf16_n8.py")),
                   "--build", str(args.build), "--report", str(output), "--candidate", candidate]
        if args.confirmation:
            command.append("--confirmation")
        subprocess.run(command, check=True, timeout=120)
        report = json.loads(output.read_bytes())
        if report["status"] != "passed" or report["candidate"] != candidate:
            raise RuntimeError("Candidate failed its exactness or coverage gate")
        result["candidate_reports"][candidate] = report
        args.report.write_text(json.dumps(result, indent=2) + "\n")
    result.update(status="passed", seconds=time.monotonic() - started)
    args.report.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
