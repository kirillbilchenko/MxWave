"""Collect all predeclared practical repeats in one loaded engine without choosing a best run."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from benchmark_full_model import write
from measure_quality_model import source_check
from quality_suite import tasks


def main() -> None:
    """Repeat the unchanged 34 tasks five times, retaining errors and source checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--label", choices=("control", "candidate"), required=True)
    parser.add_argument("--vocabulary", type=int, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cpu-audit", type=Path, required=True)
    parser.add_argument("--task-repeats", type=int, choices=(5,), required=True)
    args = parser.parse_args()
    plan = json.loads((args.root / "task-repeat-plan.json").read_bytes())
    raw = (args.root / "quality-protocol.json").read_bytes()
    suite = json.loads(raw)
    if (
        plan["rounds_per_arm"] != args.task_repeats
        or plan["quality_protocol_sha256"] != hashlib.sha256(raw).hexdigest()
        or plan["tasks_per_round"] != len(suite["tasks"])
        or plan["qualification_attempted"]
    ):
        raise ValueError("Practical diagnostic plan or frozen tasks changed")
    path = args.root / (args.label + "-quality-measurement.json")
    if path.exists():
        raise FileExistsError("Preserve prior practical measurements")
    report: dict[str, Any] = {
        "status": "running", "label": args.label,
        "experiment_type": "practical_repeat_diagnostic", "started_epoch": time.time(),
        "predeclared_rounds": args.task_repeats, "rounds": [],
        "new_quality_qualification_attempted": False,
        "throughput_benchmark_run": False,
        "quality_protocol_sha256": hashlib.sha256(raw).hexdigest(),
        "plan_sha256": hashlib.sha256((args.root / "task-repeat-plan.json").read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }

    def save(phase: str) -> None:
        report.update(phase=phase, updated_epoch=time.time())
        write(path, report)

    try:
        save("source_before")
        audit = json.loads(args.cpu_audit.read_bytes())
        report["source_before"] = source_check(args.label, audit)
        for index in range(args.task_repeats):
            label = f"{args.label}-r{index}"
            folder = args.root / label
            folder.mkdir(exist_ok=False)
            row: dict[str, Any] = {"round": index, "label": label, "started_epoch": time.time()}
            report["rounds"].append(row)
            save("round_" + str(index))
            collected = tasks(folder, suite)
            row.update(status=collected["status"], cases=len(collected["cases"]),
                       finished_epoch=time.time())
            save("finished_round_" + str(index))
        save("source_after")
        report["source_after"] = source_check(args.label, audit)
        report.update(status="collected_pending_code_sandbox", finished_epoch=time.time())
        save("collected")
    except BaseException as exc:
        report.update(status="failed", error=repr(exc), finished_epoch=time.time())
        save("failed")
        raise


if __name__ == "__main__":
    main()
