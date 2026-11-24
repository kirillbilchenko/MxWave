"""Collect fixed-length serving speed and latency, then a separate controlled C4 profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from benchmark_full_model import batch, request, write
from check_full_model import rpc
from measure_cohort_model_dispatch import BASE, cohort_request
from measure_quality_model import source_check


def snapshot() -> list[dict[str, Any]]:
    """Require both workers and the original graph-one/four, eager-eight configuration."""
    rows = rpc(BASE, "mxwave_speed_snapshot")
    if (len(rows) != 2 or {r["rank"] for r in rows} != {0, 1}
            or any(not r["graphs_enabled"] or not r["graph_modes"]["1"].endswith("FULL")
                   or not r["graph_modes"]["4"].endswith("FULL")
                   or not r["graph_modes"]["8"].endswith("NONE") for r in rows)):
        raise RuntimeError("Actual worker graph configuration differs from the frozen protocol")
    return rows


def main() -> None:
    """Preserve every timed request; existing failed quality gates remain unchanged."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--label", choices=("control", "candidate"), required=True)
    parser.add_argument("--vocabulary", type=int, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cpu-audit", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    plan_path = args.root / "paired-speed-plan.json"
    plan = json.loads(plan_path.read_bytes())
    if (plan["version"] != 1 or plan["concurrencies"] != [1, 4, 8]
            or plan["prompt_lengths"] != [256, 2048] or plan["generated_tokens"] != 64
            or plan["trials"] != 5 or plan["warmups"] != 2 or plan["warmup_tokens"] != 16
            or plan["capture_sizes"] != [1, 4] or plan["qualification_attempted"]
            or not args.profile):
        raise ValueError("Frozen paired speed protocol changed")
    protocol = json.loads((args.root / "protocol.json").read_bytes())
    if hashlib.sha256((args.root / "protocol.json").read_bytes()).hexdigest() != plan[
            "source_protocol_sha256"]:
        raise ValueError("Original timing prompts changed")
    cases = sorted(protocol["timing"], key=lambda r: (r["batch"], r["prompt_tokens"]))
    if ([(c["batch"], c["prompt_tokens"]) for c in cases]
            != [(b, n) for b in (1, 4, 8) for n in (256, 2048)]):
        raise ValueError("Require exactly the six original timing cases")
    path = args.root / (args.label + "-speed.json")
    if path.exists():
        raise FileExistsError("Preserve the first serving measurement")
    report: dict[str, Any] = {"status": "running", "label": args.label,
        "started_epoch": time.time(), "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "qualification_attempted": False, "quality_qualified": False,
        "previous_failed_gates_preserved": True, "warmups": [], "trials": []}

    def save(phase: str) -> None:
        report.update(phase=phase, updated_epoch=time.time())
        write(path, report)

    save("source_before")
    try:
        audit = json.loads(args.cpu_audit.read_bytes())
        report["source_before"] = source_check(args.label, audit)
        report["loaded_weight_checks"] = rpc(BASE, "mxwave_checkpoint_check")
        if any(r["status"] != "passed" for r in report["loaded_weight_checks"]):
            raise RuntimeError("Loaded source tensor check failed")
        report["mhc_warmup"] = rpc(BASE, "mxwave_mhc_warmup", [2048])
        report["graph_selection"] = rpc(BASE, "mxwave_set_eager", [False])
        report["snapshot_before"] = snapshot()
        for case in cases:
            for index in range(2):
                report["warmups"].append({"batch": case["batch"],
                    "prompt_tokens": case["prompt_tokens"], "index": index,
                    "measurement": batch(BASE, case["prompts"], 16)})
                save("warmups")
        for trial in range(5):
            for case in (cases[::-1] if trial % 2 else cases):
                before = snapshot()
                measured = batch(BASE, case["prompts"], 64)
                after = snapshot()
                report["trials"].append({"trial": trial, "batch": case["batch"],
                    "prompt_tokens": case["prompt_tokens"], "before": before,
                    "measurement": measured, "after": after})
                save("unprofiled_timing")
        save("profile_after_timing")
        case = next(c for c in cases if (c["batch"], c["prompt_tokens"]) == (4, 256))
        profile: dict[str, Any] = {"status": "running", "after_all_unprofiled_timing": True}
        report["profile"] = profile
        try:
            request(BASE, "/start_profile", {})
            try:
                response, receipt = cohort_request(BASE, {"model": "mxwave-glm53-pilot",
                    "prompt": case["prompts"], "max_tokens": 4, "min_tokens": 4,
                    "temperature": 0, "seed": 42, "ignore_eos": True,
                    "return_token_ids": True}, True, 1.0)
            finally:
                request(BASE, "/stop_profile", {}, timeout=240)
            profile.update(status="complete" if receipt["proof"]["passed"] else "split_cohort",
                           response=response, receipt=receipt)
        except Exception as exc:  # noqa: BLE001 -- retain the completed timing before profiling.
            profile.update(status="failed", error=repr(exc))
        save("source_after")
        report["source_after"] = source_check(args.label, audit)
        report["snapshot_after"] = snapshot()
        report.update(status="complete", finished_epoch=time.time())
        save("complete")
    except BaseException as exc:
        report.update(status="failed", error=repr(exc), finished_epoch=time.time())
        save("failed")
        raise


if __name__ == "__main__":
    main()
