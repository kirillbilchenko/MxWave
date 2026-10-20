"""Locate the split-admission latency anomaly without changing model arithmetic."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from benchmark_full_model import request, write
from check_full_model import rpc
from latency_stream import ordinary_group, split_group
from measure_high_concurrency import admission, configure, snapshot
from measure_quality_model import source_check
from quality_suite import BASE


def main() -> None:
    """Record every warmup/group and profile the first observed split plus its repeat."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--label", required=True, choices=("control", "candidate"))
    parser.add_argument("--vocabulary", required=True, type=int)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--cpu-audit", required=True, type=Path)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if not args.profile:
        raise ValueError("The diagnostic requires CPU/GPU traces")
    root, arm = args.root, args.label
    plan_bytes = (root / "latency-probe-plan.json").read_bytes()
    plan = json.loads(plan_bytes)
    raw = (root / "protocol.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == plan["source_protocol_sha256"]
    protocol = json.loads(raw)
    prompts = {(v["batch"], v["prompt_tokens"]): v["prompts"] for v in protocol["timing"]}
    path = root / f"{arm}-latency-probe.json"
    assert not path.exists()
    report: dict[str, Any] = {
        "status": "running",
        "arm": arm,
        "started_epoch": time.time(),
        "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "groups": [],
        "profile_sessions": [],
        "quality_qualified": False,
        "diagnostic_not_serving_speed": True,
        "model_arithmetic_changed": False,
        "original_failures_and_gates_preserved": True,
    }

    def save(phase: str) -> None:
        report.update(phase=phase, updated_epoch=time.time())
        write(path, report)

    def group(spec: dict[str, Any]) -> None:
        configure(3)
        label = arm + "-" + spec["name"]
        item: dict[str, Any] = {**spec, "label": label, "started_epoch": time.time()}
        report["groups"].append(item)
        save(label)
        item["armed"] = rpc(BASE, "mxwave_latency_trace", [True, label])
        try:
            ids = prompts[spec["concurrency"], spec["prompt_tokens"]]
            if spec["admission"] == "split":
                item["measurement"] = split_group(
                    BASE,
                    ids,
                    spec["generated_tokens"],
                    request,
                    grace_seconds=plan["grace_seconds"],
                    late_delay_seconds=plan["late_delay_seconds"],
                )
            else:
                assert spec["admission"] == "ordinary"
                item["measurement"] = ordinary_group(BASE, ids, spec["generated_tokens"])
            item["status"] = "complete"
        except BaseException as exc:
            item.update(status="failed", error=repr(exc))
            raise
        finally:
            item["worker_trace"] = rpc(BASE, "mxwave_latency_trace", [False, label])
            item["admission_report"] = admission("mxwave_admission_report")
            item["finished_epoch"] = time.time()
            write(root / f"{label}-forward-traces.json", item["worker_trace"])
            save(label + "-retained")

    try:
        save("source_before")
        report["source_before"] = source_check(arm, json.loads(args.cpu_audit.read_bytes()))
        report["loaded_checks"] = rpc(BASE, "mxwave_checkpoint_check")
        assert len(report["loaded_checks"]) == 2
        assert all(v["status"] == "passed" for v in report["loaded_checks"])
        report["native_mhc_policy"] = rpc(BASE, "mxwave_native_mhc_policy")
        assert all(v["native_mhc_unchanged"] for v in report["native_mhc_policy"])
        report["sparse_policy"] = rpc(BASE, "mxwave_stable_sparse_policy")
        assert all(
            v["patched_aliases"] and v["actual_decode_context_parallel_size"] == 1
            for v in report["sparse_policy"]
        )
        report["mhc_warmup"] = rpc(BASE, "mxwave_mhc_warmup", [2048])
        report["graphs"] = rpc(BASE, "mxwave_set_eager", [False])
        report["snapshot_before"] = snapshot()
        for spec in plan["groups"]:
            if spec["stage"] == "warmup":
                group(spec)
        session: dict[str, Any] = {"status": "starting", "started_epoch": time.time()}
        report["profile_sessions"].append(session)
        started = False
        try:
            session["start"] = request(BASE, "/start_profile", {})
            started = True
            for spec in plan["groups"]:
                if spec["stage"] == "profile":
                    group(spec)
            session["status"] = "complete"
        finally:
            if started:
                session["stop"] = request(BASE, "/stop_profile", {}, timeout=240)
            session["finished_epoch"] = time.time()
            save("profile_retained")
        for spec in plan["groups"]:
            if spec["stage"] == "unprofiled_after_profile":
                group(spec)
        assert len(report["groups"]) == len(plan["groups"]) == 15
        report["source_after"] = source_check(arm, json.loads(args.cpu_audit.read_bytes()))
        report["native_mhc_policy_after"] = rpc(BASE, "mxwave_native_mhc_policy")
        report["snapshot_after"] = snapshot()
        report.update(status="complete", finished_epoch=time.time())
        save("complete")
    except BaseException as exc:
        report.update(status="failed", error=repr(exc), finished_epoch=time.time())
        save("failed")
        raise


if __name__ == "__main__":
    main()
