"""Retain matched C8/C16 API timings, numerical repeats, and separate both-rank profiles."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from admission_quality import collect_concurrent
from benchmark_full_model import batch, request, write
from check_full_model import rpc
from high_concurrency_protocol import CASES, WINDOWS, validate_plan, validate_snapshot
from measure_quality_model import source_check
from quality_suite import BASE


def snapshot() -> list[dict[str, Any]]:
    """Require actual C16 replay and fixed capacity on both ranks, outside timed requests."""
    rows = rpc(BASE, "mxwave_high_concurrency_snapshot")
    validate_snapshot(rows)
    return rows


def admission(method: str, args: list[Any] | None = None) -> dict[str, Any]:
    """Address the single engine queue without confusing its result with a TP worker RPC."""
    value = request(BASE, "/collective_rpc", {"method": method, "args": args or [],
                    "kwargs": {}, "timeout": 30}, timeout=45)
    rows = value.get("results", [])
    if len(rows) != 1 or not rows[0]["only_idle_fresh_text_prefills"]:
        raise ValueError("Missing native engine admission acknowledgement")
    return rows[0]


def configure(window: int) -> dict[str, Any]:
    """Select zero or three milliseconds only between closed idle groups."""
    if window not in WINDOWS:
        raise ValueError("Unexpected high-concurrency admission window")
    deadline = time.monotonic() + 15
    while not admission("mxwave_admission_report")["engine_idle"]:
        if time.monotonic() >= deadline:
            raise TimeoutError("Previous request group did not become idle")
        time.sleep(.05)
    value = admission("mxwave_admission_configure", [window])
    if value["window_ms"] != window or value["events"]:
        raise ValueError("Admission selection/reset failed")
    return value


def main() -> None:
    """Measure without MTP; retain failures and restore the original service through the controller."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--label", choices=("control", "candidate"), required=True)
    parser.add_argument("--vocabulary", type=int, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cpu-audit", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    root = args.root
    plan_path = root / "high-concurrency-plan.json"
    plan = json.loads(plan_path.read_bytes())
    validate_plan(plan)
    raw = (root / "protocol.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != plan["source_protocol_sha256"] or not args.profile:
        raise ValueError("Frozen prompts or required separate profiling changed")
    protocol = json.loads(raw)
    prompts = {(c["batch"], c["prompt_tokens"]): c["prompts"] for c in protocol["timing"]}
    for _, count, length in CASES:
        ids = prompts[count, length]
        if len(ids) != count or any(len(row) != length for row in ids):
            raise ValueError("Incomplete high-concurrency prompt grid")
    path = root / (args.label + "-high-concurrency.json")
    if path.exists():
        raise FileExistsError("Preserve the first high-concurrency measurement")
    report: dict[str, Any] = {"status": "running", "label": args.label,
        "started_epoch": time.time(), "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "quality_qualified": False, "previous_failed_gates_preserved": True,
        "warmups": [], "trials": [], "quality": [], "profiles": []}

    def save(phase: str) -> None:
        report.update(phase=phase, updated_epoch=time.time())
        write(path, report)

    save("source_before")
    try:
        audit = json.loads(args.cpu_audit.read_bytes())
        report["source_before"] = source_check(args.label, audit)
        report["loaded_checks"] = rpc(BASE, "mxwave_checkpoint_check")
        if len(report["loaded_checks"]) != 2 or any(
                row["status"] != "passed" for row in report["loaded_checks"]):
            raise RuntimeError("Loaded source-byte check failed")
        report["mhc_warmup"] = rpc(BASE, "mxwave_mhc_warmup", [2048])
        report["graph_selection"] = rpc(BASE, "mxwave_set_eager", [False])
        report["snapshot_before"] = snapshot()
        for name, count, length in CASES:
            for window in WINDOWS:
                for index in range(2):
                    configure(window)
                    report["warmups"].append({"case": name, "window_ms": window, "index": index,
                        "measurement": batch(BASE, prompts[count, length], 16),
                        "admission": admission("mxwave_admission_report")})
                    save("warmups")
        for trial in range(6):
            cases = CASES if trial % 2 == 0 else CASES[::-1]
            for name, count, length in cases:
                for window in (WINDOWS if trial % 2 == 0 else WINDOWS[::-1]):
                    configure(window)
                    before = snapshot()
                    measured = batch(BASE, prompts[count, length], 64)
                    report["trials"].append({"case": name, "trial": trial, "window_ms": window,
                        "concurrency": count, "prompt_tokens": length, "before": before,
                        "measurement": measured, "after": snapshot(),
                        "admission": admission("mxwave_admission_report")})
                    save("unprofiled_timing")
        contexts = [protocol.get("quality_prefix_token_ids", []) + ids for ids in protocol["quality"]]
        for count in (8, 16):
            for window in WINDOWS:
                for suffix in ("a", "b"):
                    configure(window)
                    label = f"{args.label}-c{count}-w{window}-{suffix}"
                    save("untimed_quality_" + label)
                    before = snapshot()
                    collect_concurrent(BASE, contexts, root, label, args.vocabulary,
                                       concurrency=count)
                    report["quality"].append({"label": label, "concurrency": count,
                        "window_ms": window, "before": before, "after": snapshot(),
                        "admission": admission("mxwave_admission_report")})
                    save("quality_collected")
        for count in (8, 16):
            for window in WINDOWS:
                configure(window)
                profile: dict[str, Any] = {"status": "running", "concurrency": count,
                    "window_ms": window, "after_all_unprofiled_timing": True,
                    "started_epoch": time.time(), "before": snapshot()}
                report["profiles"].append(profile)
                save("separate_profile")
                started = False
                try:
                    request(BASE, "/start_profile", {})
                    started = True
                    profile["measurement"] = batch(BASE, prompts[count, 256], 4)
                    profile["admission"] = admission("mxwave_admission_report")
                    profile["status"] = "complete"
                except Exception as exc:  # noqa: BLE001 -- retain completed API timing and quality.
                    profile.update(status="failed", error=repr(exc))
                finally:
                    if started:
                        request(BASE, "/stop_profile", {}, timeout=240)
                    profile.update(after=snapshot(), finished_epoch=time.time())
                    save("profile_finished")
        configure(0)
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
