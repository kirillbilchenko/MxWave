"""Compare idle input windows with all requests retained, then collect quality and profiles."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

from admission_quality import collect_concurrent
from benchmark_full_model import completion, request, write
from check_full_model import rpc
from measure_paired_speed import snapshot
from measure_quality_model import source_check
from quality_suite import BASE

ORDERS = ((0, 3, 10), (3, 10, 0), (10, 0, 3), (10, 3, 0), (3, 0, 10), (0, 10, 3))
CASES = (("c4-short", 4, 256, 0, 12), ("c1-short", 1, 256, 0, 6),
         ("c8-short", 8, 256, 0, 6), ("c4-long", 4, 2048, 0, 6),
         ("c4-stagger", 4, 256, 2, 6))


def engine_rpc(method: str, args: list[Any] | None = None) -> dict[str, Any]:
    """Address one engine's queue control rather than the two-worker compute RPC contract."""
    value = request(BASE, "/collective_rpc", {"method": method, "args": args or [],
                    "kwargs": {}, "timeout": 30}, timeout=45)
    rows = value.get("results", [])
    if len(rows) != 1 or not rows[0]["only_idle_fresh_text_prefills"]:
        raise ValueError("Missing single-engine admission acknowledgement")
    return rows[0]


def configure(window_ms: int) -> dict[str, Any]:
    """Wait for actual idle state outside timing, then select one frozen queue window."""
    deadline = time.monotonic() + 10
    while not engine_rpc("mxwave_admission_report")["engine_idle"]:
        if time.monotonic() >= deadline:
            raise TimeoutError("The preceding closed request group did not become idle")
        time.sleep(.05)
    value = engine_rpc("mxwave_admission_configure", [window_ms])
    if value["window_ms"] != window_ms or value["events"]:
        raise ValueError("Admission reset/window selection failed")
    return value


def timed_batch(prompts: list[list[int]], tokens: int, stagger_ms: int) -> dict[str, Any]:
    """Include intended arrival spacing in group wall time and retain actual launch offsets."""
    started = time.perf_counter()

    def one(index: int, ids: list[int]) -> dict[str, Any]:
        remaining = started + index * stagger_ms / 1000 - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
        offset = time.perf_counter() - started
        return {**completion(BASE, ids, tokens), "request_index": index,
                "request_start_offset_seconds": offset,
                "intended_request_start_offset_ms": index * stagger_ms}

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as executor:
        jobs = [executor.submit(one, i, ids) for i, ids in enumerate(prompts)]
        rows = [job.result() for job in jobs]
    seconds = time.perf_counter() - started
    return {"wall_seconds": seconds, "aggregate_tokens_per_second": len(rows) * tokens / seconds,
            "median_ttft_seconds": statistics.median(r["ttft_seconds"] for r in rows),
            "stagger_between_requests_ms": stagger_ms, "requests": rows}


def main() -> None:
    """Run balanced interleaved timing on one target; keep strict numerical failures visible."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--vocabulary", type=int, required=True)
    args = parser.parse_args()
    root = args.root
    plan_path = root / "prefill-admission-plan.json"
    plan = json.loads(plan_path.read_bytes())
    raw = (root / "protocol.json").read_bytes()
    if (plan["windows_ms"] != [0, 3, 10] or plan["generated_tokens"] != 64
            or plan["warmups"] != 2 or plan["warmup_tokens"] != 16
            or plan["cases"] != [list(c) for c in CASES]
            or plan["window_orders"] != [list(o) for o in ORDERS]
            or plan["source_protocol_sha256"] != hashlib.sha256(raw).hexdigest()
            or plan["mean_decode_kl_floor"] != 1e-5
            or plan["abs_mean_decode_nll_floor"] != 1e-4
            or plan["quality_windows_ms"] != [0, 3, 10]
            or plan["quality_repeats_per_window"] != 2
            or plan["speculative_decoding"] or plan["capture_sizes"] != [1, 4]):
        raise ValueError("Frozen admission experiment protocol changed")
    protocol = json.loads(raw)
    prompts = {(c["batch"], c["prompt_tokens"]): c["prompts"] for c in protocol["timing"]}
    path = root / "prefill-admission-measurement.json"
    if path.exists():
        raise FileExistsError("Preserve the first admission comparison")
    report: dict[str, Any] = {"status": "running", "started_epoch": time.time(),
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "quality_qualified": False, "previous_failed_gates_preserved": True,
        "warmups": [], "trials": [], "quality": [], "profiles": []}

    def save(phase: str) -> None:
        report.update(phase=phase, updated_epoch=time.time())
        write(path, report)

    save("source_before")
    try:
        audit = json.loads((root / "source-audit.json").read_bytes())
        report["source_before"] = source_check("candidate", audit)
        report["loaded_checks"] = rpc(BASE, "mxwave_checkpoint_check")
        if any(r["status"] != "passed" for r in report["loaded_checks"]):
            raise RuntimeError("Loaded source-byte check failed")
        report["mhc_warmup"] = rpc(BASE, "mxwave_mhc_warmup", [2048])
        report["graph_selection"] = rpc(BASE, "mxwave_set_eager", [False])
        report["snapshot_before"] = snapshot()
        for name, count, length, stagger, _ in CASES:
            for window in (0, 3, 10):
                for index in range(2):
                    configure(window)
                    measured = timed_batch(prompts[(count, length)], 16, stagger)
                    report["warmups"].append({"case": name, "window_ms": window, "index": index,
                        "measurement": measured, "admission": engine_rpc("mxwave_admission_report")})
                    save("warmups")
        for trial in range(12):
            order = ORDERS[trial % len(ORDERS)]
            cases = CASES if trial % 2 == 0 else CASES[::-1]
            for name, count, length, stagger, repeats in cases:
                if trial >= repeats:
                    continue
                for window in order:
                    configure(window)
                    before = snapshot()
                    measured = timed_batch(prompts[(count, length)], 64, stagger)
                    report["trials"].append({"case": name, "trial": trial, "window_ms": window,
                        "concurrency": count, "prompt_tokens": length, "before": before,
                        "measurement": measured, "after": snapshot(),
                        "admission": engine_rpc("mxwave_admission_report")})
                    save("unprofiled_timing")
        contexts = [protocol.get("quality_prefix_token_ids", []) + ids for ids in protocol["quality"]]
        for window in (0, 3, 10):
            for suffix in ("a", "b"):
                configure(window)
                label = f"window{window}-{suffix}"
                save("untimed_concurrent_decode_quality_" + label)
                before = snapshot()
                collect_concurrent(BASE, contexts, root, label, args.vocabulary)
                report["quality"].append({"label": label, "window_ms": window,
                    "before": before, "after": snapshot(),
                    "admission": engine_rpc("mxwave_admission_report")})
                save("quality_collected")
        for window in (0, 3, 10):
            configure(window)
            profile: dict[str, Any] = {"status": "running", "window_ms": window,
                "after_all_unprofiled_timing": True, "started_epoch": time.time()}
            report["profiles"].append(profile)
            save("separate_profile")
            profile["weight_observation_armed"] = rpc(BASE, "mxwave_prefill_weight_audit", [True])
            started = False
            try:
                request(BASE, "/start_profile", {})
                started = True
                profile["measurement"] = timed_batch(prompts[(4, 256)], 4, 0)
                profile["admission"] = engine_rpc("mxwave_admission_report")
                profile["status"] = "complete"
            except Exception as exc:  # noqa: BLE001 -- preserve all completed timing/quality.
                profile.update(status="failed", error=repr(exc))
            finally:
                if started:
                    request(BASE, "/stop_profile", {}, timeout=240)
                profile["weight_observation"] = rpc(BASE, "mxwave_prefill_weight_audit", [False])
                profile["finished_epoch"] = time.time()
                save("profile_finished")
        configure(0)
        report["source_after"] = source_check("candidate", audit)
        report["snapshot_after"] = snapshot()
        report.update(status="complete", finished_epoch=time.time())
        save("complete")
        from analyze_prefill_admission_model import analyze

        analyze(root)
    except BaseException as exc:
        report.update(status="failed", error=repr(exc), finished_epoch=time.time())
        save("failed")
        raise


if __name__ == "__main__":
    main()
