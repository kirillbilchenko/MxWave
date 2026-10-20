"""Retain unprofiled C1 API timing, native MTP counters, and separate actual-decode repeats."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path
from typing import Any

from benchmark_full_model import batch, request, write
from check_full_model import rpc
from decode_quality import collect
from measure_quality_model import source_check
from mtp_metrics import delta
from quality_suite import BASE


def snapshot(depth: int, verify_storage: bool = False) -> list[dict[str, Any]]:
    """Require the declared loaded drafter and preserve actual graph eligibility on both ranks."""
    rows = rpc(BASE, "mxwave_mtp_snapshot", [verify_storage])
    if (len(rows) != 2 or {r["rank"] for r in rows} != {0, 1}
            or any(r["depth"] != depth or not r["stable_dispatch"]["installed"] for r in rows)):
        raise RuntimeError("Actual loaded MTP scope differs from the frozen arm")
    return rows


def metrics(root: Path, label: str, phase: str) -> str:
    """Retain the original exposition as evidence before parsing speculative counters."""
    # Let the server's asynchronous statistics catch up outside the timed requests.
    time.sleep(6)
    with urllib.request.urlopen(BASE + "/metrics", timeout=15) as response:
        raw = response.read()
    path = root / f"{label}-metrics-{phase}.txt"
    if path.exists():
        raise FileExistsError("Preserve the first native metric snapshot")
    path.write_bytes(raw)
    return raw.decode()


def main() -> None:
    """Time the same compressed target twice, then collect distributions without timing them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--label", choices=("control", "candidate"), required=True)
    parser.add_argument("--vocabulary", type=int, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cpu-audit", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    plan_path = args.root / "mtp-pilot-plan.json"
    plan = json.loads(plan_path.read_bytes())
    if (plan["version"] != 1 or plan["concurrencies"] != [1]
            or plan["prompt_lengths"] != [256, 2048] or plan["generated_tokens"] != 64
            or plan["trials"] != 5 or plan["warmups"] != 2 or plan["warmup_tokens"] != 16
            or plan["depth_by_arm"] != {"control": 0, "candidate": 2}
            or plan["capture_sizes"] != [1, 4] or plan["quality_contexts"] != 16
            or plan["quality_generated_tokens"] != 4 or plan["mean_decode_kl_floor"] != 1e-5
            or plan["abs_mean_decode_nll_floor"] != 1e-4 or not args.profile):
        raise ValueError("Frozen MTP pilot protocol changed")
    raw = (args.root / "protocol.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != plan["source_protocol_sha256"]:
        raise ValueError("Original timing/quality prompts changed")
    protocol = json.loads(raw)
    cases = sorted((c for c in protocol["timing"] if c["batch"] == 1),
                   key=lambda c: c["prompt_tokens"])
    if [c["prompt_tokens"] for c in cases] != [256, 2048] or len(protocol["quality"]) != 16:
        raise ValueError("Require the two original C1 cases and sixteen quality contexts")
    depth = plan["depth_by_arm"][args.label]
    path = args.root / (args.label + "-mtp.json")
    if path.exists():
        raise FileExistsError("Preserve the first MTP model pilot")
    report: dict[str, Any] = {"status": "running", "label": args.label, "depth": depth,
        "started_epoch": time.time(), "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "quality_qualified": False, "previous_failed_gates_preserved": True,
        "warmups": [], "trials": [], "quality_repeats": []}

    def save(phase: str) -> None:
        report.update(phase=phase, updated_epoch=time.time())
        write(path, report)

    save("source_before")
    try:
        audit = json.loads(args.cpu_audit.read_bytes())
        report["source_before"] = source_check("candidate", audit)
        report["base_loaded_checks"] = rpc(BASE, "mxwave_checkpoint_check")
        report["draft_loaded_checks"] = rpc(BASE, "mxwave_mtp_source_check")
        if any(r["status"] != "passed" for r in
               report["base_loaded_checks"] + report["draft_loaded_checks"]):
            raise RuntimeError("Loaded base/draft source check failed")
        report["mhc_warmup"] = rpc(BASE, "mxwave_mhc_warmup", [2048])
        report["snapshot_before"] = snapshot(depth, True)
        for case in cases:
            for index in range(2):
                report["warmups"].append({"prompt_tokens": case["prompt_tokens"], "index": index,
                                         "measurement": batch(BASE, case["prompts"], 16)})
                save("warmups")
        before = metrics(args.root, args.label, "before-timing")
        for trial in range(5):
            for case in (cases[::-1] if trial % 2 else cases):
                first = snapshot(depth)
                measured = batch(BASE, case["prompts"], 64)
                report["trials"].append({"trial": trial, "prompt_tokens": case["prompt_tokens"],
                    "before": first, "measurement": measured, "after": snapshot(depth)})
                save("unprofiled_timing")
        after = metrics(args.root, args.label, "after-timing")
        report["timed_speculative_counter_delta"] = delta(before, after)
        if depth and not any((name == "vllm:spec_decode_num_drafts"
                             or name.startswith("vllm:spec_decode_num_drafts{")) and value > 0
                             for name, value in report["timed_speculative_counter_delta"].items()):
            raise RuntimeError("Native MTP did not report drafting during the timed requests")
        prompts = [protocol.get("quality_prefix_token_ids", []) + ids for ids in protocol["quality"]]
        for suffix in ("a", "b"):
            save("untimed_decode_quality_" + suffix)
            label = args.label + "-" + suffix
            collect(BASE, prompts, args.root, label, args.vocabulary, generated_tokens=4)
            report["quality_repeats"].append(label + "-decode-quality.json")
        save("profile_after_timing_and_quality")
        profile: dict[str, Any] = {"status": "running", "after_unprofiled_timing": True}
        report["profile"] = profile
        try:
            request(BASE, "/start_profile", {})
            try:
                profile["response"] = request(BASE, "/v1/completions", {
                    "model": "mxwave-glm53-pilot", "prompt": cases[0]["prompts"][0],
                    "max_tokens": 16, "min_tokens": 16, "temperature": 0, "seed": 42,
                    "ignore_eos": True, "return_token_ids": True})
            finally:
                request(BASE, "/stop_profile", {}, timeout=240)
            profile["status"] = "complete"
        except Exception as exc:  # noqa: BLE001 -- preserve unprofiled measurements on trace failure.
            profile.update(status="failed", error=repr(exc))
        report["source_after"] = source_check("candidate", audit)
        report["snapshot_after"] = snapshot(depth, True)
        report.update(status="complete", finished_epoch=time.time())
        save("complete")
    except BaseException as exc:
        report.update(status="failed", error=repr(exc), finished_epoch=time.time())
        save("failed")
        raise


if __name__ == "__main__":
    main()
