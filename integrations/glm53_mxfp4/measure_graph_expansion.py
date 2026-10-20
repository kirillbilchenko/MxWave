"""Time two native graph-dispatch policies on one model, then retain C8 quality and profiles."""

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
from measure_quality_model import source_check
from quality_suite import BASE

CASES = ((1, 256), (2, 256), (4, 256), (8, 256), (2, 2048), (8, 2048))


def snapshot(policy: str) -> list[dict[str, Any]]:
    """Require exact C2/padded C2 and eager/graph C8 descriptors on both workers."""
    rows = rpc(BASE, "mxwave_graph_policy_snapshot")
    if len(rows) != 2 or {r["rank"] for r in rows} != {0, 1}:
        raise ValueError("Both graph-policy ranks are required")
    for row in rows:
        desc = row["descriptors"]
        if (row["policy"] != policy or row["physical_capture_sizes"] != [1, 2, 4, 8]
                or row["max_num_seqs"] != 8 or not row["graphs_enabled"]
                or not row["speculative_config_absent"]
                or desc["2"]["mode"] != "FULL"
                or desc["2"]["num_tokens"] != (4 if policy == "baseline" else 2)
                or desc["8"]["mode"] != ("NONE" if policy == "baseline" else "FULL")
                or desc["16"]["mode"] != "NONE"):
            raise RuntimeError("Actual graph descriptor policy differs from the frozen arm")
    return rows


def select(policy: str) -> list[dict[str, Any]]:
    """Switch only after a preceding closed request group; native replay/allocation stay intact."""
    rpc(BASE, "mxwave_graph_policy_select", [policy])
    return snapshot(policy)


def main() -> None:
    """Retain every timing trial and unchanged numerical floor, including failed repeats."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--vocabulary", type=int, required=True)
    args = parser.parse_args()
    root = args.root
    plan_path = root / "graph-expansion-plan.json"
    plan = json.loads(plan_path.read_bytes())
    raw = (root / "protocol.json").read_bytes()
    if (plan["cases"] != [list(c) for c in CASES] or plan["trials"] != 6
            or plan["policies"] != ["baseline", "expanded"]
            or plan["capture_sizes"] != [1, 2, 4, 8] or plan["max_num_seqs"] != 8
            or plan["generated_tokens"] != 64 or plan["warmups"] != 2
            or plan["warmup_tokens"] != 16 or plan["mean_decode_kl_floor"] != 1e-5
            or plan["abs_mean_decode_nll_floor"] != 1e-4
            or plan["source_protocol_sha256"] != hashlib.sha256(raw).hexdigest()
            or plan["speculative_decoding"] or plan["admission_wait_enabled"]):
        raise ValueError("Frozen graph-expansion protocol changed")
    protocol = json.loads(raw)
    prompts = {(c["batch"], c["prompt_tokens"]): c["prompts"] for c in protocol["timing"]}
    path = root / "graph-expansion-measurement.json"
    if path.exists():
        raise FileExistsError("Preserve the first graph-expansion comparison")
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
        for count, length in CASES:
            for policy in ("baseline", "expanded"):
                select(policy)
                for index in range(2):
                    report["warmups"].append({"batch": count, "prompt_tokens": length,
                        "policy": policy, "index": index,
                        "measurement": batch(BASE, prompts[count, length], 16)})
                    save("warmups")
        for trial in range(6):
            cases = CASES if trial % 2 == 0 else CASES[::-1]
            policies = ("baseline", "expanded") if trial % 2 == 0 else ("expanded", "baseline")
            for count, length in cases:
                for policy in policies:
                    before = select(policy)
                    measured = batch(BASE, prompts[count, length], 64)
                    report["trials"].append({"batch": count, "prompt_tokens": length,
                        "trial": trial, "policy": policy, "before": before,
                        "measurement": measured, "after": snapshot(policy)})
                    save("interleaved_unprofiled_timing")
        contexts = [protocol.get("quality_prefix_token_ids", []) + ids for ids in protocol["quality"]]
        for policy in ("baseline", "expanded"):
            for suffix in ("a", "b"):
                before = select(policy)
                label = policy + "-" + suffix
                save("untimed_c8_decode_quality_" + label)
                collect_concurrent(BASE, contexts, root, label, args.vocabulary, concurrency=8)
                report["quality"].append({"label": label, "before": before,
                                           "after": snapshot(policy)})
                save("quality_collected")
        for policy in ("baseline", "expanded"):
            for count in (2, 8):
                before = select(policy)
                profile: dict[str, Any] = {"status": "running", "policy": policy, "batch": count,
                    "before": before, "after_all_unprofiled_timing": True, "started_epoch": time.time()}
                report["profiles"].append(profile)
                save("separate_profile")
                try:
                    request(BASE, "/start_profile", {})
                    try:
                        profile["measurement"] = batch(BASE, prompts[count, 256], 4)
                    finally:
                        request(BASE, "/stop_profile", {}, timeout=240)
                    profile.update(status="complete", after=snapshot(policy), finished_epoch=time.time())
                except Exception as exc:  # noqa: BLE001 -- keep every measured trial and quality row.
                    profile.update(status="failed", error=repr(exc), finished_epoch=time.time())
                save("profile_finished")
        select("expanded")
        report["source_after"] = source_check("candidate", audit)
        report.update(status="complete", finished_epoch=time.time())
        save("complete")
        from analyze_graph_expansion import analyze

        analyze(root)
    except BaseException as exc:
        report.update(status="failed", error=repr(exc), finished_epoch=time.time())
        save("failed")
        raise


if __name__ == "__main__":
    main()
