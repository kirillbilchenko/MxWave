"""Measure full-model quality, resources and C1/C8/C16 serving after startup preparation."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from admission_quality import collect_concurrent
from benchmark_full_model import batch, request, write
from c8_forward_trace import forward_signature
from check_full_model import rpc
from controlled_quality_cohort import collect_controlled_count
from decode_quality import compare
from measure_c8_trace import passed
from measure_high_concurrency import admission, configure, snapshot
from measure_quality_model import source_check
from quality_suite import BASE


def model_signature(group: dict[str, Any], rank: int = 0) -> list[dict[str, Any]]:
    """Compare real forwards, excluding empty scheduler iterations with no model work."""
    worker = next(w for w in group["worker_trace"] if w["rank"] == rank)
    return [forward_signature(e) for e in worker["events"] if e["total_scheduled_tokens"] > 0]


def main() -> None:
    """Keep native mHC and verify actual grouping instead of requiring serving invariance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--vocabulary", required=True, type=int)
    parser.add_argument("--label", choices=("control", "candidate"), required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--cpu-audit", required=True, type=Path)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    assert args.profile
    root, arm = args.root, args.label
    plan_path = root / "warmed-model-plan.json"
    plan = json.loads(plan_path.read_bytes())
    raw = (root / "protocol.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == plan["source_protocol_sha256"]
    protocol = json.loads(raw)
    contexts = [protocol.get("quality_prefix_token_ids", []) + x for x in protocol["quality"]]
    assert len(contexts) == 16 and all(len(x) == 258 for x in contexts)
    prompts = {(x["batch"], x["prompt_tokens"]): x["prompts"] for x in protocol["timing"]}
    path = root / f"{arm}-warmed-model.json"
    assert not path.exists()
    report: dict[str, Any] = {
        "status": "running",
        "arm": arm,
        "started_epoch": time.time(),
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "quality": [],
        "self_comparisons": [],
        "warmups": [],
        "timing": [],
        "profiles": [],
        "quality_qualified": False,
        "prior_failures_preserved": True,
        "native_mhc": True,
        "general_serving_batch_invariance_required": False,
        "controlled_admission_used_for_speed": False,
    }

    def save(phase: str) -> None:
        report.update(phase=phase, updated_epoch=time.time())
        write(path, report)

    def quality_group(label: str, count: int, profiled: bool) -> dict[str, Any]:
        configure(3)
        group: dict[str, Any] = {
            "label": label,
            "concurrency": count,
            "profiled": profiled,
            "controlled": count in (8, 16),
            "started_epoch": time.time(),
        }
        report["quality"].append(group)
        save(label)
        group["armed"] = rpc(BASE, "mxwave_c8_trace", [True, label])
        started = False
        try:
            if profiled:
                group["profile_start"] = request(BASE, "/start_profile", {})
                started = True
            if count in (8, 16):
                data, proofs = collect_controlled_count(
                    contexts, root, label, args.vocabulary, count
                )
                group["controlled_admission_proofs"] = proofs
            else:
                data = collect_concurrent(
                    BASE, contexts, root, label, args.vocabulary, concurrency=count
                )
            group["status"] = "complete"
        finally:
            if started:
                group["profile_stop"] = request(BASE, "/stop_profile", {}, timeout=240)
            group["worker_trace"] = rpc(BASE, "mxwave_c8_trace", [False, label])
            group["admission"] = admission("mxwave_admission_report")
            group["finished_epoch"] = time.time()
            write(root / f"{label}-forward-traces.json", group["worker_trace"])
            save(label + "-retained")
        assert model_signature(group, 0) == model_signature(group, 1)
        return data

    try:
        save("source_before")
        audit = json.loads(args.cpu_audit.read_bytes())
        report["source_before"] = source_check(arm, audit)
        report["loaded_checks"] = rpc(BASE, "mxwave_checkpoint_check")
        assert len(report["loaded_checks"]) == 2
        assert all(r["status"] == "passed" for r in report["loaded_checks"])
        report["native_mhc_policy"] = rpc(BASE, "mxwave_native_mhc_policy")
        assert len(report["native_mhc_policy"]) == 2
        assert all(r["native_mhc_unchanged"] for r in report["native_mhc_policy"])
        report["sparse_policy"] = rpc(BASE, "mxwave_stable_sparse_policy")
        assert len(report["sparse_policy"]) == 2
        assert all(
            r["patched_aliases"] and r["actual_decode_context_parallel_size"] == 1
            for r in report["sparse_policy"]
        )
        report["mhc_warmup"] = rpc(BASE, "mxwave_mhc_warmup", [2048])
        report["graphs"] = rpc(BASE, "mxwave_set_eager", [False])
        report["dispatch_startup"] = rpc(BASE, "mxwave_prepare_dispatch_startup")
        save("dispatch_startup_complete")
        report["snapshot_before"] = snapshot()
        report["telemetry_before"] = rpc(BASE, "mxwave_gpu_telemetry")
        for count in (1, 8, 16):
            for repeat in range(3):
                label = f"{arm}-c{count}-r{repeat}"
                data = quality_group(label, count, False)
                if repeat:
                    first = f"{arm}-c{count}-r0"
                    prior = json.loads((root / f"{first}-decode-quality.json").read_bytes())
                    value = compare(prior, data, root)
                    a = next(g for g in report["quality"] if g["label"] == first)
                    same = model_signature(a) == model_signature(report["quality"][-1])
                    report["self_comparisons"].append(
                        {
                            "reference": first,
                            "candidate": label,
                            **value,
                            "actual_model_grouping_equal": same,
                            "passed": same and passed(value),
                        }
                    )
                save(label + "-reviewed")
        report["startup_cache_after_quality"] = rpc(BASE, "mxwave_dispatch_startup_snapshot")
        cases = [(f"c{count}-{name}", count, length)
                 for count in (1, 8, 16) for name, length in (("short", 256), ("long", 2048))]
        assert plan["timing_cases"] == [list(c) for c in cases]
        for name, count, length in cases:
            for repeat in range(2):
                configure(3)
                report["warmups"].append(
                    {
                        "case": name,
                        "repeat": repeat,
                        "measurement": batch(BASE, prompts[count, length], 16),
                    }
                )
                save("untimed_warmups")
        for repeat in range(6):
            order = cases if repeat % 2 == 0 else list(reversed(cases))
            for name, count, length in order:
                configure(3)
                before = rpc(BASE, "mxwave_gpu_telemetry")
                measured = batch(BASE, prompts[count, length], 64)
                after = rpc(BASE, "mxwave_gpu_telemetry")
                report["timing"].append(
                    {
                        "case": name,
                        "concurrency": count,
                        "prompt_tokens": length,
                        "repeat": repeat,
                        "measurement": measured,
                        "telemetry_before": before,
                        "telemetry_after": after,
                        "dispatch_cache_after": rpc(BASE, "mxwave_dispatch_startup_snapshot"),
                        "admission": admission("mxwave_admission_report"),
                    }
                )
                save("ordinary_API_timing")
        report["snapshot_before_profiles"] = snapshot()
        quality_group(f"{arm}-c16-profile", 16, True)
        for count in (8, 16):
            configure(3)
            label = f"{arm}-c{count}-mixed-profile"
            mixed = [
                ids[:n]
                for ids, n in zip(
                    prompts[count, 2048], plan["profile_mixed_prompt_lengths"][:count], strict=True
                )
            ]
            item: dict[str, Any] = {
                "label": label,
                "concurrency": count,
                "prompt_lengths": list(map(len, mixed)),
                "after_all_unprofiled_timing": True,
            }
            report["profiles"].append(item)
            item["armed"] = rpc(BASE, "mxwave_c8_trace", [True, label])
            started = False
            try:
                item["profile_start"] = request(BASE, "/start_profile", {})
                started = True
                item["measurement"] = batch(BASE, mixed, 16)
                item["status"] = "complete"
            finally:
                if started:
                    item["profile_stop"] = request(BASE, "/stop_profile", {}, timeout=240)
                item["worker_trace"] = rpc(BASE, "mxwave_c8_trace", [False, label])
                save("ordinary_mixed_profile_retained")
        report["source_after"] = source_check(arm, audit)
        report["snapshot_after"] = snapshot()
        report["telemetry_after"] = rpc(BASE, "mxwave_gpu_telemetry")
        report["startup_cache_after_profiles"] = rpc(BASE, "mxwave_dispatch_startup_snapshot")
        report["native_mhc_policy_after"] = rpc(BASE, "mxwave_native_mhc_policy")
        report.update(status="complete", finished_epoch=time.time())
        save("complete")
    except BaseException as exc:
        report.update(status="failed", error=repr(exc), finished_epoch=time.time())
        save("failed")
        raise


if __name__ == "__main__":
    main()
