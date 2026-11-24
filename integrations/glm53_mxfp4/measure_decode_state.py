"""Locate reference decode drift with fixed repeats and identical-state graph replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from benchmark_full_model import write
from check_full_model import rpc
from measure_cohort_model_dispatch import BASE, cohort_request
from measure_quality_model import source_check


def validate_plan(plan: dict[str, Any]) -> None:
    """Reject changed replay counts, samples, model arms or numerical-contract claims."""
    if (plan["version"] != 2 or plan["model_arm"] != "control" or plan["model_loads"] != 1
            or plan["iterations_per_probe"] != 5 or plan["reference_repeats"] != 0
            or plan["probe_repeats_per_target"] != 3 or plan["capture_sizes"] != [1, 4]
            or plan["targets"] != [{"group_start": 0, "graph_step": 2},
                                    {"group_start": 4, "graph_step": 0}]
            or plan["qualification_attempted"] or plan["stop_on_kl_failure"]
            or not plan["previous_failed_gates_preserved"]
            or plan["admission_grace_seconds"] != 1.0
            or plan["snapshot_backing"] != "ssd_mmap"
            or plan["reference_evidence"]["archive_sha256"] !=
                "fc9e387a3f99dd64fdd427e5d4f4b2b4251c03b54d9d2be0f0318a681ef35dbb"):
        raise ValueError("Frozen reference decode diagnostic changed")


def main() -> None:
    """Retain every reference repeat/probe, including failures, without rerunning accuracy tasks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--vocabulary", type=int, required=True)
    args = parser.parse_args()
    root = args.root
    plan_path = root / "decode-state-plan.json"
    plan = json.loads(plan_path.read_bytes())
    validate_plan(plan)
    protocol_path = root / "protocol.json"
    if hashlib.sha256(protocol_path.read_bytes()).hexdigest() != plan["source_protocol_sha256"]:
        raise ValueError("Original source contexts changed")
    proto = json.loads(protocol_path.read_bytes())
    prompts = [proto.get("quality_prefix_token_ids", []) + ids for ids in proto["quality"]]
    if len(prompts) != 16 or any(len(ids) != 258 for ids in prompts):
        raise ValueError("Require the original complete sixteen source contexts")
    path = root / "decode-state-model-review.json"
    if path.exists():
        raise FileExistsError("Keep the first diagnostic attempt")
    report: dict[str, Any] = {"status": "running", "started_epoch": time.time(),
        "model_loads": 1, "model_arm": "control", "qualification_attempted": False,
        "quality_qualified": False, "previous_failed_gates_preserved": True,
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "reference_repeats": [], "reference_comparisons": [], "probes": [],
        "reused_reference_evidence": plan["reference_evidence"]}

    def save(phase: str) -> None:
        report.update(phase=phase, updated_epoch=time.time())
        write(path, report)

    save("source_before")
    try:
        audit = json.loads((root / "source-audit.json").read_bytes())
        report["source_before"] = source_check("control", audit)
        report["loaded_weight_checks"] = rpc(BASE, "mxwave_checkpoint_check")
        if any(r["status"] != "passed" for r in report["loaded_weight_checks"]):
            raise RuntimeError("Loaded source tensor check failed")
        report["mhc_warmup"] = rpc(BASE, "mxwave_mhc_warmup", [2048])
        report["graphs"] = rpc(BASE, "mxwave_set_eager", [False])
        for target in plan["targets"]:
            for repeat in range(3):
                group, step = target["group_start"], target["graph_step"]
                label = f"g{group}s{step}r{repeat}"
                body: dict[str, Any] = {"model": "mxwave-glm53-pilot", "max_tokens": 4,
                    "min_tokens": 4, "temperature": 0, "seed": 42, "ignore_eos": True,
                    "return_token_ids": True}
                item: dict[str, Any] = {"label": label, "target": target, "repeat": repeat}
                report["probes"].append(item)
                save("preparing_probe_" + label)
                # Preserve the predecessor cohort for the original context-4 group.
                if group:
                    body["prompt"] = prompts[:4]
                    response, receipt = cohort_request(BASE, body, True, 1.0)
                    item["predecessor"] = {"response": response, "receipt": receipt}
                    if not receipt["proof"]["passed"]:
                        raise RuntimeError("Predecessor cohort split; preserve it without retry")
                body["prompt"] = prompts[group:group + 4]
                item["arm"] = rpc(BASE, "mxwave_decode_probe_arm", [label, step])
                if len(item["arm"]) != 2 or any(r["status"] != "armed" for r in item["arm"]):
                    item["disarm"] = rpc(BASE, "mxwave_decode_probe_finish")
                    raise RuntimeError("Snapshot reservation failed before probe admission")
                save("replaying_probe_" + label)
                try:
                    response, receipt = cohort_request(BASE, body, True, 1.0)
                    item.update(response=response, receipt=receipt)
                finally:
                    item["workers"] = rpc(BASE, "mxwave_decode_probe_finish")
                    save("collected_probe_" + label)
                if (not receipt["proof"]["passed"] or len(item["workers"]) != 2
                        or any(r["status"] != "complete" or len(r["replays"]) != 5
                               for r in item["workers"])):
                    raise RuntimeError("Incomplete/split diagnostic; never retry or qualify it")
        report["source_after"] = source_check("control", audit)
        report["same_state_outputs_exact"] = all(
            r["all_hidden_outputs_exact"] for p in report["probes"] for r in p["workers"])
        report["same_state_resulting_bytes_exact"] = all(
            r["all_resulting_storage_bytes_exact"] for p in report["probes"] for r in p["workers"])
        report.update(status="complete", finished_epoch=time.time(),
            limits=["Intrusive diagnostic; no new speed, accuracy or qualification claim.",
                    "Replay restores named KV and input storages plus bounded metadata tensors.",
                    "Native internal workspaces outside those storages are not reset.",
                    "Cache differences identify a lead, not necessarily the first causal operation.",
                    "Only the fifth fixed replay supplies the served output; no favorable selection.",
                    "An unreproduced failure does not amend the historical strict failure."])
        save("complete")
    except BaseException as exc:
        report.update(status="failed", error=repr(exc), finished_epoch=time.time())
        save("failed")
        raise


if __name__ == "__main__":
    main()
