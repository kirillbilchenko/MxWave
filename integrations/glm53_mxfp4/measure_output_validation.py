"""Collect missing validation and profiles without repeating or overwriting retained API timings."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

from admission_quality import collect_concurrent
from analyze_high_concurrency import review
from benchmark_full_model import batch, request, write
from check_full_model import rpc
from high_concurrency_protocol import CASES, WINDOWS, validate_plan
from measure_high_concurrency import admission, configure
from measure_output_row16 import parent_comparison, snapshot
from measure_quality_model import source_check
from quality_suite import BASE


def combined_candidate(
    prior: dict[str, Any], validation: dict[str, Any], provenance: dict[str, Any]
) -> dict[str, Any]:
    """Create a labeled derived view; never rewrite either original phase report."""
    if (
        len(prior["trials"]) != 72
        or len(prior["warmups"]) != 24
        or validation["status"] != "complete"
        or validation["trials"]
        or len(validation["quality"]) != 8
        or len(validation["profiles"]) != 4
    ):
        raise ValueError(
            "Require complete original timing and fresh validation; no new timed trials"
        )
    if (
        prior["plan_sha256"] != validation["plan_sha256"]
        or prior["label"] != validation["label"]
        or "source_after" in prior
        or provenance["original_source_after_missing"] is not True
        or provenance["comparison_has_time_gap"] is not True
    ):
        raise ValueError("Require identical protocol and explicit original provenance gap")
    result = copy.deepcopy(validation)
    result.update(
        trials=copy.deepcopy(prior["trials"]),
        warmups=copy.deepcopy(prior["warmups"]),
        derived_cross_model_load_view=True,
        phase_provenance=provenance,
        original_failed_run_preserved=True,
        original_timing_source_after_missing=True,
    )
    return result


def require_space(root: Path) -> int:
    """Fail before the next untimed stage while enough room remains to save its failure."""
    free = shutil.disk_usage(root).free
    if free < 2 * 1024**3:
        raise OSError("Require 2 GiB of remaining space before validation/profiling")
    return free


def main() -> None:
    """Keep original numerical floors, separate profiles, and explicit cross-load provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--vocabulary", required=True, type=int)
    args = parser.parse_args()
    root = args.root
    plan_path = root / "high-concurrency-plan.json"
    plan = json.loads(plan_path.read_bytes())
    validate_plan(plan)
    continuation = json.loads((root / "validation-plan.json").read_bytes())
    if continuation["revision"] != "output16qualv1" or continuation["new_timing_groups"] != 0:
        raise ValueError("Require the frozen validation-only continuation")
    prior_path = root / "original-output16-timing.json"
    prior_sha = hashlib.sha256(prior_path.read_bytes()).hexdigest()
    if prior_sha != continuation["original_timing_report_sha256"]:
        raise ValueError("Original timing report bytes changed")
    prior = json.loads(prior_path.read_bytes())
    protocol = json.loads((root / "protocol.json").read_bytes())
    if (
        hashlib.sha256((root / "protocol.json").read_bytes()).hexdigest()
        != plan["source_protocol_sha256"]
    ):
        raise ValueError("Frozen prompts changed")
    prompts = {
        (case["batch"], case["prompt_tokens"]): case["prompts"] for case in protocol["timing"]
    }
    contexts = [protocol.get("quality_prefix_token_ids", []) + ids for ids in protocol["quality"]]
    path = root / "validation-high-concurrency.json"
    if path.exists() or (root / "candidate-high-concurrency.json").exists():
        raise FileExistsError("Preserve the first validation-only attempt")
    report: dict[str, Any] = {
        "status": "running",
        "label": "candidate",
        "started_epoch": time.time(),
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "quality_qualified": False,
        "previous_failed_gates_preserved": True,
        "new_timing_groups": 0,
        "warmups": [],
        "trials": [],
        "quality": [],
        "profiles": [],
    }

    def save(phase: str) -> None:
        report.update(phase=phase, updated_epoch=time.time())
        write(path, report)

    save("source_before")
    try:
        report["disk_free_before"] = require_space(root)
        audit = json.loads((root / "source-audit.json").read_bytes())
        report["source_before"] = source_check("candidate", audit)
        report["loaded_checks"] = rpc(BASE, "mxwave_checkpoint_check")
        if len(report["loaded_checks"]) != 2 or any(
            row["status"] != "passed" for row in report["loaded_checks"]
        ):
            raise RuntimeError("Loaded source-byte checks failed")
        report["mhc_warmup"] = rpc(BASE, "mxwave_mhc_warmup", [2048])
        report["graph_selection"] = rpc(BASE, "mxwave_set_eager", [False])
        report["snapshot_before"] = snapshot()
        for name, count, length in CASES:
            for window in WINDOWS:
                for index in range(2):
                    configure(window)
                    report["warmups"].append(
                        {
                            "case": name,
                            "window_ms": window,
                            "index": index,
                            "measurement": batch(BASE, prompts[count, length], 16),
                            "admission": admission("mxwave_admission_report"),
                        }
                    )
                    save("fresh_untimed_warmups")
        for count in (8, 16):
            for window in WINDOWS:
                for suffix in ("a", "b"):
                    require_space(root)
                    configure(window)
                    label = f"candidate-c{count}-w{window}-{suffix}"
                    save("untimed_quality_" + label)
                    before = snapshot()
                    collect_concurrent(
                        BASE, contexts, root, label, args.vocabulary, concurrency=count
                    )
                    report["quality"].append(
                        {
                            "label": label,
                            "concurrency": count,
                            "window_ms": window,
                            "before": before,
                            "after": snapshot(),
                            "admission": admission("mxwave_admission_report"),
                        }
                    )
                    save("quality_collected")
        for count in (8, 16):
            for window in WINDOWS:
                require_space(root)
                configure(window)
                profile: dict[str, Any] = {
                    "status": "running",
                    "concurrency": count,
                    "window_ms": window,
                    "after_all_unprofiled_timing": True,
                    "timing_from_previous_model_load": True,
                    "started_epoch": time.time(),
                    "before": snapshot(),
                }
                report["profiles"].append(profile)
                save("separate_profile")
                started = False
                try:
                    request(BASE, "/start_profile", {})
                    started = True
                    profile["measurement"] = batch(BASE, prompts[count, 256], 4)
                    profile["admission"] = admission("mxwave_admission_report")
                    profile["status"] = "complete"
                except Exception as exc:  # noqa: BLE001 -- preserve numerical data if profiling fails.
                    profile.update(status="failed", error=repr(exc))
                finally:
                    if started:
                        request(BASE, "/stop_profile", {}, timeout=240)
                    profile.update(after=snapshot(), finished_epoch=time.time())
                    save("profile_finished")
        configure(0)
        report["source_after"] = source_check("candidate", audit)
        report["snapshot_after"] = snapshot()
        report.update(status="complete", finished_epoch=time.time())
        save("complete")
        provenance = {
            "timing_revision": "output16v1",
            "timing_report": prior_path.name,
            "timing_report_sha256": prior_sha,
            "timing_archive_sha256": continuation["original_timing_archive_sha256"],
            "validation_revision": "output16qualv1",
            "validation_report": path.name,
            "validation_report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "comparison_has_time_gap": True,
            "original_source_after_missing": True,
        }
        write(
            root / "candidate-high-concurrency.json", combined_candidate(prior, report, provenance)
        )
        result = review(root)
        result.update(
            stock_control_reused=True,
            stock_control_revision="c816v1",
            fresh_paired_stock_load=False,
            comparison_has_time_gap=True,
            prior_compressed_revision="c816retryv2",
            prior_compressed_comparisons=parent_comparison(root),
            validation_only_continuation=True,
            output_row16_plan=json.loads((root / "output-row16-plan.json").read_bytes()),
            phase_provenance=provenance,
            original_failed_run_preserved=True,
            original_timing_source_after_missing=True,
            new_timing_groups=0,
            validation_plan=continuation,
            throughput_issue_declared_fixed=False,
            automatic_promotion_allowed=False,
        )
        write(root / "quality-model-review.json", result)
    except BaseException as exc:
        report.update(status="failed", error=repr(exc), finished_epoch=time.time())
        save("failed")
        raise


if __name__ == "__main__":
    main()
