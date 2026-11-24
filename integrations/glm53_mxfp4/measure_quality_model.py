"""Evaluate one loaded model arm without collecting new throughput benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import runpy
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from benchmark_full_model import request, write
from check_full_model import rpc
from decode_quality import collect, compare
from quality_suite import BASE, freeze, heldout, tasks


def source_check(label: str, audit: dict[str, Any]) -> list[dict[str, Any]]:
    """Compare every actual scale group on both ranks with the independent source audit."""
    result = rpc(BASE, "mxwave_quality_source_report")
    if len(audit["cases"]) != 168 or {r["tp_rank"] for r in result} != {0, 1}:
        raise ValueError("Incomplete independent source coverage")
    for row in result:
        expected = Counter(
            (tuple(c["shape"]), c["raw_sha256"])
            for c in audit["cases"]
            if c["rank"] == row["tp_rank"]
        )
        actual = Counter((tuple(c["shape"]), c["sha256"]) for c in row["source_scale_hashes"])
        if (
            row["status"] != "passed"
            or row["candidate"] != (label == "candidate")
            or actual != expected
        ):
            raise RuntimeError("Actual loaded source scales differ from the declared model arm")
        if (
            label == "candidate"
            and row["native_sha256"]
            != "9f5d96dcb0762d7e5ed5436ca296f70d590e73797d40522e4d8c7d6f3902f5f0"
        ):
            raise RuntimeError("Loaded candidate is not the qualified U8 shared-stage library")
    return result


def fingerprints(root: Path, prompts: list[list[int]]) -> dict[str, Any]:
    """Retain eager layer hashes when normal repeat windows fail; these calls are intrusive."""
    report: dict[str, Any] = {
        "status": "running",
        "runs": [],
        "limits": [
            "GPU-to-CPU hooks synchronize execution and may change the observed drift.",
            "Fingerprints diagnose repeatability; they cannot qualify normal execution.",
        ],
    }
    path = root / "repeat-fingerprints.json"
    write(path, report)
    for attempt in range(2):
        try:
            armed = rpc(BASE, "mxwave_quality_trace", [True])
            if any(r["status"] != "armed" for r in armed):
                raise RuntimeError("Eager repeat hooks did not arm on both ranks")
            response = request(
                BASE,
                "/v1/completions",
                {
                    "model": "mxwave-glm53-pilot",
                    "prompt": prompts[0],
                    "max_tokens": 4,
                    "min_tokens": 4,
                    "temperature": 0,
                    "seed": 42,
                    "ignore_eos": True,
                    "return_token_ids": True,
                },
            )
        finally:
            trace = rpc(BASE, "mxwave_quality_trace", [False])
        report["runs"].append({"attempt": attempt, "response": response, "workers": trace})
        write(path, report)
    report["status"] = "complete"
    write(path, report)
    return report


def main() -> None:
    """Verify source, collect repeat modes, score frozen test windows and run practical tasks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--label", choices=("control", "candidate"), required=True)
    parser.add_argument("--vocabulary", type=int, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cpu-audit", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--require-stable-repeat", action="store_true")
    args = parser.parse_args()
    path = args.root / (args.label + "-quality-measurement.json")
    if path.exists():
        raise FileExistsError("Preserve prior quality evidence")
    report: dict[str, Any] = {
        "status": "running",
        "label": args.label,
        "started_epoch": time.time(),
        "throughput_benchmark_run": False,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "phases": [],
    }

    def save(phase: str) -> None:
        report.update(phase=phase, updated_epoch=time.time())
        write(path, report)

    try:
        audit = json.loads(args.cpu_audit.read_bytes())
        report["source_before"] = source_check(args.label, audit)
        save("source_verified")
        diagnostic = args.root / (args.label + "-source-diagnostic")
        diagnostic.mkdir()
        sys.argv = [
            str(Path(__file__).with_name("check_full_model.py")),
            "--root",
            str(diagnostic),
            "--protocol",
            str(args.root / "protocol.json"),
            "--vocabulary",
            str(args.vocabulary),
        ]
        runpy.run_path(sys.argv[0], run_name="__main__")
        checked = json.loads((diagnostic / "model-diagnostic.json").read_bytes())
        report["source_diagnostic"] = {
            "status": checked["status"],
            "all_source_checks": all(
                x["status"] == "passed" for x in checked["loaded_weight_checks"]
            ),
            "graph_eager_chats": all(x["sanity_passed"] for x in checked["phases"]),
        }
        if not all(
            (
                report["source_diagnostic"]["all_source_checks"],
                report["source_diagnostic"]["graph_eager_chats"],
            )
        ):
            raise RuntimeError("Loaded-model source/chat gate failed")
        suite = freeze(args.root, args.data)
        report["quality_protocol_sha256"] = hashlib.sha256(
            (args.root / "quality-protocol.json").read_bytes()
        ).hexdigest()
        proto = json.loads((args.root / "protocol.json").read_bytes())
        prompts = [proto.get("quality_prefix_token_ids", []) + ids for ids in proto["quality"]]
        for name, eager in (("graph", False), ("eager", True)):
            save("repeat_" + name)
            dispatch = rpc(BASE, "mxwave_set_eager", [eager])
            if any(x["status"] != "complete" for x in dispatch):
                raise RuntimeError("Requested graph/eager dispatch was not acknowledged")
            request(
                BASE,
                "/v1/completions",
                {
                    "model": "mxwave-glm53-pilot",
                    "prompt": prompts[0],
                    "max_tokens": 4,
                    "temperature": 0,
                    "seed": 42,
                    "ignore_eos": True,
                },
            )
            first = collect(BASE, prompts, args.root, args.label + "-" + name, args.vocabulary)
            second = collect(
                BASE, prompts, args.root, args.label + "-" + name + "-repeat", args.vocabulary
            )
            report["phases"].append(
                {
                    "mode": name,
                    "dispatch": dispatch,
                    "self_repeat": compare(first, second, args.root),
                }
            )
            save("finished_repeat_" + name)
        if any(p["self_repeat"]["mean_decode_kl"] > 1e-5 for p in report["phases"]):
            trace_root = args.root / (args.label + "-repeat-trace")
            trace_root.mkdir()
            save("intrusive_repeat_diagnostic")
            try:
                report["repeat_fingerprints"] = fingerprints(trace_root, prompts)
            except Exception as exc:  # Diagnostic failure must not discard scored quality evidence.
                cleared = rpc(BASE, "mxwave_quality_trace", [False])
                if any(x["status"] != "complete" for x in cleared):
                    raise RuntimeError("Could not remove intrusive diagnostic hooks") from exc
                report["repeat_fingerprints"] = {"status": "failed", "error": repr(exc)}
                write(trace_root / "repeat-fingerprints.json", report["repeat_fingerprints"])
        if args.require_stable_repeat and any(
            p["self_repeat"]["mean_decode_kl"] > 1e-5
            or abs(p["self_repeat"]["mean_decode_nll_difference"]) > 1e-4
            for p in report["phases"]
        ):
            raise RuntimeError("Unchanged strict repeat floor failed; skip longer quality work")
        restored = rpc(BASE, "mxwave_set_eager", [False])
        if any(x["status"] != "complete" for x in restored):
            raise RuntimeError("Default graph dispatch was not restored")
        save("heldout")
        report["heldout"] = heldout(args.root, suite, args.label)
        save("heldout_repeat")
        report["heldout_repeat"] = heldout(args.root, suite, args.label + "-repeat")
        if args.require_stable_repeat and abs(
            report["heldout_repeat"]["mean_nll"] - report["heldout"]["mean_nll"]
        ) > 1e-4:
            raise RuntimeError("Unchanged held-out repeat floor failed; skip practical tasks")
        task_root = args.root / args.label
        task_root.mkdir()
        save("practical_tasks")
        report["practical"] = tasks(task_root, suite)
        save("source_after")
        report["source_after"] = source_check(args.label, audit)
        if args.profile:
            save("profile_after_scoring")
            try:
                request(BASE, "/start_profile", {})
                try:
                    response = request(BASE, "/v1/completions", {
                        "model": "mxwave-glm53-pilot", "prompt": prompts[0],
                        "max_tokens": 4, "min_tokens": 4, "temperature": 0,
                        "seed": 42, "ignore_eos": True, "return_token_ids": True,
                    })
                finally:
                    request(BASE, "/stop_profile", {}, timeout=240)
                report["profile"] = {
                    "status": "complete", "after_all_unprofiled_quality_measurements": True,
                    "generated_token_ids": response["choices"][0]["token_ids"],
                    "location": "Each rank's owned /traces mount; profiler configured at launch",
                }
            except Exception as exc:  # noqa: BLE001 -- retain scored evidence if profiling fails.
                report["profile"] = {"status": "failed", "error": repr(exc)}
        report.update(status="collected_pending_code_sandbox", finished_epoch=time.time())
        save("collected")
    except BaseException as exc:
        report.update(status="failed", error=repr(exc), finished_epoch=time.time())
        save("failed")
        raise


if __name__ == "__main__":
    main()
