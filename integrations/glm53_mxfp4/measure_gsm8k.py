"""Collect every full-test GSM8K answer once, retaining raw responses and bounded concurrency."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from benchmark_full_model import request, write
from check_full_model import rpc
from gsm8k_benchmark import body, score, summarize
from measure_cohort_model_dispatch import BASE, snapshot
from measure_quality_model import source_check


def case_request(
    case: dict[str, Any], protocol: dict[str, Any], directory: Path,
) -> dict[str, Any]:
    """Retain one natural completion or API failure; never retry or omit a source question."""
    path = directory / f"case-{case['source_index']:04d}.json"
    if path.exists():
        raise FileExistsError("Never replace a previous GSM8K answer")
    started = time.perf_counter()
    payload = body(case, protocol)
    row: dict[str, Any] = {"source_index": case["source_index"],
        "prompt_sha256": hashlib.sha256(case["prompt"].encode()).hexdigest(),
        "request_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
        "expected": case["expected"], "answer_exact_match": False, "strict_format_match": False}
    try:
        row["response"] = request(BASE, "/v1/chat/completions", payload, timeout=300)
        row.update(score(row["response"], case["expected"]), api_error=False)
    except Exception as exc:  # noqa: BLE001 -- errors count as failed questions, with raw evidence.
        row.update(api_error=True, error=repr(exc))
    row["elapsed_seconds"] = time.perf_counter() - started
    write(path, row)
    compact = {k: v for k, v in row.items() if k not in ("response", "final_answer")}
    compact.update(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return compact


def main() -> None:
    """Verify source storage, run all questions at C4 and preserve the failed KL gate separately."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--label", choices=("control", "candidate"), required=True)
    parser.add_argument("--vocabulary", type=int, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cpu-audit", type=Path, required=True)
    args = parser.parse_args()
    root, label = args.root, args.label
    protocol_path = root / "gsm8k-protocol.json"
    protocol = json.loads(protocol_path.read_bytes())
    if (len(protocol["cases"]) != 1319 or protocol["concurrency"] != 4
            or protocol["seed"] != 42 or protocol["max_tokens"] != 1024
            or protocol["fewshot"] != 0 or protocol["qualification_attempted"]
            or protocol["speculative_decoding"]):
        raise ValueError("Frozen full-test benchmark protocol changed")
    path = root / (label + "-gsm8k.json")
    if path.exists():
        raise FileExistsError("Preserve the first complete benchmark attempt")
    directory = root / label / "gsm8k"
    directory.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {"status": "running", "label": label, "started_epoch": time.time(),
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "qualification_attempted": False, "previous_failed_gates_preserved": True, "cases": []}

    def save(phase: str) -> None:
        report.update(phase=phase, updated_epoch=time.time(), summary=summarize(report["cases"]))
        write(path, report)

    save("source_checks")
    try:
        audit = json.loads(args.cpu_audit.read_bytes())
        report["source_before"] = source_check(label, audit)
        report["loaded_weight_checks"] = rpc(BASE, "mxwave_checkpoint_check")
        if any(row["status"] != "passed" for row in report["loaded_weight_checks"]):
            raise RuntimeError("Source tensor check failed before the benchmark")
        report["warmup"] = rpc(BASE, "mxwave_mhc_warmup", [2048])
        report["graph_selection"] = rpc(BASE, "mxwave_set_eager", [False])
        report["snapshot_before"] = snapshot(BASE)
        if any(not row["graphs_enabled"] or not row["graph_modes"]["4"].endswith("FULL")
               for row in report["snapshot_before"]):
            raise RuntimeError("Both workers must retain the frozen graph-four configuration")
        report["benchmark_started_epoch"] = time.time()
        save("gsm8k_full_test")
        failures = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(case_request, case, protocol, directory)
                       for case in protocol["cases"]]
            try:
                for future in concurrent.futures.as_completed(futures):
                    row = future.result()
                    report["cases"].append(row)
                    failures = failures + 1 if row["api_error"] else 0
                    if len(report["cases"]) % 8 == 0 or failures or len(report["cases"]) == 1319:
                        save("gsm8k_full_test")
                    if failures >= 4:
                        raise RuntimeError("Four consecutive API failures; stop without retries")
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        report["benchmark_finished_epoch"] = time.time()
        report["cases"].sort(key=lambda row: row["source_index"])
        if [row["source_index"] for row in report["cases"]] != list(range(1319)):
            raise RuntimeError("Incomplete or duplicate test questions")
        save("source_after")
        report["source_after"] = source_check(label, audit)
        report["snapshot_after"] = snapshot(BASE)
        report.update(status="complete", finished_epoch=time.time())
        save("complete")
    except BaseException as exc:
        report.update(status="failed", error=repr(exc), finished_epoch=time.time())
        save("failed")
        raise


if __name__ == "__main__":
    main()
