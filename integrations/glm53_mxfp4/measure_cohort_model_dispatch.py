"""Prove four-request admission before comparing eager and graph-four numerical behavior."""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

from benchmark_full_model import batch, request, write
from decode_quality import compare
from measure_eager_model_dispatch import collect_parallel, switch
from measure_quality_model import source_check

BASE = "http://127.0.0.1:18784"
ORDER = ["eager-qualified-a", "eager-direct-a", "graph-qualified-a",
         "graph-qualified-b", "eager-direct-b", "eager-qualified-b"]


def snapshot(base: str) -> list[dict[str, Any]]:
    """Require independent counter and graph observations from both tensor-parallel workers."""
    rows = request(base, "/collective_rpc", {"method": "mxwave_batch_snapshot", "args": [],
                    "kwargs": {}, "timeout": 60})["results"]
    if len(rows) != 2 or any(not row["stable_dispatch"]["installed"] for row in rows):
        raise RuntimeError("Both canonical-dispatch workers must report batch coverage")
    return rows


def select(base: str, mode: str) -> dict[str, Any]:
    """Select a host policy and actual graph-manager mode while preserving weights and graphs."""
    if mode not in ("eager-qualified", "eager-direct", "graph-qualified"):
        raise ValueError("Unknown cohort mode")
    host = switch(base, "eager-direct" if mode == "eager-direct" else "qualified")
    eager = mode != "graph-qualified"
    graphs = request(base, "/collective_rpc", {"method": "mxwave_set_eager", "args": [eager],
                         "kwargs": {}, "timeout": 60})["results"]
    if len(graphs) != 2 or any(row.get("status") != "complete" or row["eager"] != eager
                               for row in graphs):
        raise RuntimeError("Both retained graph managers must acknowledge the selected mode")
    observed = snapshot(base)
    expected = "NONE" if eager else "FULL"
    if any(not row["graph_modes"]["4"].endswith(expected) for row in observed):
        raise RuntimeError("Actual four-request descriptor does not select " + expected)
    return {"mode": mode, "host": host, "graphs": graphs, "snapshot": observed}


def batch_proof(
    before: list[dict[str, Any]], after: list[dict[str, Any]],
    prefill_tokens: int, generated_tokens: int, graph: bool,
) -> dict[str, Any]:
    """Require exactly one four-prompt prefill and the expected eager/graph decode coverage."""
    if prefill_tokens <= 4 or len(before) != 2 or len(after) != 2:
        raise ValueError("A cohort proof needs two ranks and a distinct prefill token count")
    expected = {str(prefill_tokens): 42}
    if not graph and generated_tokens > 1:
        expected["4"] = 42 * (generated_tokens - 1)
    deltas = []
    for left, right in zip(before, after, strict=True):
        a, b = (row["stable_dispatch"]["calls_by_tokens"] for row in (left, right))
        delta = {key: b.get(key, 0) - a.get(key, 0) for key in a.keys() | b.keys()
                 if b.get(key, 0) != a.get(key, 0)}
        deltas.append(delta)
    return {"passed": all(delta == expected for delta in deltas),
            "expected_counter_delta_per_rank": expected, "actual_counter_deltas": deltas,
            "scope": "Python canonicalization counters; graph replays do not increment them"}


def cohort_request(
    base: str, body: dict[str, Any], graph: bool, grace_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Queue one four-prompt request under keep-pause and prove its shape after resume."""
    if base != BASE or len(body["prompt"]) != 4:
        raise ValueError("Pause control is restricted to our isolated four-request experiment")
    lengths = {len(ids) for ids in body["prompt"]}
    if len(lengths) != 1:
        raise ValueError("A controlled cohort requires equal prompt lengths")
    before = snapshot(base)
    receipt: dict[str, Any] = {"before": before, "admission_grace_seconds": grace_seconds,
                                "started_epoch": time.time()}
    paused = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        try:
            receipt["pause"] = request(base, "/pause?mode=keep&clear_cache=false", {})
            paused = True
            if not request(base, "/is_paused")["is_paused"]:
                raise RuntimeError("Scheduler did not acknowledge keep-pause")
            future = pool.submit(request, base, "/v1/completions", body, timeout=300)
            time.sleep(grace_seconds)
            if future.done():
                future.result()
                raise RuntimeError("A completion returned before scheduler resume")
            if not request(base, "/is_paused")["is_paused"]:
                raise RuntimeError("Scheduler resumed before controlled admission")
        finally:
            if paused:
                receipt["resume"] = request(base, "/resume", {})
        result = future.result()
    receipt["after"] = snapshot(base)
    receipt["proof"] = batch_proof(before, receipt["after"], 4 * next(iter(lengths)),
                                     body["max_tokens"], graph)
    receipt["finished_epoch"] = time.time()
    return result, receipt


def quality_leg(
    base: str, prompts: list[list[int]], root: Path, label: str, vocabulary: int,
    grace_seconds: float,
) -> dict[str, Any]:
    """Keep every raw distribution and admission receipt; never retry a split group."""
    receipts: list[dict[str, Any]] = []

    def completion(base: str, route: str, body: dict[str, Any], timeout: int) -> Any:
        if route != "/v1/completions":
            raise ValueError("Unexpected controlled quality endpoint")
        result, receipt = cohort_request(base, body, label.startswith("graph-"), grace_seconds)
        receipts.append(receipt)
        write(root / (label + "-batch-proofs.json"), receipts)
        if not receipt["proof"]["passed"]:
            with gzip.open(root / (label + "-rejected-cohort.json.gz"), "wt",
                           compresslevel=1) as target:
                json.dump(result, target, allow_nan=False)
            raise RuntimeError("Actual quality cohort split; keep its evidence and stop")
        return result

    decoded = collect_parallel(base, prompts, root, label, vocabulary,
                               completion_request=completion, gzip_level=1)
    return {"decode": decoded, "batch_proofs": receipts}


def main() -> None:
    """Run the fixed six-leg quality protocol, real API timing and controlled both-rank profiles."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--vocabulary", required=True, type=int)
    parser.add_argument("--base", default=BASE)
    args = parser.parse_args()
    plan_path = args.root / "cohort-plan.json"
    plan = json.loads(plan_path.read_bytes())
    if (args.base != BASE or plan["quality_order"] != ORDER
            or plan["admission_grace_seconds"] != 1.0 or plan["capture_sizes"] != [1, 4]
            or plan["trials_per_mode_per_order"] != 5 or plan["timing_tokens"] != 64):
        raise ValueError("Frozen controlled-cohort protocol changed")
    proto = json.loads((args.root / "protocol.json").read_bytes())
    prompts = [proto.get("quality_prefix_token_ids", []) + ids for ids in proto["quality"]]
    timing = [row for row in proto["timing"] if row["batch"] == 4 and row["prompt_tokens"] == 256]
    if len(timing) != 1 or len(prompts) != 16 or any(len(ids) != 258 for ids in prompts):
        raise ValueError("Require the existing fixed quality and API timing contexts")
    timed_prompts = timing[0]["prompts"]
    audit = json.loads((args.root / "source-audit.json").read_bytes())
    path = args.root / "cohort-dispatch-model-review.json"
    if path.exists():
        raise FileExistsError("Preserve the first controlled-cohort comparison")
    report: dict[str, Any] = {"status": "running", "started_epoch": time.time(), "model_loads": 1,
        "same_v9_weights_and_retained_graphs": True, "previous_failed_gates_preserved": True,
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "smokes": [], "quality": {}, "timing_blocks": [], "profiles": []}
    write(path, report)
    try:
        report["source_before"] = source_check("candidate", audit)
        for mode in ("eager-qualified", "eager-direct", "graph-qualified"):
            chosen = select(args.base, mode)
            result, receipt = cohort_request(args.base, {
                "model": "mxwave-glm53-pilot", "prompt": prompts[:4], "max_tokens": 4,
                "min_tokens": 4, "temperature": 0, "seed": 42, "ignore_eos": True,
                "return_token_ids": True}, mode.startswith("graph-"), 1.0)
            report["smokes"].append({"selection": chosen, "receipt": receipt, "response": result})
            write(path, report)
            if not receipt["proof"]["passed"]:
                raise RuntimeError("Controlled batch smoke failed before expensive quality collection")
        for label in ORDER:
            mode = label.rsplit("-", 1)[0]
            report["quality"][label] = {"selection": select(args.base, mode),
                **quality_leg(args.base, prompts, args.root, label, args.vocabulary, 1.0)}
            write(path, report)
        pairs = [("qualified_repeat", ORDER[0], ORDER[-1]),
                 ("direct_repeat", ORDER[1], ORDER[-2]),
                 ("graph_repeat", ORDER[2], ORDER[3]),
                 ("qualified_vs_direct", ORDER[0], ORDER[1]),
                 ("qualified_vs_graph", ORDER[0], ORDER[2])]
        report["quality_comparisons"] = {name: compare(report["quality"][left]["decode"],
            report["quality"][right]["decode"], args.root) for name, left, right in pairs}
        report["numerical_gate_passed"] = all(
            row["expected_decode_steps"] == row["decode_steps"] == 48
            and row["all_decode_conditioning_matches"] and row["all_generated_ids_match"]
            and row["mean_decode_kl"] <= 1e-5 and abs(row["mean_decode_nll_difference"]) <= 1e-4
            for row in report["quality_comparisons"].values())
        write(path, report)
        names = ["eager-qualified", "graph-qualified"]
        for reverse in (False, True):
            for mode in names:
                select(args.base, mode)
                for _ in range(2):
                    batch(args.base, timed_prompts, 16)
            block: dict[str, Any] = {"reverse_first_order": reverse, "trials": [], "medians": {}}
            report["timing_blocks"].append(block)
            for index in range(5):
                for mode in (names[::-1] if (index + reverse) % 2 else names):
                    chosen = select(args.base, mode)
                    before = snapshot(args.base)
                    measured = batch(args.base, timed_prompts, 64)
                    block["trials"].append({"mode": mode, "index": index, "selection": chosen,
                        "measurement": measured, "before": before, "after": snapshot(args.base)})
                    write(path, report)
            for mode in names:
                values = [row["measurement"] for row in block["trials"] if row["mode"] == mode]
                block["medians"][mode] = {"aggregate_tokens_per_second": statistics.median(
                    row["aggregate_tokens_per_second"] for row in values),
                    "ttft_seconds": statistics.median(row["median_ttft_seconds"] for row in values)}
            write(path, report)
        for mode in names:
            chosen = select(args.base, mode)
            request(args.base, "/start_profile", {})
            try:
                response, receipt = cohort_request(args.base, {
                    "model": "mxwave-glm53-pilot", "prompt": prompts[:4], "max_tokens": 4,
                    "min_tokens": 4, "temperature": 0, "seed": 42, "ignore_eos": True,
                    "return_token_ids": True}, mode.startswith("graph-"), 1.0)
            finally:
                request(args.base, "/stop_profile", {}, timeout=240)
            report["profiles"].append({"selection": chosen, "receipt": receipt,
                                        "response": response, "finished_epoch": time.time()})
            write(path, report)
            if not receipt["proof"]["passed"]:
                raise RuntimeError("Profile cohort shape differs; preserve it and stop")
        report["source_after"] = source_check("candidate", audit)
        report["final_selection"] = select(args.base, "graph-qualified")
        report.update(status="complete", finished_epoch=time.time())
        write(path, report)
    except BaseException as exc:
        report.update(status="failed", error=repr(exc), finished_epoch=time.time())
        write(path, report)
        raise


if __name__ == "__main__":
    main()
