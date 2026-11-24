"""Compare two eager dispatch modes in one resident v9 model, with live concurrent decode checks."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import statistics
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from benchmark_full_model import batch, request, write
from decode_quality import compare
from measure_quality_model import source_check


def switch(base: str, mode: str) -> list[dict[str, Any]]:
    """Change the worker function only after every running/waiting request gauge reaches zero."""
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        with urllib.request.urlopen(base + "/metrics", timeout=3) as response:
            text = response.read().decode()
        gauges = [float(line.rsplit(" ", 1)[1]) for line in text.splitlines()
                  if line.startswith(("vllm:num_requests_running{", "vllm:num_requests_waiting{"))]
        if len(gauges) >= 2 and all(value == 0 for value in gauges):
            break
        time.sleep(.05)
    else:
        raise RuntimeError("Refuse to switch dispatch with requests still in flight")
    result = request(base, "/collective_rpc", {"method": "mxwave_kda_dispatch", "args": [mode],
                                               "kwargs": {}, "timeout": 60})["results"]
    if len(result) != 2 or any(row["mode"] != mode or row["existing_graphs_recaptured"]
                               or row["weights_or_pools_replaced"] for row in result):
        raise RuntimeError("Both workers must acknowledge the same non-mutating dispatch selection")
    return result


def collect_parallel(
    base: str, prompts: list[list[int]], root: Path, label: str, vocabulary: int,
    *, completion_request: Callable[..., Any] | None = None, gzip_level: int = 9,
) -> dict[str, Any]:
    """Retain complete distributions from four concurrent real-KV requests, not rebuilt prefills."""
    if len(prompts) != 16:
        raise ValueError("Require the original sixteen quality contexts")
    report: dict[str, Any] = {"status": "running", "label": label, "vocabulary": vocabulary,
                              "generated_tokens": 4, "contexts": [], "batch_size": 4,
                              "raw_api_responses": []}
    path = root / (label + "-decode-quality.json")
    if path.exists():
        raise FileExistsError("Preserve the first quality group")
    write(path, report)
    for start in range(0, len(prompts), 4):
        group = prompts[start:start + 4]
        result = (completion_request or request)(base, "/v1/completions", {
            "model": "mxwave-glm53-pilot", "prompt": group, "max_tokens": 4, "min_tokens": 4,
            "temperature": 0, "seed": 42, "ignore_eos": True, "return_token_ids": True,
            "return_tokens_as_token_ids": True, "logprobs": vocabulary}, timeout=300)
        choices = sorted(result["choices"], key=lambda row: row["index"])
        if [row["index"] for row in choices] != list(range(4)):
            raise ValueError("Incomplete concurrent quality response")
        raw_path = root / f"{label}-group{start // 4}.json.gz"
        with gzip.open(raw_path, "wt", encoding="utf-8", compresslevel=gzip_level) as target:
            json.dump(result, target, ensure_ascii=False, allow_nan=False)
        report["raw_api_responses"].append({"file": raw_path.name,
            "sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest()})
        for offset, choice in enumerate(choices):
            index = start + offset
            tokens, tops = choice["token_ids"], choice["logprobs"]["top_logprobs"]
            if len(tokens) != 4 or len(tops) != 4:
                raise ValueError("Concurrent decode did not return all four steps")
            context: dict[str, Any] = {"index": index, "prompt_sha256": hashlib.sha256(
                json.dumps(group[offset], separators=(",", ":")).encode()).hexdigest(),
                "token_ids": tokens, "steps": []}
            for position, top in enumerate(tops):
                values = np.full(vocabulary, -np.inf, dtype="<f4")
                for key, value in top.items():
                    if not key.startswith("token_id:"):
                        raise ValueError("Require vocabulary-indexed log probabilities")
                    token = int(key.removeprefix("token_id:"))
                    if not 0 <= token < vocabulary or math.isnan(value):
                        raise ValueError("Invalid vocabulary distribution")
                    values[token] = value
                if len(top) != vocabulary or not np.isfinite(values).any():
                    raise ValueError("Missing full concurrent decode distribution")
                file = root / f"{label}-ctx{index:02d}-step{position}.f32"
                file.write_bytes(values.tobytes())
                context["steps"].append({"position": position,
                    "kind": "prefill" if position == 0 else "decode",
                    "conditioning_token_ids": tokens[:position], "file": file.name,
                    "sha256": hashlib.sha256(file.read_bytes()).hexdigest()})
            report["contexts"].append(context)
        write(path, report)
    report["status"] = "complete"
    write(path, report)
    return report


def main() -> None:
    """Collect all four quality legs, both timing orders and separate both-rank traces."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--vocabulary", required=True, type=int)
    parser.add_argument("--base", default="http://127.0.0.1:18784")
    args = parser.parse_args()
    if args.base != "http://127.0.0.1:18784":
        raise ValueError("Source verification is bound to the isolated loopback benchmark endpoint")
    plan = json.loads((args.root / "dispatch-plan.json").read_bytes())
    if (plan["quality_order"] != ["qualified-a", "eager-a", "eager-b", "qualified-b"]
            or plan["trials_per_mode_per_order"] != 5 or plan["generated_timing_tokens"] != 64):
        raise ValueError("Frozen dispatch protocol changed")
    proto = json.loads((args.root / "protocol.json").read_bytes())
    prompts = [proto.get("quality_prefix_token_ids", []) + ids for ids in proto["quality"]]
    timing = [case for case in proto["timing"] if case["batch"] == 4
              and case["prompt_tokens"] == 256]
    if len(timing) != 1:
        raise ValueError("Require the existing concurrency-four / 256-token case")
    timed_prompts = timing[0]["prompts"]
    audit = json.loads((args.root / "source-audit.json").read_bytes())
    path = args.root / "eager-dispatch-model-review.json"
    if path.exists():
        raise FileExistsError("Preserve the first full-model dispatch comparison")
    report: dict[str, Any] = {"status": "running", "quality": {}, "timing_blocks": [],
        "profiles": [], "started_epoch": time.time(), "model_loads": 1,
        "both_modes_use_same_v9_compressed_weights": True,
        "previous_full_model_strict_failure_preserved": True,
        "plan_sha256": hashlib.sha256((args.root / "dispatch-plan.json").read_bytes()).hexdigest()}
    write(path, report)
    try:
        report["source_before"] = source_check("candidate", audit)
        for label in plan["quality_order"]:
            mode = "qualified" if label.startswith("qualified") else "eager-direct"
            report["quality"][label] = {"selection": switch(args.base, mode),
                "decode": collect_parallel(args.base, prompts, args.root, label, args.vocabulary)}
            write(path, report)
        comparisons = {}
        for name, left, right in (("qualified_repeat", "qualified-a", "qualified-b"),
                                   ("eager_repeat", "eager-a", "eager-b"),
                                   ("cross_mode", "qualified-a", "eager-a")):
            comparisons[name] = compare(report["quality"][left]["decode"],
                                         report["quality"][right]["decode"], args.root)
        report["quality_comparisons"] = comparisons
        report["numerical_gate_passed"] = all(
            row["expected_decode_steps"] == row["decode_steps"] == 48
            and row["all_decode_conditioning_matches"] and row["all_generated_ids_match"]
            and row["mean_decode_kl"] <= 1e-5
            and abs(row["mean_decode_nll_difference"]) <= 1e-4
            for row in comparisons.values())
        write(path, report)
        for reverse in (False, True):
            names = ["qualified", "eager-direct"]
            for mode in names:
                switch(args.base, mode)
                for _ in range(2):
                    batch(args.base, timed_prompts, 16)
            block: dict[str, Any] = {"reverse_first_order": reverse, "trials": [], "medians": {}}
            report["timing_blocks"].append(block)
            for index in range(5):
                for mode in (names[::-1] if (index + reverse) % 2 else names):
                    selected = switch(args.base, mode)
                    block["trials"].append({"mode": mode, "index": index, "selection": selected,
                                             "measurement": batch(args.base, timed_prompts, 64)})
                    write(path, report)
            for mode in names:
                values = [trial["measurement"] for trial in block["trials"] if trial["mode"] == mode]
                block["medians"][mode] = {"aggregate_tokens_per_second": statistics.median(
                    value["aggregate_tokens_per_second"] for value in values),
                    "ttft_seconds": statistics.median(value["median_ttft_seconds"] for value in values)}
            write(path, report)
        for mode in ("qualified", "eager-direct"):
            selected = switch(args.base, mode)
            request(args.base, "/start_profile", {})
            try:
                measurement = batch(args.base, timed_prompts, 4)
            finally:
                request(args.base, "/stop_profile", {}, timeout=240)
            report["profiles"].append({"mode": mode, "selection": selected,
                                        "measurement": measurement, "finished_epoch": time.time()})
            write(path, report)
        report["source_after"] = source_check("candidate", audit)
        report["final_selection"] = switch(args.base, "qualified")
        report.update(status="complete", finished_epoch=time.time())
        write(path, report)
    except BaseException as exc:
        report.update(status="failed", error=repr(exc), finished_epoch=time.time())
        write(path, report)
        raise


if __name__ == "__main__":
    main()
