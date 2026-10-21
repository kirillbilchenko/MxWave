"""Retained end-to-end timing and full GSM8K; failed qualification stays failed."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import statistics
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

from control_stock_trial import write


class Client:
    """Use identical authenticated requests on either local endpoint."""

    def __init__(self, port: int) -> None:
        self.base = f"http://127.0.0.1:{port}"
        key = Path("/home/kirya/local-spark/runtime/secrets/llm_api_key").read_text().strip()
        self.headers = {"Content-Type": "application/json", "Authorization": "Bearer " + key}

    def request(self, path: str, payload: Any = None, timeout: int = 300) -> bytes:
        """Return the complete response without logging authentication headers."""
        query = urllib.request.Request(
            self.base + path, headers=self.headers,
            data=None if payload is None else json.dumps(payload).encode())
        with urllib.request.urlopen(query, timeout=timeout) as response:
            return response.read()

    def stream(self, prompt: list[int], tokens: int, barrier: threading.Barrier,
               *, natural_stop: bool = False, cache_salt: str = "") -> dict[str, Any]:
        """Retain token arrivals, usage and partial responses even on failure."""
        body = {"model": "glm-5.3-flash", "prompt": prompt, "max_tokens": tokens,
                "temperature": 0, "seed": 42, "stream": True, "return_token_ids": True,
                "stream_options": {"include_usage": True}}
        if not cache_salt:
            raise ValueError("Require the frozen per-request prefix-cache salt")
        body["cache_salt"] = cache_salt
        if not natural_stop:
            body.update(min_tokens=tokens, ignore_eos=True)
        query = urllib.request.Request(self.base + "/v1/completions", headers=self.headers,
                                       data=json.dumps(body).encode())
        barrier.wait(timeout=30)
        start = time.perf_counter()
        result: dict[str, Any] = {"start_epoch": time.time(), "token_ids": [], "events": [],
                                  "text": "", "api_error": False, "prompt_tokens": len(prompt),
                                  "cache_salt": cache_salt}
        try:
            with urllib.request.urlopen(query, timeout=600) as response:
                for line in response:
                    if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                        continue
                    event = json.loads(line[6:])
                    elapsed = time.perf_counter() - start
                    if "error" in event:
                        raise RuntimeError(str(event["error"]))
                    result["events"].append({"seconds": elapsed, "event": event})
                    for choice in event.get("choices", []):
                        ids = choice.get("token_ids") or []
                        text = choice.get("text") or ""
                        if ids or text:
                            result.setdefault("ttft_seconds", elapsed)
                        result["token_ids"].extend(ids)
                        result["text"] += text
                        if choice.get("finish_reason"):
                            result["finish_reason"] = choice["finish_reason"]
                    if event.get("usage"):
                        result["usage"] = event["usage"]
            usage = result.get("usage", {})
            if (usage.get("prompt_tokens") != len(prompt)
                    or usage.get("completion_tokens") != len(result["token_ids"])
                    or not natural_stop and len(result["token_ids"]) != tokens):
                raise RuntimeError("Token IDs/usage do not reconcile with the frozen request")
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            result.update(api_error=True, error=repr(exc))
        result["latency_seconds"] = time.perf_counter() - start
        return result


def counters(raw: str) -> dict[str, float]:
    """Keep labelled cumulative counters and gauges, excluding creation timestamps."""
    values = {}
    for line in raw.splitlines():
        if not line.startswith("vllm:") or "_created" in line:
            continue
        try:
            name, value = line.rsplit(" ", 1)
            values[name] = float(value)
        except ValueError:
            continue
    return values


def delta(before: str, after: str) -> dict[str, float]:
    """Retain all speculative and prefix counter deltas without merging positions."""
    a, b = counters(before), counters(after)
    return {key: value - a.get(key, 0) for key, value in b.items()
            if "spec_decode" in key or "prefix_cache" in key}


def cohort(client: Client, prompts: list[list[int]], tokens: int,
           *, salt: str, natural_stop: bool = False) -> dict[str, Any]:
    """Synchronize admissions and retain queue samples, failures and partial requests."""
    stopped = threading.Event()
    samples: list[dict[str, Any]] = []
    before = client.request("/metrics").decode()

    def sample() -> None:
        while not stopped.is_set():
            try:
                raw = client.request("/metrics", timeout=5).decode()
                values = {k: v for k, v in counters(raw).items() if re.match(
                    r"vllm:(num_requests_(running|waiting)|kv_cache_usage_perc)", k)}
                samples.append({"epoch": time.time(), "values": values})
            except (OSError, ValueError, RuntimeError) as exc:
                samples.append({"epoch": time.time(), "error": repr(exc)})
            stopped.wait(.5)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    start = time.perf_counter()
    barrier = threading.Barrier(len(prompts))
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as pool:
            jobs = [pool.submit(client.stream, p, tokens, barrier, natural_stop=natural_stop,
                                cache_salt=f"{salt}-{i}") for i, p in enumerate(prompts)]
            responses = [job.result() for job in jobs]
    finally:
        stopped.set()
        sampler.join(timeout=10)
    elapsed = time.perf_counter() - start
    after = client.request("/metrics").decode()
    generated = sum(len(r["token_ids"]) for r in responses)
    return {"epoch": time.time(), "wall_seconds": elapsed, "requests": responses,
            "output_tokens": generated, "output_tokens_per_second": generated / elapsed,
            "api_errors": sum(r["api_error"] for r in responses),
            "metrics_before": before, "metrics_after": after, "counters_delta": delta(before, after),
            "queue_samples": samples}


def cache_probe(client: Client, prompts: list[int], salt: str) -> dict[str, Any]:
    """Prove reuse with the same salt and a cold request with a fresh salt."""
    rows = [cohort(client, [prompts], 1, salt=f"{salt}-{suffix}")
            for suffix in ("shared", "shared", "fresh")]
    hits = [sum(v for k, v in row["counters_delta"].items()
                if k.startswith("vllm:prefix_cache_hits_total")) for row in rows]
    return {"requests": rows, "hit_deltas": hits,
            "passed": hits[0] == 0 and hits[1] > 0 and hits[2] == 0
            and not any(row["api_errors"] for row in rows)}


def timing(root: Path, arm: str, trial: int, client: Client) -> None:
    """Measure one immutable block; all outcomes remain diagnostic until gates pass."""
    plan = json.loads((root / "workloads.json").read_bytes())
    output = root / f"timing-{arm}-{trial}"
    output.mkdir()
    state: dict[str, Any] = {"arm": arm, "trial": trial, "status": "running",
                            "qualification_passed": False, "completed_cells": []}
    write(output / "progress.json", state)
    probe = cache_probe(client, plan["cache_probe_ids"], f"{plan['salt']}-probe-{trial}")
    write(output / "cache-probe.json", probe)
    if not probe["passed"]:
        raise RuntimeError("Cannot establish equivalent cold prefix-cache control")
    # Token IDs are frozen once, with both tokenizer byte identities retained.
    cells = plan["cells"] if trial % 2 == 0 else list(reversed(plan["cells"]))
    for cell in cells:
        state.update(cell=cell["name"], phase="warmup")
        write(output / "progress.json", state)
        salt = f"{plan['salt']}-{trial}-{cell['name']}"
        row: dict[str, Any] = {"cell": cell, "trial": trial, "arm": arm}
        row["warmup"] = cohort(client, cell["prompts"], 256, salt=salt + "-warm")
        state["phase"] = "timing"
        write(output / "progress.json", state)
        row["measurement"] = cohort(client, cell["prompts"], 256, salt=salt + "-measure")
        row["cold_prefix_hits"] = sum(v for k, v in row["measurement"]["counters_delta"].items()
                                     if k.startswith("vllm:prefix_cache_hits_total"))
        write(output / (cell["name"] + ".json"), row)
        if row["cold_prefix_hits"] != 0:
            raise RuntimeError("Measured prefix was not cold; keep the failed cell")
        state["completed_cells"].append(cell["name"])
        write(output / "progress.json", state)
    if trial == 5:
        for probe in plan["long_context"]:
            state.update(cell=probe["name"], phase="long_context")
            write(output / "progress.json", state)
            row = {"probe": probe, "measurement": cohort(
                client, [probe["prompt_ids"]], 128, natural_stop=True,
                salt=f"{plan['salt']}-{probe['name']}")}
            answer = row["measurement"]["requests"][0]["text"]
            row["retrieval_passed"] = all(value in answer for value in probe["expected"])
            write(output / (probe["name"] + ".json"), row)
    state.update(status="complete", phase="complete", finished_epoch=time.time())
    write(output / "progress.json", state)


def gsm_case(client: Client, case: dict[str, Any], protocol: dict[str, Any],
             output: Path) -> dict[str, Any]:
    """Count one pinned question exactly once, including API errors and truncation."""
    import gsm8k_benchmark as gsm

    target = output / f"{case['source_index']:04d}.json"
    # Reserve before sending: a crash must not result in an unnoticed retry.
    with target.open("x") as stream:
        json.dump({"status": "started", "source_index": case["source_index"]}, stream)
    row: dict[str, Any] = {"source_index": case["source_index"], "expected": case["expected"],
                           "started_epoch": time.time()}
    payload = gsm.body(case, protocol)
    payload["model"] = "glm-5.3-flash"
    payload["cache_salt"] = f"{protocol['cache_salt_prefix']}-{case['source_index']}"
    try:
        response = json.loads(client.request("/v1/chat/completions", payload, timeout=900))
        row.update(response=response, **gsm.score(response, case["expected"]))
        row["api_error"] = False
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        row.update(api_error=True, error=repr(exc), answer_exact_match=False,
                   strict_format_match=False, truncated=False, naturally_stopped=False)
    row.update(status="complete", finished_epoch=time.time())
    write(target, row)
    return row


def gsm8k(root: Path, arm: str, client: Client) -> None:
    """Run all 1,319 questions at C4, preserving the previously frozen grading policy."""
    import gsm8k_benchmark as gsm

    protocol = json.loads((root / "gsm8k-protocol.json").read_bytes())
    if (len(protocol["cases"]) != 1319 or protocol["speculative_decoding"]
            or protocol["retries"] != 0):
        raise RuntimeError("Require full frozen speculation-off, no-retry GSM8K")
    output = root / f"gsm8k-{arm}"
    output.mkdir()
    before = client.request("/metrics").decode()
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(gsm_case, client, case, protocol, output)
                for case in protocol["cases"]]
        for job in concurrent.futures.as_completed(jobs):
            rows.append(job.result())
            write(output / "progress.json", {"arm": arm, "status": "running",
                                             **gsm.summarize(rows)})
    after = client.request("/metrics").decode()
    spec = {k: v for k, v in delta(before, after).items() if "spec_decode" in k}
    if any(v != 0 for v in spec.values()):
        raise RuntimeError("Speculation counters changed during speculation-off quality")
    write(output / "summary.json", {"arm": arm, "status": "complete",
          "protocol_sha256": hashlib.sha256((root / "gsm8k-protocol.json").read_bytes()).hexdigest(),
          "metrics_before": before, "metrics_after": after, "speculative_counter_deltas": spec,
          **gsm.summarize(rows)})
    write(output / "progress.json", {"arm": arm, "status": "complete", **gsm.summarize(rows)})


def summarize(root: Path) -> dict[str, Any]:
    """Report all trials, ranges and paired question transitions without gate changes."""
    result: dict[str, Any] = {"qualification_passed": False, "cells": [], "quality": {}}
    for arm in ("A", "B"):
        groups: dict[str, list[dict[str, Any]]] = {}
        for path in sorted(root.glob(f"timing-{arm}-*/*.json")):
            row = json.loads(path.read_bytes())
            if "cell" in row and "measurement" in row:
                groups.setdefault(row["cell"]["name"], []).append(row["measurement"])
        for name, rows in sorted(groups.items()):
            speeds = [r["output_tokens_per_second"] for r in rows]
            requests = [r for group in rows for r in group["requests"]]
            ttft = sorted(r["ttft_seconds"] for r in requests if "ttft_seconds" in r)
            latency = [r["latency_seconds"] for r in requests]
            observations: dict[str, list[float]] = {}
            spec: dict[str, float] = {}
            for group in rows:
                for sample in group["queue_samples"]:
                    for key, value in sample.get("values", {}).items():
                        observations.setdefault(key, []).append(value)
                for key, value in group["counters_delta"].items():
                    if "spec_decode" in key:
                        spec[key] = spec.get(key, 0) + value
            result["cells"].append({
                "arm": arm, "cell": name, "trials": len(rows),
                "tok_s_median": statistics.median(speeds), "tok_s_range": [min(speeds), max(speeds)],
                "ttft_mean_seconds": statistics.mean(ttft) if ttft else None,
                "ttft_p95_seconds": ttft[min(len(ttft) - 1, int(.95 * len(ttft)))] if ttft else None,
                "request_latency_mean_seconds": statistics.mean(latency),
                "request_latency_range_seconds": [min(latency), max(latency)],
                "api_errors": sum(r["api_errors"] for r in rows),
                "running_waiting_kv": {key: {"median": statistics.median(values),
                                            "min": min(values), "max": max(values)}
                                      for key, values in observations.items()},
                "speculative_counters_sum": spec,
                "verification_rounds_note": "Retain native draft-call counters as the proxy; "
                    "not a measured target graph-launch count. Per-position acceptance "
                    "denominators are unavailable if the engine omits position proposal counts.",
                "all_trials_retained": True})
        summary = root / f"gsm8k-{arm}/summary.json"
        if summary.exists():
            result["quality"][arm] = json.loads(summary.read_bytes())
    if len(result["quality"]) == 2:
        paired = {}
        for index in range(1319):
            a, b = [json.loads((root / f"gsm8k-{arm}/{index:04d}.json").read_bytes())
                    for arm in ("A", "B")]
            key = f"A_{int(a['answer_exact_match'])}_B_{int(b['answer_exact_match'])}"
            paired[key] = paired.get(key, 0) + 1
        result["quality"]["paired_outcomes"] = paired
    write(root / "comparison-summary.json", result)
    return result


def main() -> None:
    """Run one bounded controller-owned stage or regenerate the retained summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", choices=("A", "B"))
    parser.add_argument("--trial", type=int)
    parser.add_argument("--port", type=int)
    parser.add_argument("--mode", choices=("timing", "gsm8k", "summary"), required=True)
    args = parser.parse_args()
    if args.mode == "summary":
        summarize(args.root)
    elif args.mode == "timing":
        timing(args.root, args.arm, args.trial, Client(args.port))
    else:
        gsm8k(args.root, args.arm, Client(args.port))


if __name__ == "__main__":
    main()
