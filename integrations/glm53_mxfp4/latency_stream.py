"""Retain absolute client timing for a bounded six-plus-two diagnostic admission."""

from __future__ import annotations

import concurrent.futures
import json
import time
import urllib.request
from collections.abc import Callable
from typing import Any


def stream_one(base: str, prompt: list[int], tokens: int) -> dict[str, Any]:
    """Keep every actual token arrival and an explicit common monotonic clock origin."""
    body = {
        "model": "mxwave-glm53-pilot",
        "prompt": prompt,
        "min_tokens": tokens,
        "max_tokens": tokens,
        "temperature": 0,
        "seed": 42,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
    }
    query = urllib.request.Request(
        base + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    start_ns, epoch_ns = time.perf_counter_ns(), time.time_ns()
    arrivals, output, usage = [], [], None
    with urllib.request.urlopen(query, timeout=240) as response:
        for line in response:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            event = json.loads(line[6:])
            if "error" in event:
                raise RuntimeError(str(event["error"]))
            for choice in event.get("choices", []):
                delta = choice.get("token_ids") or []
                if delta:
                    now_ns = time.perf_counter_ns()
                    arrivals.append(
                        {
                            "monotonic_ns": now_ns,
                            "seconds": (now_ns - start_ns) / 1e9,
                            "tokens": len(delta),
                        }
                    )
                    output.extend(delta)
            usage = event.get("usage") or usage
    end_ns = time.perf_counter_ns()
    if len(output) != tokens or not arrivals or usage is None:
        raise ValueError("Incomplete diagnostic stream")
    if usage["prompt_tokens"] != len(prompt) or usage["completion_tokens"] != tokens:
        raise ValueError("Diagnostic usage does not match the frozen request")
    return {
        "request_start_monotonic_ns": start_ns,
        "request_start_epoch_ns": epoch_ns,
        "request_end_monotonic_ns": end_ns,
        "elapsed_seconds": (end_ns - start_ns) / 1e9,
        "ttft_seconds": arrivals[0]["seconds"],
        "token_ids": output,
        "arrivals": arrivals,
        "usage": usage,
    }


def split_group(
    base: str,
    prompts: list[list[int]],
    tokens: int,
    request: Callable[..., Any],
    *,
    grace_seconds: float = 1.0,
    late_delay_seconds: float = 0.1,
    complete: Callable[[str, list[int], int], dict[str, Any]] = stream_one,
) -> dict[str, Any]:
    """Queue six under keep-pause, resume, then submit two; always attempt resumption."""
    if base != "http://127.0.0.1:18784" or len(prompts) != 8:
        raise ValueError("Split admission is restricted to the isolated eight-request service")
    if not 0.1 <= grace_seconds <= 2 or not 0.02 <= late_delay_seconds <= 0.5:
        raise ValueError("Require bounded diagnostic queue and late-admission delays")
    receipt: dict[str, Any] = {
        "first_requests": 6,
        "late_requests": 2,
        "grace_seconds": grace_seconds,
        "late_delay_seconds": late_delay_seconds,
        "diagnostic_not_serving_speed": True,
        "started_epoch": time.time(),
        "client_start_monotonic_ns": time.perf_counter_ns(),
    }
    paused = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        try:
            receipt["pause"] = request(base, "/pause?mode=keep&clear_cache=false", {})
            paused = True
            if not request(base, "/is_paused")["is_paused"]:
                raise RuntimeError("Isolated scheduler did not pause")
            futures = [pool.submit(complete, base, ids, tokens) for ids in prompts[:6]]
            time.sleep(grace_seconds)
            for future in futures:
                if future.done():
                    future.result()
                    raise RuntimeError("A controlled request completed while paused")
            if not request(base, "/is_paused")["is_paused"]:
                raise RuntimeError("Isolated scheduler resumed unexpectedly")
            receipt["resume_call_monotonic_ns"] = time.perf_counter_ns()
            receipt["resume"] = request(base, "/resume", {})
            receipt["resume_return_monotonic_ns"] = time.perf_counter_ns()
            paused = False
            time.sleep(late_delay_seconds)
            receipt["late_submission_monotonic_ns"] = time.perf_counter_ns()
            futures.extend(pool.submit(complete, base, ids, tokens) for ids in prompts[6:])
            receipt["requests"] = [future.result() for future in futures]
            receipt["status"] = "complete"
        finally:
            if paused:
                receipt["recovery_resume"] = request(base, "/resume", {})
    receipt["client_end_monotonic_ns"] = time.perf_counter_ns()
    receipt["finished_epoch"] = time.time()
    return receipt


def ordinary_group(base: str, prompts: list[list[int]], tokens: int) -> dict[str, Any]:
    """Use ordinary concurrent admission with the same instrumented client clock."""
    start_ns = time.perf_counter_ns()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        futures = [pool.submit(stream_one, base, ids, tokens) for ids in prompts]
        outputs = [future.result() for future in futures]
    end_ns = time.perf_counter_ns()
    return {
        "status": "complete",
        "requests": outputs,
        "client_start_monotonic_ns": start_ns,
        "client_end_monotonic_ns": end_ns,
        "wall_seconds": (end_ns - start_ns) / 1e9,
        "diagnostic_not_serving_speed": True,
    }
