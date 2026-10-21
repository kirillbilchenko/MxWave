"""Bounded selected-CT memory and drafter smoke; not a performance qualification."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path
from typing import Any


def call(port: int, path: str, body: Any = None) -> Any:
    """Call the private loopback service and retain native responses."""
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/{path}",
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as response:
        return response.read().decode() if path == "metrics" else json.load(response)


def counters(metrics: str) -> dict[str, float]:
    """Read speculative cumulative counters while excluding creation timestamps."""
    result: dict[str, float] = {}
    for line in metrics.splitlines():
        if line.startswith("vllm:spec_decode_"):
            name, value = line.rsplit(" ", 1)
            if name.split("{", 1)[0].endswith("_total"):
                result[name] = float(value)
    return result


def main() -> None:
    """Measure the frozen small CT configuration and preserve repeat failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--port", required=True, type=int)
    args = parser.parse_args()
    plan = json.loads((args.root / "profile-plan.json").read_text())
    result: dict[str, Any] = {
        "scope": "memory and speculative execution diagnostic; not a speed comparison",
        "quality_qualified": False, "speed_qualified": False, "responses": [],
        "models": call(args.port, "v1/models"),
    }
    if not any(m["max_model_len"] == 8192 for m in result["models"]["data"]):
        raise RuntimeError("Requested 8k profile did not survive startup")

    def memory() -> Any:
        return call(args.port, "collective_rpc", {
            "method": "mxwave_profile_report", "timeout": 120, "args": [], "kwargs": {},
        })

    def save(name: str) -> None:
        (args.root / name).write_text(json.dumps(result, indent=2) + "\n")

    result["memory_before"] = memory()
    result["metrics_before"] = call(args.port, "metrics")
    save("profile-progress.json")
    for case in plan["cases"]:
        for repeat in range(2):
            body = {"model": "glm-5.3-flash", "prompt": case["prompt_ids"],
                    "max_tokens": 128, "ignore_eos": True, "temperature": 0, "seed": 42,
                    "add_special_tokens": False, "return_token_ids": True}
            before = call(args.port, "metrics")
            started = time.perf_counter()
            response = call(args.port, "v1/completions", body)
            elapsed = time.perf_counter() - started
            after = call(args.port, "metrics")
            a, b = counters(before), counters(after)
            result["responses"].append({
                "id": case["id"], "domain": case["domain"], "repeat": repeat,
                "response": response, "elapsed_seconds_diagnostic_only": elapsed,
                "metrics_before": before, "metrics_after": after,
                "speculative_counters_delta": {k: b[k] - a.get(k, 0) for k in b},
            })
            if response["usage"]["prompt_tokens"] != len(case["prompt_ids"]):
                raise RuntimeError("Frozen input IDs changed")
            if response["usage"]["completion_tokens"] != 128:
                raise RuntimeError("Diagnostic output budget was not completed")
            save("profile-progress.json")
    result["memory_after"] = memory()
    result["metrics_after"] = call(args.port, "metrics")
    result["repeat_outputs_equal"] = {
        case["id"]: rows[0]["response"]["choices"][0] == rows[1]["response"]["choices"][0]
        for case in plan["cases"]
        if len(rows := [r for r in result["responses"] if r["id"] == case["id"]]) == 2
    }
    result["status"] = "collected_not_qualified"
    save("profile-result.json")


if __name__ == "__main__":
    main()
