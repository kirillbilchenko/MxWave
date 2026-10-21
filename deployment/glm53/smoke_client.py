"""Check repeated-prefix serving correctness; do not report this as a speed benchmark."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path
from typing import Any


def request(port: int, path: str, body: dict[str, Any] | None = None) -> Any:
    """Call only the loopback test API, retaining complete response objects."""
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/{path}",
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.load(response) if path != "metrics" else response.read().decode()


def main() -> None:
    """Retain prefix reuse and output identity evidence before performance measurements."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    result: dict[str, Any] = {"scope": "startup and prefix-cache smoke only", "responses": []}
    result["models"] = request(args.port, "v1/models")
    if not any(m["max_model_len"] == 500000 for m in result["models"]["data"]):
        raise RuntimeError("Requested 500k context did not survive startup")
    # Freeze real source text during preparation; exceed the hybrid cache block size.
    prompt = (args.root / "prefix-smoke-prompt.txt").read_text()
    body = {"model": "glm-5.3-flash", "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "seed": 42, "max_tokens": 64,
            "chat_template_kwargs": {"reasoning_effort": "low"}}
    result["request"] = body
    result["metrics_before"] = request(args.port, "metrics")
    for _ in range(3):
        start = time.perf_counter()
        response = request(args.port, "v1/chat/completions", body)
        result["responses"].append({"response": response,
                                    "elapsed_seconds": time.perf_counter() - start})
        (args.root / "smoke-progress.json").write_text(json.dumps(result, indent=2) + "\n")
    result["metrics_after"] = request(args.port, "metrics")
    messages = [r["response"]["choices"][0]["message"] for r in result["responses"]]
    result["repeated_outputs_equal"] = messages[0] == messages[1] == messages[2]
    cached = [r["response"].get("usage", {}).get("prompt_tokens_details", {})
              .get("cached_tokens", 0) for r in result["responses"]]
    result["cached_tokens"] = cached

    def hits(metrics: str) -> float:
        return sum(float(line.rsplit(" ", 1)[1]) for line in metrics.splitlines()
                   if line.startswith("vllm:prefix_cache_hits") and not line.startswith("#"))

    result["prefix_hit_counter_delta"] = hits(result["metrics_after"]) - hits(result["metrics_before"])
    result["prefix_reuse_observed"] = any(cached[1:]) or result["prefix_hit_counter_delta"] > 0
    result["status"] = ("passed" if result["repeated_outputs_equal"]
                        and result["prefix_reuse_observed"] else "failed")
    result["speed_qualified"] = False
    result["long_context_qualified"] = False
    (args.root / "smoke-result.json").write_text(json.dumps(result, indent=2) + "\n")
    if result["status"] != "passed":
        raise RuntimeError("Prefix reuse/output identity smoke failed; preserve all responses")


if __name__ == "__main__":
    main()
