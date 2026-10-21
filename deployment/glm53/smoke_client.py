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


def summarize(result: dict[str, Any], *, min_prompt_tokens: int) -> dict[str, Any]:
    """Check actual reuse and repeatability, accepting nullable OpenAI usage details."""
    responses = result["responses"]
    if len(responses) != 3:
        raise ValueError("Require all three retained responses")
    messages = [r["response"]["choices"][0]["message"] for r in responses]
    usages = [r["response"].get("usage") or {} for r in responses]
    cached = [(u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) for u in usages]

    def hits(metrics: str) -> float:
        # Counter creation timestamps are not cache hits.
        names = {"vllm:prefix_cache_hits", "vllm:prefix_cache_hits_total"}
        return sum(float(line.rsplit(" ", 1)[1]) for line in metrics.splitlines()
                   if line and line.split("{", 1)[0].split()[0] in names)

    delta = hits(result["metrics_after"]) - hits(result["metrics_before"])
    equal = messages[0] == messages[1] == messages[2]
    reuse = any(cached[1:]) or delta > 0
    coverage = all(u.get("prompt_tokens", 0) >= min_prompt_tokens for u in usages)
    return {
        "status": "passed" if equal and reuse and coverage else "failed",
        "repeated_outputs_equal": equal, "cached_tokens": cached,
        "prefix_hit_counter_delta": delta, "prefix_reuse_observed": reuse,
        "prompt_covers_cache_blocks": coverage, "min_prompt_tokens": min_prompt_tokens,
        "speed_qualified": False, "long_context_qualified": False,
    }


def main() -> None:
    """Retain prefix reuse and output identity evidence before performance measurements."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--min-prompt-tokens", type=int, default=9280,
                        help="Cover two resolved 4608-token blocks plus a query suffix")
    args = parser.parse_args()
    result: dict[str, Any] = {"scope": "startup and prefix-cache smoke only", "responses": []}
    result["models"] = request(args.port, "v1/models")
    if not any(m["max_model_len"] == 500000 for m in result["models"]["data"]):
        raise RuntimeError("Requested 500k context did not survive startup")
    # Freeze real source text during preparation; cover resolved blocks, not CLI block size.
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
    result.update(summarize(result, min_prompt_tokens=args.min_prompt_tokens))
    (args.root / "smoke-result.json").write_text(json.dumps(result, indent=2) + "\n")
    if result["status"] != "passed":
        raise RuntimeError("Prefix reuse/output identity smoke failed; preserve all responses")


if __name__ == "__main__":
    main()
