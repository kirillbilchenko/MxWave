"""Fixed-token, full-vocabulary self-repeat diagnostic; never a timing benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import urllib.request
from pathlib import Path
from typing import Any


def call(port: int, path: str, body: Any = None) -> Any:
    """Call the private diagnostic API and preserve explicit HTTP failures."""
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/{path}",
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as response:
        return response.read().decode() if path == "metrics" else json.load(response)


def distribution(response: dict[str, Any], vocabulary: int) -> list[float]:
    """Require a complete finite next-token distribution indexed by native token ID."""
    values = response["choices"][0]["logprobs"]["top_logprobs"][0]
    if len(values) != vocabulary or set(values) != {f"token_id:{i}" for i in range(vocabulary)}:
        raise ValueError("Incomplete or ambiguous full-vocabulary response")
    result = [float(values[f"token_id:{i}"]) for i in range(vocabulary)]
    if not all(math.isfinite(x) for x in result):
        raise ValueError("Nonfinite vocabulary log probabilities")
    return result


def compare(left: list[float], right: list[float]) -> dict[str, Any]:
    """Compute normalized KL in float64; retain bit identity separately from the KL gate."""
    def normalized(values: list[float]) -> list[float]:
        largest = max(values)
        offset = largest + math.log(math.fsum(math.exp(x - largest) for x in values))
        return [x - offset for x in values]

    a, b = normalized(left), normalized(right)
    kl = math.fsum(math.exp(x) * (x - y) for x, y in zip(a, b, strict=True))
    return {"logprobs_equal": left == right, "kl": kl, "kl_pass_1e_5": kl < 1e-5,
            "argmax_equal": left.index(max(left)) == right.index(max(right)),
            "max_abs_logprob_delta": max(abs(x - y) for x, y in zip(left, right, strict=True))}


def main() -> None:
    """Retain cold/warm controls and traced repetitions on frozen token sequences."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    root = args.root
    plan = json.loads((root / "numerical-plan.json").read_bytes())
    result: dict[str, Any] = {"scope": "speculation-off fixed-input diagnostic",
                              "status": "running", "cases": {},
                              "quality_qualified": False, "speed_qualified": False}
    result["models"] = call(args.port, "v1/models")
    for case in plan["cases"]:
        bucket: dict[str, Any] = {"runs": {}, "comparisons": {}}
        result["cases"][case["name"]] = bucket
        vectors = {}
        for label, cold, traced in [
            ("cold0", True, False), ("warm0", False, False), ("warm1", False, False),
            ("cold1", True, False), ("trace-cold0", True, True),
            ("trace-cold1", True, True), ("trace-warm0", False, True),
            ("trace-warm1", False, True),
        ]:
            name = f"{case['name']}-{label}"
            if cold:
                for attempt in range(20):
                    reset = call(args.port, "reset_prefix_cache", {})
                    if reset["success"]:
                        break
                    time.sleep(.25)
                else:
                    raise RuntimeError("Prefix reset did not release the previous request")
            row = {"reset_prefix": cold, "traced": traced,
                   "metrics_before": call(args.port, "metrics")}
            if traced:
                row["armed"] = call(args.port, "collective_rpc", {
                    "method": "mxwave_numerical_trace", "args": [name], "timeout": 60})
            body = {"model": "glm-5.3-flash", "prompt": case["token_ids"],
                    "max_tokens": 1, "temperature": 0, "seed": 42,
                    "logprobs": -1, "return_tokens_as_token_ids": True,
                    "return_token_ids": True, "add_special_tokens": False}
            started = time.monotonic()
            try:
                response = call(args.port, "v1/completions", body)
                (root / f"{name}-response.json").write_text(json.dumps(response) + "\n")
                vectors[label] = distribution(response, plan["vocabulary"])
                if response["usage"]["prompt_tokens"] != len(case["token_ids"]):
                    raise RuntimeError("The API changed the frozen token input")
                row.update(elapsed_seconds=time.monotonic() - started,
                           response_file=f"{name}-response.json",
                           response_sha256=hashlib.sha256(
                               (root / f"{name}-response.json").read_bytes()).hexdigest(),
                           output_token_ids=response["choices"][0]["token_ids"])
            finally:
                if traced:
                    trace = call(args.port, "collective_rpc", {
                        "method": "mxwave_numerical_trace", "args": [], "timeout": 90})
                    (root / f"{name}-trace.json").write_text(json.dumps(trace) + "\n")
            row["metrics_after"] = call(args.port, "metrics")
            bucket["runs"][label] = row
            (root / "numerical-progress.json").write_text(json.dumps(result, indent=2) + "\n")
        for left, right in [
            ("cold0", "cold1"), ("warm0", "warm1"), ("cold0", "warm0"),
            ("trace-cold0", "trace-cold1"), ("trace-warm0", "trace-warm1"),
            ("cold0", "trace-cold0"), ("warm0", "trace-warm0"),
        ]:
            bucket["comparisons"][f"{left}-vs-{right}"] = compare(vectors[left], vectors[right])
    result["status"] = "complete"
    (root / "numerical-result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
