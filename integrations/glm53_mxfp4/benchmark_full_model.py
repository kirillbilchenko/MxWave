"""Matched API timing, teacher-forced quality and separate whole-engine traces."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

TEXTS = (
    ("A research team compared two measurements of the same river. The first instrument "
    "recorded its depth every hour, while the second recorded changes after each storm. "
    "They checked the calibration before interpreting any difference in the observations. "),
    ("In a distributed database, a transaction can read several records and update one of "
    "them. A concurrent writer may change the version before the transaction commits. "
    "Describe the assumptions carefully when discussing consistency and retries. "),
    ("The train left the coastal station at sunrise and crossed a valley covered in mist. "
    "Passengers could see farms and a narrow canal beyond the bridge. Later, the conductor "
    "announced the next stop and explained how to reach the connecting platform. "),
    ("A computer stores floating point values using a sign, an exponent and a significand. "
    "Lossless storage must reconstruct every original bit. A smaller representation alone "
    "does not establish that its decoder improves the speed of the complete application. "),
    ("Die Bibliothek bewahrt alte Karten und handschriftliche Briefe auf. Vor der "
    "Digitalisierung prüft das Team den Zustand jeder Seite und dokumentiert fehlende "
    "Stellen. Die Bilder werden anschließend mit den ursprünglichen Dokumenten verglichen. "),
    ("У невеликому місті відкрили нову наукову бібліотеку. Студенти можуть працювати з "
    "історичними документами та сучасними дослідженнями. Працівники допомагають знаходити "
    "джерела й пояснюють, як перевіряти походження кожного твердження. "),
    ("Suppose a rectangular garden has a perimeter of forty metres. Its length exceeds "
    "its width by four metres. Define the variables before writing the equations, and "
    "substitute the result into both constraints to check the calculation. "),
    ("The service returned a timeout after the network connection briefly failed. The "
    "client saved a request identifier so that retrying the operation would not create a "
    "duplicate payment. The logs distinguish a failed response from an uncommitted change. "),
)


def request(base: str, route: str, body: dict[str, Any] | None = None,
            timeout: float = 180) -> Any:
    """Make a bounded request to the isolated loopback benchmark server."""
    encoded = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + route, data=encoded,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read()
        return json.loads(raw) if raw else None


def write(path: Path, value: Any) -> None:
    """Atomically retain partial results if later profiling or restoration fails."""
    temporary = path.with_suffix(".incomplete")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def allocations(base: str) -> dict[str, Any]:
    """Require observations from both workers, rather than silently reporting one rank."""
    result = request(base, "/collective_rpc", {
        "method": "mxwave_scale_report", "args": [], "kwargs": {}, "timeout": 60})
    if not isinstance(result, dict) or len(result.get("results", [])) != 2:
        raise RuntimeError("Allocation RPC did not report both TP2 ranks")
    return result


def protocol(base: str, path: Path) -> dict[str, Any]:
    """Freeze exact GLM token IDs once for all three arms, with no source substitution."""
    if path.exists():
        return json.loads(path.read_bytes())
    tokenized = [request(base, "/tokenize", {"model": "mxwave-glm53-pilot",
                    "prompt": text * 160, "add_special_tokens": False})["tokens"]
                 for text in TEXTS]
    if any(len(ids) < 4096 for ids in tokenized):
        raise ValueError("The fixed text pool is too short for the declared cases")
    value = {"version": 1, "source": "Authored diagnostic text, not a standard PPL dataset",
             "texts": list(TEXTS), "timing": [], "quality": [], "quality_gates": {
                 "mean_kl": "max(1e-7, 5 * raw repeat KL)",
                 "mean_nll_absolute_difference": "max(1e-5, 5 * raw repeat difference)",
                 "scale_bytes": "All compressed scale tensors restore original bytes exactly"},
             "timing_warmups": 2, "timing_trials": 5, "generated_tokens": 64,
             "note": "W4A16 Marlin path, TP2, no draft model; fixed KV and prompt tokens"}
    for length in (256, 2048):
        for batch in (1, 4, 8):
            value["timing"].append({"prompt_tokens": length, "batch": batch,
                "prompts": [tokenized[i][37:37 + length] for i in range(batch)]})
    for i in range(16):
        offset = 67 + (i // len(TEXTS)) * 311
        value["quality"].append(tokenized[i % len(TEXTS)][offset:offset + 256])
    write(path, value)
    return value


def completion(base: str, ids: list[int], tokens: int = 64) -> dict[str, Any]:
    """Time stream arrival separately from token generation, retaining every token ID."""
    body = {"model": "mxwave-glm53-pilot", "prompt": ids, "max_tokens": tokens,
            "min_tokens": tokens, "temperature": 0, "seed": 42, "ignore_eos": True,
            "stream": True, "stream_options": {"include_usage": True},
            "return_token_ids": True}
    req = urllib.request.Request(base + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    arrivals, output, usage = [], [], None
    with urllib.request.urlopen(req, timeout=240) as response:
        for line in response:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            event = json.loads(line[6:])
            if "error" in event:
                raise RuntimeError(str(event["error"]))
            for choice in event.get("choices", []):
                delta = choice.get("token_ids") or []
                if delta:
                    arrivals.append({"seconds": time.perf_counter() - start,
                                     "tokens": len(delta)})
                    output.extend(delta)
            usage = event.get("usage") or usage
    end = time.perf_counter() - start
    if len(output) != tokens or not arrivals or usage is None:
        raise ValueError("Incomplete timed completion or missing token/usage evidence")
    decode_seconds = arrivals[-1]["seconds"] - arrivals[0]["seconds"]
    return {"elapsed_seconds": end, "ttft_seconds": arrivals[0]["seconds"],
            "decode_tokens_per_second": ((tokens - arrivals[0]["tokens"]) / decode_seconds
                                         if decode_seconds else None),
            "token_ids": output, "arrivals": arrivals, "usage": usage}


def batch(base: str, prompts: list[list[int]], tokens: int) -> dict[str, Any]:
    """Measure a closed, concurrent group including its total wall time."""
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        futures = [pool.submit(completion, base, ids, tokens) for ids in prompts]
        outputs = [future.result() for future in futures]
    seconds = time.perf_counter() - start
    return {"wall_seconds": seconds, "aggregate_tokens_per_second": len(prompts)*tokens/seconds,
            "median_ttft_seconds": statistics.median(o["ttft_seconds"] for o in outputs),
            "requests": outputs}


def quality(base: str, prompts: list[list[int]], root: Path, label: str,
            vocabulary: int) -> dict[str, Any]:
    """Retain full next-token distributions and exact teacher-forced prompt NLL."""
    results = []
    for index, ids in enumerate(prompts):
        body = {"model": "mxwave-glm53-pilot", "prompt": ids, "max_tokens": 1,
                "min_tokens": 1, "temperature": 0, "seed": 42, "ignore_eos": True,
                "return_token_ids": True, "return_tokens_as_token_ids": True,
                "prompt_logprobs": 1, "logprobs": vocabulary}
        result = request(base, "/v1/completions", body)
        choice = result["choices"][0]
        top = choice["logprobs"]["top_logprobs"][0]
        row = np.full(vocabulary, -np.inf, dtype=np.float32)
        for key, value in top.items():
            if not key.startswith("token_id:"):
                raise ValueError("Full vocabulary logprobs were not returned by token ID")
            row[int(key.removeprefix("token_id:"))] = value
        if len(top) != vocabulary or not np.isfinite(row).any():
            raise ValueError("Missing full next-token distribution")
        probabilities = choice["prompt_logprobs"]
        if len(probabilities) != len(ids):
            raise ValueError("Prompt logprobs do not match the frozen teacher-forced input")
        logprobs = []
        for token, candidates in zip(ids[1:], probabilities[1:], strict=True):
            logprobs.append(float(candidates[str(token)]["logprob"]))
        if not all(math.isfinite(value) for value in logprobs):
            raise ValueError("Non-finite teacher-forced logprob")
        path = root / f"{label}-next-{index:02d}.f32"
        path.write_bytes(row.astype("<f4", copy=False).tobytes())
        results.append({"index": index, "next_logprobs_file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "next_token_ids": choice["token_ids"], "prompt_logprobs": logprobs,
            "nll": -sum(logprobs)/len(logprobs)})
    return {"label": label, "vocabulary": vocabulary, "contexts": results,
            "mean_nll": statistics.mean(x["nll"] for x in results)}


def main() -> None:
    """Collect one qualified arm; profiling requests never contribute to timing medians."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:18784")
    parser.add_argument("--mode", choices=("stock", "raw", "aligned", "kda"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--vocabulary", type=int, required=True)
    args = parser.parse_args()
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{args.mode}-benchmark.json"
    if path.exists():
        raise FileExistsError("Preserve previous benchmark evidence")
    proto = protocol(args.base, root / "protocol.json")
    state: dict[str, Any] = {"status": "running", "mode": args.mode,
        "started_epoch": time.time(), "cases": [],
        "protocol_sha256": hashlib.sha256((root / "protocol.json").read_bytes()).hexdigest()}
    try:
        state["allocation_before"] = allocations(args.base)
        for case in proto["timing"]:
            for _ in range(proto["timing_warmups"]):
                batch(args.base, case["prompts"], 16)
            trials = [batch(args.base, case["prompts"], proto["generated_tokens"])
                      for _ in range(proto["timing_trials"])]
            state["cases"].append({"prompt_tokens": case["prompt_tokens"], "batch": case["batch"],
                "trials": trials, "median_aggregate_tokens_per_second": statistics.median(
                    trial["aggregate_tokens_per_second"] for trial in trials),
                "median_ttft_seconds": statistics.median(
                    trial["median_ttft_seconds"] for trial in trials)})
            write(path, state)
        quality_prompts = [proto.get("quality_prefix_token_ids", []) + ids
                           for ids in proto["quality"]]
        state["quality"] = quality(args.base, quality_prompts, root, args.mode, args.vocabulary)
        if args.mode in ("raw", "stock"):
            state["quality_repeat"] = quality(args.base, quality_prompts, root,
                                               args.mode + "-repeat", args.vocabulary)
        write(path, state)
        if args.mode in ("raw", "aligned", "stock", "kda"):
            request(args.base, "/start_profile", {})
            try:
                # A short decode and a longer concurrent prefill are identifiable
                # by their token counts in detailed annotations in both-rank traces.
                indices = proto.get("profile_case_indices", [0, len(proto["timing"]) - 2])
                state["profile_requests"] = [
                    batch(args.base, proto["timing"][index]["prompts"],
                          proto.get("profile_generated_tokens", 16)) for index in indices]
            finally:
                request(args.base, "/stop_profile", {}, timeout=240)
        state["allocation_after"] = allocations(args.base)
        state.update(status="complete", finished_epoch=time.time())
        write(path, state)
    except BaseException as exc:
        state.update(status="failed", error=repr(exc), updated_epoch=time.time())
        write(path, state)
        raise


if __name__ == "__main__":
    main()
