"""Measure deterministic prompt perplexity through an OpenAI-compatible API."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast


@dataclass(frozen=True)
class ChunkResult:
    """Likelihood measurements for one fixed corpus chunk."""

    index: int
    text_sha256: str
    token_sha256: str
    characters: int
    prompt_tokens: int
    scored_tokens: int
    negative_log_likelihood: float
    mean_negative_log_likelihood: float
    perplexity: float
    elapsed_seconds: float


def build_parser() -> argparse.ArgumentParser:
    """Build the evaluator command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="UTF-8 evaluation corpus")
    parser.add_argument("--output", required=True, help="Destination JSON report")
    parser.add_argument("--model", required=True, help="Served model name")
    parser.add_argument("--api-key-file", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--chunk-characters", type=int, default=8192)
    parser.add_argument("--num-chunks", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--dataset-revision", default="")
    parser.add_argument("--dataset-sha256", default="")
    return parser


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _chunks(text: str, chunk_characters: int, num_chunks: int) -> list[str]:
    if chunk_characters <= 0:
        raise ValueError("--chunk-characters must be positive")
    if num_chunks <= 0:
        raise ValueError("--num-chunks must be positive")
    chunks = [
        text[start : start + chunk_characters]
        for start in range(0, len(text), chunk_characters)
    ]
    chunks = [chunk for chunk in chunks if chunk.strip()]
    if len(chunks) < num_chunks:
        raise ValueError(
            f"Corpus provides only {len(chunks)} non-empty chunks; {num_chunks} requested"
        )
    return chunks[:num_chunks]


def _token_digest(tokens: list[Any]) -> str:
    digest = hashlib.sha256()
    for token in tokens:
        encoded = str(token).encode("utf-8", errors="surrogatepass")
        digest.update(len(encoded).to_bytes(8, byteorder="little"))
        digest.update(encoded)
    return digest.hexdigest()


def _post_json(
    url: str,
    payload: dict[str, object],
    api_key: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                value = json.loads(response.read())
            if not isinstance(value, dict):
                raise TypeError("API response is not a JSON object")
            return cast(dict[str, Any], value)
        except (TimeoutError, urllib.error.HTTPError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(2**attempt)
    raise RuntimeError(f"API request failed after three attempts: {last_error}")


def _score_chunk(
    index: int,
    text: str,
    *,
    url: str,
    model: str,
    api_key: str,
    timeout_seconds: float,
) -> ChunkResult:
    started = time.monotonic()
    response = _post_json(
        url,
        {
            "model": model,
            "prompt": text,
            "max_tokens": 0,
            "echo": True,
            "logprobs": 1,
            "temperature": 0,
        },
        api_key,
        timeout_seconds,
    )
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise TypeError(f"Chunk {index}: API returned an invalid choices list")
    logprobs = choices[0].get("logprobs")
    if not isinstance(logprobs, dict):
        raise TypeError(f"Chunk {index}: API returned no prompt logprobs")
    raw_values = logprobs.get("token_logprobs")
    tokens = logprobs.get("tokens")
    if not isinstance(raw_values, list) or not isinstance(tokens, list):
        raise TypeError(f"Chunk {index}: prompt token arrays are missing")
    if len(raw_values) != len(tokens):
        raise ValueError(f"Chunk {index}: token and logprob lengths differ")
    values = [float(value) for value in raw_values if value is not None]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError(f"Chunk {index}: prompt logprobs are empty or non-finite")
    nll = -sum(values)
    mean_nll = nll / len(values)
    return ChunkResult(
        index=index,
        text_sha256=_sha256_bytes(text.encode()),
        token_sha256=_token_digest(tokens),
        characters=len(text),
        prompt_tokens=len(tokens),
        scored_tokens=len(values),
        negative_log_likelihood=nll,
        mean_negative_log_likelihood=mean_nll,
        perplexity=math.exp(mean_nll),
        elapsed_seconds=time.monotonic() - started,
    )


def run(args: argparse.Namespace) -> Path:
    """Run the fixed-corpus likelihood evaluation and save its JSON report."""
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    input_path = Path(args.input)
    output_path = Path(args.output)
    corpus_bytes = input_path.read_bytes()
    corpus = corpus_bytes.decode("utf-8")
    chunks = _chunks(corpus, args.chunk_characters, args.num_chunks)
    api_key = Path(args.api_key_file).read_text().strip()
    if not api_key:
        raise ValueError("API key file is empty")

    url = f"{args.base_url.rstrip('/')}/completions"
    started = time.monotonic()
    results: list[ChunkResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                _score_chunk,
                index,
                chunk,
                url=url,
                model=args.model,
                api_key=api_key,
                timeout_seconds=args.timeout_seconds,
            )
            for index, chunk in enumerate(chunks)
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"chunk {result.index + 1}/{len(chunks)}: "
                f"tokens={result.scored_tokens} ppl={result.perplexity:.6f}",
                flush=True,
            )
    results.sort(key=lambda item: item.index)

    total_nll = sum(item.negative_log_likelihood for item in results)
    total_scored = sum(item.scored_tokens for item in results)
    mean_nll = total_nll / total_scored
    report = {
        "format": "mxwave-api-perplexity-v1",
        "model": args.model,
        "dataset": args.dataset,
        "dataset_revision": args.dataset_revision,
        "dataset_sha256": args.dataset_sha256,
        "corpus_sha256": _sha256_bytes(corpus_bytes),
        "chunk_characters": args.chunk_characters,
        "num_chunks": len(results),
        "characters": sum(item.characters for item in results),
        "prompt_tokens": sum(item.prompt_tokens for item in results),
        "scored_tokens": total_scored,
        "negative_log_likelihood": total_nll,
        "mean_negative_log_likelihood": mean_nll,
        "perplexity": math.exp(mean_nll),
        "elapsed_seconds": time.monotonic() - started,
        "chunks": [asdict(item) for item in results],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.incomplete")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(output_path)
    print(
        f"aggregate: tokens={total_scored} ppl={report['perplexity']:.6f} "
        f"report={output_path}",
        flush=True,
    )
    return output_path


def main() -> int:
    """Run the evaluator and return a process exit code."""
    try:
        run(build_parser().parse_args())
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"evaluation error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
