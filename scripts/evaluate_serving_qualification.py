"""Evaluate deterministic MTP parity and semantic long-context retrieval.

The evaluator has three deliberately separate phases:

* ``prepare`` creates an immutable, fully materialized prompt manifest;
* ``collect`` sends those prompts sequentially to one OpenAI-compatible server;
* ``compare`` checks two collections for emitted-token parity and task accuracy.

Collections are written atomically after every case and can be resumed safely.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

_MANIFEST_FORMAT = "mxwave-serving-qualification-cases-v1"
_COLLECTION_FORMAT = "mxwave-serving-qualification-collection-v1"
_COMPARISON_FORMAT = "mxwave-serving-qualification-comparison-v1"
_SEED = 1234
_TOKENS_PER_DISTRACTOR_RECORD = 21
_LONG_CONTEXT_RESERVE = 250


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _case(
    *,
    case_id: str,
    category: str,
    content: str,
    max_tokens: int,
    scoring: dict[str, object],
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    messages = [{"role": "user", "content": content}]
    return {
        "id": case_id,
        "category": category,
        "messages": messages,
        "prompt_sha256": _sha256_bytes(_canonical_json(messages)),
        "max_tokens": max_tokens,
        "scoring": scoring,
        "metadata": metadata or {},
    }


def _short_cases() -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for index in range(20):
        left = 137 + index * 29
        right = 43 + index * 7
        offset = 5 + index % 7
        expected = str(left * right - offset)
        cases.append(
            _case(
                case_id=f"arithmetic-{index:02d}",
                category="arithmetic",
                content=(
                    f"Calculate ({left} * {right}) - {offset}. "
                    "Return only the integer, without explanation."
                ),
                max_tokens=32,
                scoring={"kind": "exact", "expected": expected},
            )
        )

    for index in range(15):
        values = [91 + index * 3, 17 + index, 63 - index, 42 + index * 2, 28 - index]
        rank = index % 3 + 1
        expected = str(sorted(values)[rank - 1])
        rendered = ", ".join(str(value) for value in values)
        cases.append(
            _case(
                case_id=f"ordering-{index:02d}",
                category="logic",
                content=(
                    f"Values: {rendered}. Return only the {rank} smallest value as an integer."
                ),
                max_tokens=32,
                scoring={"kind": "exact", "expected": expected},
            )
        )

    for index in range(15):
        expected_object = {
            "enabled": index % 2 == 0,
            "items": index + 3,
            "project": f"project-{index:02d}",
        }
        cases.append(
            _case(
                case_id=f"json-{index:02d}",
                category="structured-output",
                content=(
                    "Return one JSON object and no Markdown. It must contain exactly these values: "
                    f"project is project-{index:02d}, items is {index + 3}, and enabled is "
                    f"{'true' if index % 2 == 0 else 'false'}."
                ),
                max_tokens=64,
                scoring={"kind": "json", "expected": expected_object},
            )
        )

    for index in range(10):
        records = [
            f"Record KEY-{slot:02d} has value VALUE-{index:02d}-{slot:02d}."
            for slot in range(12)
        ]
        selected = (index * 7 + 3) % len(records)
        expected = f"VALUE-{index:02d}-{selected:02d}"
        content = "\n".join(records)
        content += f"\nReturn only the value assigned to KEY-{selected:02d}."
        cases.append(
            _case(
                case_id=f"short-retrieval-{index:02d}",
                category="retrieval",
                content=content,
                max_tokens=32,
                scoring={"kind": "exact", "expected": expected},
            )
        )

    code_tasks = (
        "Write a Python function named clamp(value, low, high) with a docstring.",
        "Write a Python function named chunks(items, size) that yields fixed-size lists.",
        "Write a Python function named is_palindrome(text) that ignores case and spaces.",
        "Write a Python function named flatten_once(items) for a list of lists.",
        "Write a Python function named safe_divide(left, right) that returns None on zero.",
        "Write a Python function named dedupe(items) preserving first-seen order.",
        "Write a Python function named parse_bool(value) accepting true/false strings.",
        "Write a Python function named moving_pairs(items) returning adjacent pairs.",
    )
    for index, task in enumerate(code_tasks):
        cases.append(
            _case(
                case_id=f"code-{index:02d}",
                category="code-parity",
                content=task + " Return only one Python code block.",
                max_tokens=160,
                scoring={"kind": "none"},
            )
        )

    summaries = (
        "A deployment restarted twice after a malformed health check, but requests never failed.",
        "The new cache reduced repeated-prefix latency while increasing idle memory slightly.",
        "Three tests passed, one was skipped because the optional image fixture was unavailable.",
        "The migration preserved all rows and reduced the database file by twelve percent.",
        "The model answered correctly but violated the request to return JSON without Markdown.",
    )
    for index, text in enumerate(summaries):
        cases.append(
            _case(
                case_id=f"summary-{index:02d}",
                category="summary-parity",
                content=f"Summarize in exactly seven words: {text}",
                max_tokens=48,
                scoring={"kind": "none"},
            )
        )
    if len(cases) != 73:
        raise AssertionError(f"Expected 73 short cases, created {len(cases)}")
    return cases


def _distractor_records(count: int) -> list[str]:
    return [
        f"Archive record D{index:06d} contains irrelevant value V{index:06d}."
        for index in range(count)
    ]


def _replace_record(records: list[str], ratio: float, value: str) -> int:
    index = min(len(records) - 1, max(0, round((len(records) - 1) * ratio)))
    records[index] = value
    return index


def _long_context_cases(target_tokens: int) -> list[dict[str, object]]:
    record_count = max(
        1,
        (target_tokens - _LONG_CONTEXT_RESERVE) // _TOKENS_PER_DISTRACTOR_RECORD,
    )
    cases: list[dict[str, object]] = []
    target_label = f"{target_tokens // 1000}k"

    for ordinal, ratio in enumerate((0.10, 0.50, 0.90)):
        records = _distractor_records(record_count)
        expected = f"NEBULA-{target_tokens}-{ordinal}-7315"
        position = _replace_record(
            records,
            ratio,
            f"CRITICAL FACT: The Atlas access code is {expected}.",
        )
        records.append(
            "Question: What is the Atlas access code? Return only the code, with no explanation."
        )
        cases.append(
            _case(
                case_id=f"long-{target_label}-single-{ordinal}",
                category="long-single-retrieval",
                content="\n".join(records),
                max_tokens=32,
                scoring={"kind": "exact", "expected": expected},
                metadata={
                    "target_context_tokens": target_tokens,
                    "needle_position_ratio": ratio,
                    "needle_record_index": position,
                    "distractor_records": record_count,
                },
            )
        )

    for ordinal in range(3):
        records = _distractor_records(record_count)
        expected_values = [
            f"EMBER-{target_tokens}-{ordinal}",
            f"GLACIER-{target_tokens}-{ordinal}",
            f"ORBIT-{target_tokens}-{ordinal}",
        ]
        positions = []
        for label, ratio, value in zip(
            ("red", "green", "blue"),
            (0.12 + ordinal * 0.01, 0.50, 0.88 - ordinal * 0.01),
            expected_values,
            strict=True,
        ):
            positions.append(
                _replace_record(
                    records,
                    ratio,
                    f"CRITICAL FACT: The {label} vault contains {value}.",
                )
            )
        expected = "|".join(expected_values)
        records.append(
            "Question: Return the red, green, and blue vault values in that order, joined "
            "with | and with no spaces or explanation."
        )
        cases.append(
            _case(
                case_id=f"long-{target_label}-multi-{ordinal}",
                category="long-multi-retrieval",
                content="\n".join(records),
                max_tokens=48,
                scoring={"kind": "exact", "expected": expected},
                metadata={
                    "target_context_tokens": target_tokens,
                    "needle_position_ratios": [0.12 + ordinal * 0.01, 0.50, 0.88 - ordinal * 0.01],
                    "needle_record_indices": positions,
                    "distractor_records": record_count,
                },
            )
        )

    two_hop_positions = ((0.10, 0.90), (0.25, 0.75), (0.42, 0.58))
    for ordinal, (first_ratio, second_ratio) in enumerate(two_hop_positions):
        records = _distractor_records(record_count)
        archive_key = f"KEY-{target_tokens}-{ordinal}-X9"
        expected = f"QUARTZ-{target_tokens}-{ordinal}-4421"
        first_position = _replace_record(
            records,
            first_ratio,
            f"LINK FACT: Project Orchid delegates authentication to {archive_key}.",
        )
        second_position = _replace_record(
            records,
            second_ratio,
            f"LINK FACT: Archive key {archive_key} resolves to final code {expected}.",
        )
        records.append(
            "Question: Follow the two linked facts. What final authentication code does Project "
            "Orchid use? Return only the final code."
        )
        cases.append(
            _case(
                case_id=f"long-{target_label}-two-hop-{ordinal}",
                category="long-two-hop",
                content="\n".join(records),
                max_tokens=48,
                scoring={"kind": "exact", "expected": expected},
                metadata={
                    "target_context_tokens": target_tokens,
                    "needle_position_ratios": [first_ratio, second_ratio],
                    "needle_record_indices": [first_position, second_position],
                    "distractor_records": record_count,
                },
            )
        )
    return cases


def prepare_manifest(output: Path, context_lengths: list[int]) -> Path:
    """Create the immutable qualification prompt manifest."""
    if not context_lengths:
        raise ValueError("At least one context length is required")
    if len(set(context_lengths)) != len(context_lengths):
        raise ValueError("Context lengths must be unique")
    if any(length < 1000 or length > 64000 for length in context_lengths):
        raise ValueError("Context lengths must be between 1,000 and 64,000 tokens")
    cases = _short_cases()
    for target_tokens in context_lengths:
        cases.extend(_long_context_cases(target_tokens))
    ids = [str(item["id"]) for item in cases]
    if len(set(ids)) != len(ids):
        raise AssertionError("Generated duplicate case IDs")
    manifest = {
        "format": _MANIFEST_FORMAT,
        "seed": _SEED,
        "long_context_generation": {
            "tokens_per_distractor_record_calibration": _TOKENS_PER_DISTRACTOR_RECORD,
            "reserved_tokens": _LONG_CONTEXT_RESERVE,
            "context_lengths": context_lengths,
            "cases_per_context_length": 9,
        },
        "case_count": len(cases),
        "scoreable_case_count": sum(
            cast(dict[str, object], item["scoring"])["kind"] != "none" for item in cases
        ),
        "cases": cases,
    }
    _write_json_atomic(output, manifest)
    print(
        f"prepared {len(cases)} cases ({9 * len(context_lengths)} long-context): {output}",
        flush=True,
    )
    return output


def _load_manifest(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or value.get("format") != _MANIFEST_FORMAT:
        raise ValueError("Unsupported qualification manifest")
    cases = value.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Qualification manifest has no cases")
    return cast(dict[str, Any], value), _sha256_bytes(raw)


def _read_env_secret(path: Path, key: str) -> str:
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() != key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if not value or "\n" in value or "\r" in value:
            raise ValueError(f"{key} must be one non-empty line")
        return value
    raise ValueError(f"{key} is missing from {path}")


def _request_json(
    *,
    url: str,
    api_key: str,
    payload: dict[str, object] | None,
    timeout_seconds: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=None if payload is None else _canonical_json(payload),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="GET" if payload is None else "POST",
    )
    last_error: BaseException | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                value = json.loads(response.read())
            if not isinstance(value, dict):
                raise TypeError("API response is not a JSON object")
            return cast(dict[str, Any], value)
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code < 500 or attempt == 2:
                break
        except (TimeoutError, urllib.error.URLError) as error:
            last_error = error
            if attempt == 2:
                break
        time.sleep(2**attempt)
    detail = type(last_error).__name__ if last_error is not None else "unknown error"
    if isinstance(last_error, urllib.error.HTTPError):
        detail += f" status={last_error.code}"
    raise RuntimeError(f"API request failed after three attempts: {detail}")


def _discover_models(base_url: str, api_key: str, timeout_seconds: float) -> list[str]:
    response = _request_json(
        url=f"{base_url.rstrip('/')}/models",
        api_key=api_key,
        payload=None,
        timeout_seconds=timeout_seconds,
    )
    data = response.get("data")
    if not isinstance(data, list):
        raise TypeError("Model discovery returned no data list")
    result = []
    for item in data:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            result.append(str(item["id"]))
    return sorted(result)


def _score_content(case: dict[str, Any], content: str) -> bool | None:
    scoring = case.get("scoring")
    if not isinstance(scoring, dict):
        raise TypeError(f"Case {case.get('id')} has invalid scoring")
    kind = scoring.get("kind")
    if kind == "none":
        return None
    expected = scoring.get("expected")
    if kind == "exact":
        return content.strip() == expected
    if kind == "json":
        try:
            value = json.loads(content)
        except json.JSONDecodeError:
            return False
        return value == expected
    raise ValueError(f"Case {case.get('id')} has unsupported scorer {kind}")


def _token_trace(choice: dict[str, Any]) -> tuple[list[dict[str, object]], str | None]:
    logprobs = choice.get("logprobs")
    if not isinstance(logprobs, dict):
        return [], None
    raw_entries = logprobs.get("content")
    if not isinstance(raw_entries, list):
        return [], None
    entries: list[dict[str, object]] = []
    digest = hashlib.sha256()
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            raise TypeError("Logprob entry is not an object")
        token = raw_entry.get("token")
        raw_bytes = raw_entry.get("bytes")
        if not isinstance(token, str):
            raise TypeError("Logprob entry has no token string")
        if raw_bytes is None:
            token_bytes = list(token.encode("utf-8", errors="surrogatepass"))
            byte_source = "token-utf8-fallback"
        elif isinstance(raw_bytes, list) and all(
            isinstance(value, int) and 0 <= value <= 255 for value in raw_bytes
        ):
            token_bytes = cast(list[int], raw_bytes)
            byte_source = "api"
        else:
            raise TypeError("Logprob entry has invalid bytes")
        encoded = bytes(token_bytes)
        digest.update(len(encoded).to_bytes(8, byteorder="little"))
        digest.update(encoded)
        entries.append(
            {
                "token": token,
                "bytes": token_bytes,
                "byte_source": byte_source,
                "logprob": raw_entry.get("logprob"),
            }
        )
    return entries, digest.hexdigest()


def _empty_collection(
    *,
    label: str,
    model: str,
    expected_mtp_tokens: int,
    context_length: int,
    manifest_path: Path,
    manifest_sha256: str,
    discovered_models: list[str],
    checkpoint_sha256: str,
) -> dict[str, object]:
    return {
        "format": _COLLECTION_FORMAT,
        "complete": False,
        "checkpoint_sha256": checkpoint_sha256,
        "protocol_sha256": manifest_sha256,
        "label": label,
        "model": model,
        "runtime": {
            "expected_mtp_tokens": expected_mtp_tokens,
            "context_length": context_length,
            "temperature": 0,
            "seed": _SEED,
            "concurrency": 1,
            "enable_thinking": False,
            "logprobs": True,
            "top_logprobs": 0,
        },
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "discovered_models": discovered_models,
        "started_at": _utc_now(),
        "finished_at": None,
        "elapsed_seconds": None,
        "results": [],
    }


def collect(
    *,
    manifest_path: Path,
    output: Path,
    label: str,
    model: str,
    env_file: Path,
    base_url: str,
    expected_mtp_tokens: int,
    context_length: int,
    timeout_seconds: float,
    checkpoint_sha256: str,
) -> Path:
    """Collect one resumable deterministic response set."""
    if expected_mtp_tokens not in {0, 2}:
        raise ValueError("Expected MTP tokens must be 0 or 2")
    if context_length < 1:
        raise ValueError("Context length must be positive")
    manifest, manifest_sha256 = _load_manifest(manifest_path)
    api_key = _read_env_secret(env_file, "LLM_API_KEY")
    discovered_models = _discover_models(base_url, api_key, timeout_seconds)
    if model not in discovered_models:
        raise ValueError(f"Requested model {model!r} is not served: {discovered_models}")

    if output.exists():
        report = json.loads(output.read_bytes())
        if not isinstance(report, dict) or report.get("format") != _COLLECTION_FORMAT:
            raise ValueError("Existing output is not a qualification collection")
        expected_identity = (
            label,
            model,
            manifest_sha256,
            expected_mtp_tokens,
            context_length,
            checkpoint_sha256,
        )
        runtime = report.get("runtime")
        if not isinstance(runtime, dict):
            raise ValueError("Existing collection has no runtime metadata")
        actual_identity = (
            report.get("label"),
            report.get("model"),
            report.get("manifest_sha256"),
            runtime.get("expected_mtp_tokens"),
            runtime.get("context_length"),
            report.get("checkpoint_sha256"),
        )
        if actual_identity != expected_identity:
            raise ValueError("Existing collection identity does not match this run")
        report["complete"] = False
        report["finished_at"] = None
        report["elapsed_seconds"] = None
    else:
        report = _empty_collection(
            label=label,
            model=model,
            expected_mtp_tokens=expected_mtp_tokens,
            context_length=context_length,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
            discovered_models=discovered_models,
            checkpoint_sha256=checkpoint_sha256,
        )

    raw_results = report.get("results")
    if not isinstance(raw_results, list):
        raise TypeError("Collection results are invalid")
    successful_ids = {
        item.get("case_id")
        for item in raw_results
        if isinstance(item, dict) and item.get("error") is None
    }
    raw_results[:] = [
        item
        for item in raw_results
        if isinstance(item, dict) and item.get("error") is None
    ]
    cases = cast(list[dict[str, Any]], manifest["cases"])
    run_started = time.monotonic()
    failures = 0
    for ordinal, case in enumerate(cases, start=1):
        case_id = str(case["id"])
        if case_id in successful_ids:
            print(f"case {ordinal}/{len(cases)} {case_id}: resume-skip", flush=True)
            continue
        payload: dict[str, object] = {
            "model": model,
            "messages": case["messages"],
            "max_tokens": int(case["max_tokens"]),
            "temperature": 0,
            "seed": _SEED,
            "logprobs": True,
            "top_logprobs": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        started = time.monotonic()
        try:
            response = _request_json(
                url=f"{base_url.rstrip('/')}/chat/completions",
                api_key=api_key,
                payload=payload,
                timeout_seconds=timeout_seconds,
            )
            choices = response.get("choices")
            if (
                not isinstance(choices, list)
                or len(choices) != 1
                or not isinstance(choices[0], dict)
            ):
                raise TypeError("API response has no single choice")
            choice = cast(dict[str, Any], choices[0])
            message = choice.get("message")
            if not isinstance(message, dict) or not isinstance(message.get("content"), str):
                raise TypeError("API response has no text content")
            content = str(message["content"])
            trace, trace_sha256 = _token_trace(choice)
            if trace_sha256 is None:
                raise ValueError("API response did not include token logprobs")
            result: dict[str, object] = {
                "case_id": case_id,
                "category": case["category"],
                "prompt_sha256": case["prompt_sha256"],
                "request_sha256": _sha256_bytes(_canonical_json(payload)),
                "elapsed_seconds": time.monotonic() - started,
                "finish_reason": choice.get("finish_reason"),
                "content": content,
                "content_sha256": _sha256_bytes(content.encode()),
                "emitted_token_count": len(trace),
                "emitted_token_bytes_sha256": trace_sha256,
                "token_trace": trace,
                "usage": response.get("usage"),
                "quality_pass": _score_content(case, content),
                "error": None,
            }
            status = "pass" if result["quality_pass"] is not False else "FAIL"
            usage = result.get("usage")
            prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
            print(
                f"case {ordinal}/{len(cases)} {case_id}: {status} "
                f"prompt_tokens={prompt_tokens} elapsed={result['elapsed_seconds']:.2f}s",
                flush=True,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            failures += 1
            result = {
                "case_id": case_id,
                "category": case.get("category"),
                "prompt_sha256": case.get("prompt_sha256"),
                "request_sha256": _sha256_bytes(_canonical_json(payload)),
                "elapsed_seconds": time.monotonic() - started,
                "error": {"type": type(error).__name__, "message": str(error)},
            }
            print(
                f"case {ordinal}/{len(cases)} {case_id}: ERROR {type(error).__name__}",
                flush=True,
            )
        raw_results.append(result)
        report["results"] = raw_results
        _write_json_atomic(output, report)

    ordered = {str(case["id"]): index for index, case in enumerate(cases)}
    raw_results.sort(key=lambda item: ordered[str(item["case_id"])])
    report["complete"] = failures == 0 and len(raw_results) == len(cases)
    report["finished_at"] = _utc_now()
    report["elapsed_seconds"] = time.monotonic() - run_started
    report["results"] = raw_results
    _write_json_atomic(output, report)
    print(
        f"collection {label}: cases={len(raw_results)} failures={failures} report={output}",
        flush=True,
    )
    if failures:
        raise RuntimeError(f"Collection completed with {failures} failed requests")
    return output


def _load_collection(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict) or value.get("format") != _COLLECTION_FORMAT:
        raise ValueError(f"Unsupported qualification collection: {path}")
    return cast(dict[str, Any], value)


def _index_results(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_results = report.get("results")
    if not isinstance(raw_results, list):
        raise TypeError("Collection has no results")
    indexed: dict[str, dict[str, Any]] = {}
    for raw_result in raw_results:
        if not isinstance(raw_result, dict) or not isinstance(raw_result.get("case_id"), str):
            raise TypeError("Collection contains an invalid result")
        case_id = str(raw_result["case_id"])
        if case_id in indexed:
            raise ValueError(f"Collection contains duplicate case {case_id}")
        indexed[case_id] = cast(dict[str, Any], raw_result)
    return indexed


def _quality_summary(
    cases: dict[str, dict[str, Any]],
    results: dict[str, dict[str, Any]],
) -> dict[str, object]:
    groups: dict[str, list[bool]] = defaultdict(list)
    long_lengths: dict[int, list[bool]] = defaultdict(list)
    prompt_tokens_by_length: dict[int, list[int]] = defaultdict(list)
    for case_id, case in cases.items():
        result = results[case_id]
        quality_pass = result.get("quality_pass")
        if isinstance(quality_pass, bool):
            groups[str(case["category"])].append(quality_pass)
        metadata = case.get("metadata")
        if isinstance(metadata, dict) and isinstance(metadata.get("target_context_tokens"), int):
            target = int(metadata["target_context_tokens"])
            if isinstance(quality_pass, bool):
                long_lengths[target].append(quality_pass)
            usage = result.get("usage")
            if isinstance(usage, dict) and isinstance(usage.get("prompt_tokens"), int):
                prompt_tokens_by_length[target].append(int(usage["prompt_tokens"]))

    def summarize(values: list[bool]) -> dict[str, object]:
        passed = sum(values)
        return {"passed": passed, "total": len(values), "accuracy": passed / len(values)}

    return {
        "overall": summarize([value for values in groups.values() for value in values]),
        "by_category": {name: summarize(values) for name, values in sorted(groups.items())},
        "long_context_by_target_tokens": {
            str(target): {
                **summarize(long_lengths[target]),
                "observed_prompt_tokens_min": min(prompt_tokens_by_length[target]),
                "observed_prompt_tokens_max": max(prompt_tokens_by_length[target]),
            }
            for target in sorted(long_lengths)
        },
    }


def compare(
    *,
    manifest_path: Path,
    first_path: Path,
    second_path: Path,
    output: Path,
) -> Path:
    """Compare two collections and enforce parity and semantic gates."""
    manifest, manifest_sha256 = _load_manifest(manifest_path)
    first = _load_collection(first_path)
    second = _load_collection(second_path)
    for report, path in ((first, first_path), (second, second_path)):
        if report.get("manifest_sha256") != manifest_sha256:
            raise ValueError(f"Collection manifest hash does not match: {path}")
        if not report.get("complete"):
            raise ValueError(f"Collection is incomplete: {path}")
        if report.get("protocol_sha256") != manifest_sha256:
            raise ValueError(f"Collection protocol hash does not match: {path}")
    checkpoint_sha256 = first.get("checkpoint_sha256")
    if not isinstance(checkpoint_sha256, str) or second.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("Serving collections do not bind to the same checkpoint")
    first_results = _index_results(first)
    second_results = _index_results(second)
    raw_cases = cast(list[dict[str, Any]], manifest["cases"])
    cases = {str(case["id"]): case for case in raw_cases}
    if set(first_results) != set(cases) or set(second_results) != set(cases):
        raise ValueError("Collection case IDs do not match the manifest")

    per_case = []
    for case_id in cases:
        left = first_results[case_id]
        right = second_results[case_id]
        token_equal = left.get("emitted_token_bytes_sha256") == right.get(
            "emitted_token_bytes_sha256"
        )
        content_equal = left.get("content") == right.get("content")
        finish_equal = left.get("finish_reason") == right.get("finish_reason")
        per_case.append(
            {
                "case_id": case_id,
                "category": cases[case_id]["category"],
                "content_equal": content_equal,
                "emitted_token_bytes_equal": token_equal,
                "finish_reason_equal": finish_equal,
                "first_quality_pass": left.get("quality_pass"),
                "second_quality_pass": right.get("quality_pass"),
                "first_content_sha256": left.get("content_sha256"),
                "second_content_sha256": right.get("content_sha256"),
            }
        )
    token_equal_count = sum(bool(item["emitted_token_bytes_equal"]) for item in per_case)
    content_equal_count = sum(bool(item["content_equal"]) for item in per_case)
    finish_equal_count = sum(bool(item["finish_reason_equal"]) for item in per_case)
    first_quality = _quality_summary(cases, first_results)
    second_quality = _quality_summary(cases, second_results)
    first_long = cast(dict[str, dict[str, object]], first_quality["long_context_by_target_tokens"])
    second_long = cast(
        dict[str, dict[str, object]], second_quality["long_context_by_target_tokens"]
    )
    semantic_gate = all(
        int(summary["passed"]) == int(summary["total"])
        for summary in [*first_long.values(), *second_long.values()]
    )
    parity_gate = token_equal_count == len(per_case)
    report = {
        "format": _COMPARISON_FORMAT,
        "checkpoint_sha256": checkpoint_sha256,
        "protocol_sha256": manifest_sha256,
        "manifest_sha256": manifest_sha256,
        "first": {
            "path": str(first_path),
            "sha256": _sha256_file(first_path),
            "label": first.get("label"),
            "runtime": first.get("runtime"),
            "quality": first_quality,
        },
        "second": {
            "path": str(second_path),
            "sha256": _sha256_file(second_path),
            "label": second.get("label"),
            "runtime": second.get("runtime"),
            "quality": second_quality,
        },
        "parity": {
            "total": len(per_case),
            "content_equal": content_equal_count,
            "emitted_token_bytes_equal": token_equal_count,
            "finish_reason_equal": finish_equal_count,
            "accuracy": token_equal_count / len(per_case),
        },
        "gates": {
            "token_parity": parity_gate,
            "long_context_semantic_accuracy": semantic_gate,
            "pass": parity_gate and semantic_gate,
        },
        "mismatches": [
            item
            for item in per_case
            if not (
                item["content_equal"]
                and item["emitted_token_bytes_equal"]
                and item["finish_reason_equal"]
            )
        ],
        "cases": per_case,
    }
    _write_json_atomic(output, report)
    print(
        f"parity: token_bytes={token_equal_count}/{len(per_case)} "
        f"content={content_equal_count}/{len(per_case)}; "
        f"long_context_gate={'pass' if semantic_gate else 'FAIL'}; report={output}",
        flush=True,
    )
    return output


def _parse_context_lengths(value: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("Context lengths must be comma-separated integers") from error
    if not values:
        raise argparse.ArgumentTypeError("At least one context length is required")
    return values


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument(
        "--context-lengths",
        type=_parse_context_lengths,
        default=[8000, 32000, 60000],
    )

    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--cases", type=Path, required=True)
    collect_parser.add_argument("--output", type=Path, required=True)
    collect_parser.add_argument("--label", required=True)
    collect_parser.add_argument("--model", required=True)
    collect_parser.add_argument("--env-file", type=Path, required=True)
    collect_parser.add_argument("--base-url", default="http://127.0.0.1:18000/v1")
    collect_parser.add_argument("--expected-mtp-tokens", type=int, choices=(0, 2), required=True)
    collect_parser.add_argument("--context-length", type=int, default=65536)
    collect_parser.add_argument("--timeout-seconds", type=float, default=900.0)
    collect_parser.add_argument("--checkpoint-sha256", required=True)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--cases", type=Path, required=True)
    compare_parser.add_argument("--first", type=Path, required=True)
    compare_parser.add_argument("--second", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    """Run the selected qualification phase."""
    args = build_parser().parse_args()
    try:
        if args.command == "prepare":
            prepare_manifest(args.output, args.context_lengths)
        elif args.command == "collect":
            collect(
                manifest_path=args.cases,
                output=args.output,
                label=args.label,
                model=args.model,
                env_file=args.env_file,
                base_url=args.base_url,
                expected_mtp_tokens=args.expected_mtp_tokens,
                context_length=args.context_length,
                timeout_seconds=args.timeout_seconds,
                checkpoint_sha256=args.checkpoint_sha256,
            )
        elif args.command == "compare":
            compare(
                manifest_path=args.cases,
                first_path=args.first,
                second_path=args.second,
                output=args.output,
            )
        else:
            raise AssertionError(f"Unhandled command: {args.command}")
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"evaluation error: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
