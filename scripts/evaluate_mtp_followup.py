"""Freeze and evaluate MTP parity, a fresh task screen and matched-target context lengths."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import itertools
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from analyze_kl_prefixes import _hash_file, _paired_summary, _read_metrics
from evaluate_fast_pilots import IMAGE, PG19_REVISION, SEED, _download, _engine_kwargs
from evaluate_next_token_kl import _extract_full_logprobs, _model_vocab_size, _token_digest
from evaluate_quantization_controls import _load_protocol as _load_previous
from evaluate_quantization_controls import _write_json
from mtp_task_screen import score, task_cases
from run_mtp_followup import _source_digest
from safetensors import safe_open
from safetensors.torch import save_file

LENGTHS = (512, 2048, 8192)
TARGETS = (12288, 16384)
FORMAT = "mxwave-mtp-followup-protocol-v1"


def _load(path: Path) -> tuple[dict[str, Any], str]:
    document = json.loads(path.read_bytes())
    if document["format"] != FORMAT:
        raise ValueError("Unsupported follow-up protocol")
    ref = document["token_data"]
    if Path(ref["filename"]).name != ref["filename"]:
        raise ValueError("Token sidecar must be a sibling")
    token_path = path.parent / ref["filename"]
    if _hash_file(token_path) != ref["sha256"]:
        raise ValueError("Token sidecar hash mismatch")
    with safe_open(str(token_path), framework="pt", device="cpu") as source:
        if list(source.keys()) != ["token_ids"]:
            raise ValueError("Unexpected token sidecar tensors")
        tokens = source.get_tensor("token_ids")
    if tokens.ndim != 1 or tokens.dtype != torch.int64 or len(tokens) != ref["count"]:
        raise ValueError("Unexpected sidecar dtype or shape")
    cursor = 0
    for book in document["books"]:
        if book["token_offset"] != cursor or book["token_count"] <= max(LENGTHS):
            raise ValueError("Invalid book token range")
        book["token_ids"] = tokens[cursor : cursor + book["token_count"]].tolist()
        if _token_digest(book["token_ids"]) != book["token_ids_sha256"]:
            raise ValueError("Book token hash mismatch")
        cursor += book["token_count"]
    if cursor != len(tokens):
        raise ValueError("Unreferenced sidecar tokens")
    for context in document["contexts"]:
        book = document["books"][context["book_index"]]
        end, length = context["end_offset"], context["prefix_tokens"]
        if not 0 <= end - length < end < len(book["token_ids"]):
            raise ValueError("Invalid context end/length")
        context["token_ids"] = book["token_ids"][end - length : end]
        if (
            _token_digest(context["token_ids"]) != context["token_ids_sha256"]
            or book["token_ids"][end] != context["target_token_id"]
        ):
            raise ValueError("Context hash or target mismatch")
    for case in document["diagnostics"] + document["tasks"]:
        if _token_digest(case["token_ids"]) != case["token_ids_sha256"]:
            raise ValueError("Prompt token hash mismatch")
    return document, _hash_file(path)


def prepare(args: argparse.Namespace) -> None:
    """Freeze three reused diagnostic prompts, new tasks and new books before inference."""
    from transformers import AutoTokenizer

    root, previous_root = Path(args.output_dir), Path(args.previous_root)
    if (root / "protocol.json").exists() or (root / "protocol-tokens.safetensors").exists():
        raise FileExistsError("Protocol already frozen")
    if not 16 <= args.books <= 32:
        raise ValueError("The follow-up requires 16 to 32 new independent books")
    old, old_sha = _load_previous(previous_root / "validation/protocol.json")
    previous_comparison = json.loads((previous_root / "pilot-comparison.json").read_text())
    if previous_comparison["protocol_sha256"] != old_sha:
        raise ValueError("Previous comparison protocol changed")
    reports = {}
    for name in ("h64-mtp0", "h64-mtp2"):
        path = previous_root / "measurements" / f"{name}.json"
        report = json.loads(path.read_text())
        if (
            not report["complete"]
            or report["protocol_sha256"] != old_sha
            or _hash_file(path) != previous_comparison["report_sha256"][name]
        ):
            raise ValueError("Previous report identity mismatch")
        reports[name] = report
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    diagnostics = []
    for case in old["serving_cases"]:
        sequences = [
            next(
                r["token_ids"]
                for r in reports[name]["serving"]
                if r["case_id"] == case["id"] and r["repetition"] == 0
            )
            for name in ("h64-mtp0", "h64-mtp2")
        ]
        if sequences[0] == sequences[1]:
            continue
        first = next(i for i, (a, b) in enumerate(zip(*sequences, strict=True)) if a != b)
        diagnostics.append(
            {
                **case,
                "first_difference": first,
                "previous_off_tokens": sequences[0],
                "previous_on_tokens": sequences[1],
            }
        )
    if len(diagnostics) != 3:
        raise ValueError("Expected the three original mismatches")
    tasks = task_cases()
    for case in tasks:
        case["id"] = "fresh-" + case["id"]
        case["messages"] = [{"role": "user", "content": case.pop("content")}]
        text = tokenizer.apply_chat_template(
            case["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        case["token_ids"] = tokenizer.encode(text, add_special_tokens=False)
        case["token_ids_sha256"] = _token_digest(case["token_ids"])
    manifest_url = (
        "https://huggingface.co/datasets/deepmind/pg19/resolve/"
        f"{PG19_REVISION}/data/validation_files.txt"
    )
    manifest = _download(manifest_url, root / "dataset-source/validation_files.txt")
    excluded = {w["book"] for w in old["kl_windows"]}
    books, contexts, tokens, skipped = [], [], [], []
    for filename in manifest.decode().splitlines():
        if not filename.startswith("validation/") or ".." in filename:
            raise ValueError("Unexpected PG-19 manifest entry")
        if filename in excluded:
            continue
        url = f"https://storage.googleapis.com/deepmind-gutenberg/{filename}"
        data = _download(url, root / "dataset-source" / filename)
        text = data.decode()
        raw = tokenizer.encode(text[:200000], add_special_tokens=False)
        if len(raw) <= max(TARGETS):
            skipped.append(filename)
            continue
        start = min(TARGETS) - max(LENGTHS)
        values = raw[start : max(TARGETS) + 1]
        index = len(books)
        books.append(
            {
                "index": index,
                "book": filename,
                "source_url": url,
                "source_sha256": hashlib.sha256(data).hexdigest(),
                "character_start": 0,
                "character_end": min(len(text), 200000),
                "source_token_start": start,
                "token_offset": len(tokens),
                "token_count": len(values),
                "token_ids_sha256": _token_digest(values),
            }
        )
        tokens.extend(values)
        for target in TARGETS:
            end = target - start
            for length in LENGTHS:
                contexts.append(
                    {
                        "book_index": index,
                        "source_target_position": target,
                        "end_offset": end,
                        "prefix_tokens": length,
                        "target_token_id": values[end],
                        "token_ids_sha256": _token_digest(values[end - length : end]),
                    }
                )
        if len(books) == args.books:
            break
    if len(books) != args.books:
        raise ValueError("Too few eligible, previously unused books")
    root.mkdir(parents=True, exist_ok=True)
    path = root / "protocol-tokens.safetensors"
    save_file({"token_ids": torch.tensor(tokens, dtype=torch.int64)}, str(path))
    settings = {
        "repetitions": 2,
        "max_tokens": 384,
        "temperature": 0,
        "ignore_eos": False,
        "thinking": False,
        "concurrency": 1,
        "diagnostic_max_tokens": 192,
        "diagnostic_logprobs": 8,
    }
    document = {
        "format": FORMAT,
        "executed_source_sha256": _source_digest(),
        "runtime_image": IMAGE,
        "previous_protocol_sha256": old_sha,
        "previous_report_sha256": {
            n: _hash_file(previous_root / "measurements" / f"{n}.json") for n in reports
        },
        "dataset": {
            "repository": "deepmind/pg19",
            "revision": PG19_REVISION,
            "split": "validation",
            "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
            "excluded_previous_books": sorted(excluded),
            "skipped_short_books": skipped,
            "selection": "first eligible unused books in pinned manifest",
        },
        "books": books,
        "contexts": contexts,
        "diagnostics": diagnostics,
        "tasks": tasks,
        "token_data": {"filename": path.name, "sha256": _hash_file(path), "count": len(tokens)},
        "serving": settings,
        "seed": SEED,
        "profiles": {
            f"{label}-{profile}-mtp{mtp}": _kwargs("/model", label, profile, mtp)
            for label, profile, mtp in (
                ("h64", "graphs", 0),
                ("h64", "graphs", 2),
                ("h64", "eager", 0),
                ("h64", "eager", 2),
                ("bf16", "graphs", 0),
            )
        },
        "scope": "diagnostic reused prompts; fresh authored tasks; exploratory matched "
        "target-token length screen; no deployment or promotion",
    }
    _write_json(root / "protocol.json", document)
    _load(root / "protocol.json")
    print(
        f"FROZEN {len(books)} new books, {len(contexts)} positions, {len(tasks)} tasks", flush=True
    )


def _kwargs(model: str, label: str, profile: str, mtp: int) -> dict[str, Any]:
    kwargs = _engine_kwargs(model, label, mtp)
    kwargs["logprobs_mode"] = "raw_logprobs"
    if profile == "eager":
        kwargs["enforce_eager"] = True
    return kwargs


def _timing(events: list[list[float]], tokens: list[int]) -> dict[str, Any]:
    if not tokens or not events or events[-1][1] != len(tokens):
        raise ValueError("Missing stream tokens")
    times = [e[0] for e in events]
    counts = [e[1] for e in events]
    if (
        not all(math.isfinite(t) and t > 0 for t in times)
        or any(a > b for a, b in itertools.pairwise(times))
        or any(a >= b for a, b in itertools.pairwise(counts))
    ):
        raise ValueError("Invalid streaming sequence")
    after_first = len(tokens) - counts[0]
    return {
        "ttft_seconds": times[0],
        "last_token_seconds": times[-1],
        "tpot_seconds": (times[-1] - times[0]) / after_first if after_first else None,
        "output_tokens": len(tokens),
        "events": events,
    }


async def _request(engine: Any, tokens: list[int], params: Any, name: str) -> dict[str, Any]:
    events, final = [], None
    started = time.monotonic()
    async for result in engine.generate({"prompt_token_ids": tokens}, params, name):
        final = result
        count = len(result.outputs[0].token_ids) if result.outputs else 0
        if count and (not events or count > events[-1][1]):
            events.append([time.monotonic() - started, count])
    if final is None or not final.finished or len(final.outputs) != 1:
        raise ValueError("Incomplete request")
    completion = final.outputs[0]
    values = list(completion.token_ids)
    output = {
        "token_ids": values,
        "token_ids_sha256": _token_digest(values),
        "text": completion.text,
        "finish_reason": completion.finish_reason,
        "prompt_sha256": _token_digest(tokens),
        "prompt_tokens": len(tokens),
        "speculative_decoding": (
            completion.spec_decode_metrics.to_dict()
            if completion.spec_decode_metrics is not None
            else None
        ),
        **_timing(events, values),
    }
    if params.logprobs is not None:
        if completion.logprobs is None or len(completion.logprobs) != len(values):
            raise ValueError("Missing diagnostic probabilities")
        output["logprobs"] = [
            {
                str(token): {
                    "logprob": value.logprob,
                    "rank": value.rank,
                    "decoded_token": value.decoded_token,
                }
                for token, value in row.items()
            }
            for row in completion.logprobs
        ]
    return output


async def _collect(args: argparse.Namespace) -> None:
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.logprobs import FlatLogprobs
    from vllm.v1.engine.async_llm import AsyncLLM

    protocol, digest = _load(Path(args.protocol))
    if protocol["executed_source_sha256"] != _source_digest():
        raise ValueError("Executed sources changed after freezing")
    name = f"{args.label}-{args.profile}-mtp{args.mtp_tokens}-{args.phase}"
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if (root / f"{name}.json").exists():
        raise FileExistsError("Refusing to overwrite complete measurement")
    kwargs = _kwargs(args.model, args.label, args.profile, args.mtp_tokens)
    if kwargs != protocol["profiles"][f"{args.label}-{args.profile}-mtp{args.mtp_tokens}"]:
        raise ValueError("Runtime differs from frozen profile")
    started = time.monotonic()
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**kwargs))
    report: dict[str, Any] = {
        "format": "mxwave-mtp-followup-measurement-v1",
        "name": name,
        "protocol_sha256": digest,
        "executed_source_sha256": _source_digest(),
        "checkpoint_sha256": args.checkpoint_sha256,
        "runtime_image": IMAGE,
        "engine_kwargs": kwargs,
        "load_seconds": time.monotonic() - started,
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "diagnostics": [],
        "conditioned": [],
        "tasks": [],
    }
    try:
        if args.phase == "long":
            if args.mtp_tokens or args.profile != "graphs":
                raise ValueError("Long-context comparison requires the normal MTP-off profile")
            vocab, _ = _model_vocab_size(Path(args.model))
            matrix = torch.empty((len(protocol["contexts"]), vocab), dtype=torch.float32)
            params = SamplingParams(
                max_tokens=1,
                temperature=1,
                seed=SEED,
                logprobs=-1,
                flat_logprobs=True,
                detokenize=False,
            )
            for i, context in enumerate(protocol["contexts"]):
                final = None
                async for result in engine.generate(
                    {"prompt_token_ids": context["token_ids"]}, params, f"long-{i}"
                ):
                    final = result
                if final is None or not final.finished or len(final.outputs) != 1:
                    raise ValueError("Incomplete distribution request")
                probabilities = final.outputs[0].logprobs
                if not isinstance(probabilities, FlatLogprobs):
                    raise TypeError("Expected full-vocabulary FlatLogprobs")
                matrix[i] = _extract_full_logprobs(probabilities, vocab)
                print(f"{name}: KL {i + 1}/{len(matrix)}", flush=True)
            path = root / f"{args.label}-matched-logprobs.safetensors"
            if path.exists():
                raise FileExistsError("Refusing to overwrite distribution matrix")
            save_file({"logprobs": matrix}, str(path), metadata={"protocol_sha256": digest})
            report["distributions"] = {"sha256": _hash_file(path), "positions": len(matrix)}
        else:
            settings = protocol["serving"]
            for i, case in enumerate(protocol["diagnostics"]):
                await _request(
                    engine,
                    case["token_ids"],
                    SamplingParams(max_tokens=32, temperature=0),
                    f"warmup-{i}",
                )
            for repetition in range(settings["repetitions"]):
                for case in protocol["diagnostics"]:
                    item = await _request(
                        engine,
                        case["token_ids"],
                        SamplingParams(max_tokens=192, temperature=0, seed=SEED, logprobs=8),
                        f"trace-{case['id']}-{repetition}",
                    )
                    report["diagnostics"].append(
                        {"case_id": case["id"], "repetition": repetition, **item}
                    )
                    _write_json(root / f"{name}-progress.json", report)
                    print(f"{name}: trace {case['id']} repetition {repetition}", flush=True)
            for case in protocol["diagnostics"]:
                prefix = case["token_ids"] + case["previous_off_tokens"][: case["first_difference"]]
                item = await _request(
                    engine,
                    prefix,
                    SamplingParams(max_tokens=1, temperature=0, seed=SEED, logprobs=8),
                    f"condition-{case['id']}",
                )
                report["conditioned"].append({"case_id": case["id"], **item})
            if args.profile == "graphs":
                for i in (0, 8):
                    await _request(
                        engine,
                        protocol["tasks"][i]["token_ids"],
                        SamplingParams(max_tokens=32, temperature=0),
                        f"task-warm-{i}",
                    )
                for repetition in range(settings["repetitions"]):
                    for case in protocol["tasks"]:
                        item = await _request(
                            engine,
                            case["token_ids"],
                            SamplingParams(
                                max_tokens=settings["max_tokens"], temperature=0, seed=SEED
                            ),
                            f"task-{case['id']}-{repetition}",
                        )
                        if args.mtp_tokens and (
                            item["speculative_decoding"] is None
                            or item["speculative_decoding"]["num_draft_tokens"] <= 0
                        ):
                            raise ValueError("No genuine MTP proposals in task request")
                        report["tasks"].append(
                            {
                                "case_id": case["id"],
                                "category": case["category"],
                                "repetition": repetition,
                                **item,
                            }
                        )
                        _write_json(root / f"{name}-progress.json", report)
                        print(f"{name}: task {len(report['tasks'])}/32", flush=True)
        report["complete"] = True
        _write_json(root / f"{name}.json", report)
    finally:
        engine.shutdown()


def _margin(row: dict[str, Any], chosen: int, other: int) -> dict[str, Any]:
    if (
        not row
        or str(chosen) not in row
        or any(not math.isfinite(item["logprob"]) for item in row.values())
    ):
        raise ValueError("Invalid diagnostic probabilities")
    ranked = sorted(row.items(), key=lambda item: (-item[1]["logprob"], int(item[0])))
    best = ranked[0][1]["logprob"]
    selected = row[str(chosen)]["logprob"]
    return {
        "selected_token": chosen,
        "selected_is_argmax": selected >= best - 1e-6,
        "top1_top2_logit_margin": best - ranked[1][1]["logprob"] if len(ranked) > 1 else None,
        "chosen_minus_other_logit": (
            selected - row[str(other)]["logprob"] if str(other) in row else None
        ),
        "top_tokens": row,
    }


def _diagnostic_compare(off: dict[str, Any], on: dict[str, Any]) -> list[dict[str, Any]]:
    a = {(r["case_id"], r["repetition"]): r for r in off["diagnostics"]}
    b = {(r["case_id"], r["repetition"]): r for r in on["diagnostics"]}
    if a.keys() != b.keys():
        raise ValueError("Diagnostic pairing mismatch")
    rows = []
    for key, left in a.items():
        right = b[key]
        if left["prompt_sha256"] != right["prompt_sha256"]:
            raise ValueError("Diagnostic prompt mismatch")
        first = next(
            (
                i
                for i, (x, y) in enumerate(zip(left["token_ids"], right["token_ids"], strict=False))
                if x != y
            ),
            None,
        )
        row = {
            "case_id": key[0],
            "repetition": key[1],
            "exact_parity": left["token_ids"] == right["token_ids"],
            "first_difference": first,
        }
        if first is not None:
            x, y = left["token_ids"][first], right["token_ids"][first]
            common = left["logprobs"][first].keys() & right["logprobs"][first].keys()
            row.update(
                {
                    "common_prefix_sha256": _token_digest(left["token_ids"][:first]),
                    "off": _margin(left["logprobs"][first], x, y),
                    "on": _margin(right["logprobs"][first], y, x),
                    "common_top_token_logprob_differences": {
                        token: right["logprobs"][first][token]["logprob"]
                        - left["logprobs"][first][token]["logprob"]
                        for token in common
                    },
                }
            )
        rows.append(row)
    return rows


def _task_compare(
    protocol: dict[str, Any], off: dict[str, Any], on: dict[str, Any]
) -> dict[str, Any]:
    a = {(r["case_id"], r["repetition"]): r for r in off["tasks"]}
    b = {(r["case_id"], r["repetition"]): r for r in on["tasks"]}
    cases = {c["id"]: c for c in protocol["tasks"]}
    expected = {(c, r) for c in cases for r in range(protocol["serving"]["repetitions"])}
    if a.keys() != expected or b.keys() != expected:
        raise ValueError("Incomplete task pairing")
    rows = []
    for key in sorted(expected):
        case, left, right = cases[key[0]], a[key], b[key]
        if left["prompt_sha256"] != right["prompt_sha256"] or (
            left["prompt_sha256"] != case["token_ids_sha256"]
        ):
            raise ValueError("Task prompt identity mismatch")
        rows.append(
            {
                "case_id": key[0],
                "repetition": key[1],
                "category": case["category"],
                "off": score(case, left["text"]),
                "on": score(case, right["text"]),
                "exact_token_parity": left["token_ids"] == right["token_ids"],
                "off_finish_reason": left["finish_reason"],
                "on_finish_reason": right["finish_reason"],
                "latency_difference_seconds": right["last_token_seconds"]
                - left["last_token_seconds"],
            }
        )
    groups = []
    for category in ("all", "math", "code"):
        keys = [k for k in expected if category == "all" or cases[k[0]]["category"] == category]
        paired_tpot = [
            a[k]["tpot_seconds"] / b[k]["tpot_seconds"]
            for k in keys
            if a[k]["tpot_seconds"] and b[k]["tpot_seconds"]
        ]
        proposed = sum(b[k]["speculative_decoding"]["num_draft_tokens"] for k in keys)
        accepted = sum(b[k]["speculative_decoding"]["num_accepted_draft_tokens"] for k in keys)
        distinct = [
            r
            for r in rows
            if r["repetition"] == 0 and (category == "all" or r["category"] == category)
        ]
        groups.append(
            {
                "category": category,
                "distinct_tasks": len(distinct),
                "requests": len(keys),
                "off_pass_count": sum(r["off"]["passed"] for r in distinct),
                "on_pass_count": sum(r["on"]["passed"] for r in distinct),
                "token_parity_requests": sum(a[k]["token_ids"] == b[k]["token_ids"] for k in keys),
                "off_median_ttft_seconds": statistics.median(a[k]["ttft_seconds"] for k in keys),
                "on_median_ttft_seconds": statistics.median(b[k]["ttft_seconds"] for k in keys),
                "off_median_total_seconds": statistics.median(
                    a[k]["last_token_seconds"] for k in keys
                ),
                "on_median_total_seconds": statistics.median(
                    b[k]["last_token_seconds"] for k in keys
                ),
                "paired_median_decode_speedup": statistics.median(paired_tpot)
                if paired_tpot
                else None,
                "paired_median_total_speedup": statistics.median(
                    a[k]["last_token_seconds"] / b[k]["last_token_seconds"] for k in keys
                ),
                "proposed_tokens": proposed,
                "accepted_tokens": accepted,
                "draft_acceptance": accepted / proposed if proposed else None,
            }
        )
    return {
        "scope": "16 authored tasks; repetitions are timing samples, not independent quality",
        "rows": rows,
        "groups": groups,
    }


def _book_values(values: np.ndarray, contexts: list[dict[str, Any]]) -> np.ndarray:
    books = sorted({c["book_index"] for c in contexts})
    result = np.empty((len(LENGTHS), len(books)))
    for i, length in enumerate(LENGTHS):
        for j, book in enumerate(books):
            indices = [
                k
                for k, c in enumerate(contexts)
                if c["book_index"] == book and c["prefix_tokens"] == length
            ]
            if len(indices) != len(TARGETS) or {
                contexts[k]["source_target_position"] for k in indices
            } != set(TARGETS):
                raise ValueError("Incomplete book/target/length pairing")
            result[i, j] = values[indices].mean()
    return result


def _replay_summary(protocol: dict[str, Any], reports: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for profile in ("graphs", "eager"):
        modes = {}
        conditioned = {}
        for mtp in (0, 2):
            report = reports[f"h64-{profile}-mtp{mtp}-diagnostics"]
            prior_key = "previous_off_tokens" if mtp == 0 else "previous_on_tokens"
            cases = {c["id"]: c for c in protocol["diagnostics"]}
            rows = []
            for request in report["diagnostics"]:
                violations = sum(
                    not _margin(row, token, token)["selected_is_argmax"]
                    for token, row in zip(request["token_ids"], request["logprobs"], strict=True)
                )
                rows.append(
                    {
                        "case_id": request["case_id"],
                        "repetition": request["repetition"],
                        "matches_previous_output": request["token_ids"]
                        == cases[request["case_id"]][prior_key],
                        "greedy_argmax_violations": violations,
                        "traced_tokens": len(request["token_ids"]),
                    }
                )
            modes[f"mtp{mtp}"] = rows
            for request in report["conditioned"]:
                case = cases[request["case_id"]]
                original = case["first_difference"]
                selected = request["token_ids"][0]
                row = request["logprobs"][0]
                item = {
                    "prompt_sha256": request["prompt_sha256"],
                    "selected_token": selected,
                    "probabilities": row,
                    "original_off_token": case["previous_off_tokens"][original],
                    "original_on_token": case["previous_on_tokens"][original],
                    "selected_is_argmax": _margin(row, selected, selected)["selected_is_argmax"],
                }
                conditioned.setdefault(request["case_id"], {})[f"mtp{mtp}"] = item
        for pair in conditioned.values():
            if pair["mtp0"]["prompt_sha256"] != pair["mtp2"]["prompt_sha256"]:
                raise ValueError("Conditioned context identity mismatch")
        result[profile] = {"replay": modes, "conditioned_on_original_common_history": conditioned}
    return result


def compare(args: argparse.Namespace) -> None:
    """Verify all evidence and compare target margins, fixed task scores and paired book KL."""
    protocol, digest = _load(Path(args.protocol))
    root, output = Path(args.output_dir), Path(args.output)
    if output.exists():
        raise FileExistsError("Comparison already exists")
    names = [
        f"h64-{profile}-mtp{mtp}-diagnostics" for profile in ("graphs", "eager") for mtp in (0, 2)
    ] + [f"{label}-graphs-mtp0-long" for label in ("h64", "bf16")]
    reports = {}
    for name in names:
        report = json.loads((root / f"{name}.json").read_text())
        if (
            not report.get("complete")
            or report["protocol_sha256"] != digest
            or report["executed_source_sha256"] != protocol["executed_source_sha256"]
        ):
            raise ValueError("Measurement identity mismatch")
        reports[name] = report
    if len({r["checkpoint_sha256"] for n, r in reports.items() if n.startswith("h64")}) != 1:
        raise ValueError("H64 checkpoint changed between profiles")
    for label in ("bf16", "h64"):
        if (
            _hash_file(root / f"{label}-matched-logprobs.safetensors")
            != (reports[f"{label}-graphs-mtp0-long"]["distributions"]["sha256"])
        ):
            raise ValueError("Distribution hash mismatch")
    kl, top1 = _read_metrics(
        root / "bf16-matched-logprobs.safetensors",
        root / "h64-matched-logprobs.safetensors",
        protocol_sha=digest,
        positions=len(protocol["contexts"]),
    )
    book_kl = _book_values(kl, protocol["contexts"])
    samples = np.random.default_rng(SEED).integers(
        0, len(protocol["books"]), size=(20000, len(protocol["books"]))
    )
    result = {
        "format": "mxwave-mtp-followup-comparison-v1",
        "complete": True,
        "protocol_sha256": digest,
        "executed_source_sha256": protocol["executed_source_sha256"],
        "report_sha256": {n: _hash_file(root / f"{n}.json") for n in names},
        "checkpoint_sha256": {n: r["checkpoint_sha256"] for n, r in reports.items()},
        "replay_and_conditioning": _replay_summary(protocol, reports),
        "diagnostics": {
            p: _diagnostic_compare(
                reports[f"h64-{p}-mtp0-diagnostics"], reports[f"h64-{p}-mtp2-diagnostics"]
            )
            for p in ("graphs", "eager")
        },
        "tasks": _task_compare(
            protocol, reports["h64-graphs-mtp0-diagnostics"], reports["h64-graphs-mtp2-diagnostics"]
        ),
        "long_context": {
            "independent_books": len(protocol["books"]),
            "targets_per_book": len(TARGETS),
            "scope": "same target token, truncated histories; "
            "length effects include lost context and positions remain sparse; no task accuracy",
            "prefixes": [
                {
                    "prefix_tokens": length,
                    "forward_kl": _paired_summary(book_kl[i], samples),
                    "top1_agreement_count": int(
                        top1[
                            [
                                j
                                for j, c in enumerate(protocol["contexts"])
                                if c["prefix_tokens"] == length
                            ]
                        ].sum()
                    ),
                    "mean_kl_per_book": book_kl[i].tolist(),
                }
                for i, length in enumerate(LENGTHS)
            ],
            "longest_minus_shortest": _paired_summary(book_kl[-1] - book_kl[0], samples),
            "kl_per_position": kl.tolist(),
        },
    }
    _write_json(output, result)
    print(
        json.dumps(
            {
                "diagnostics": result["diagnostics"],
                "tasks": result["tasks"]["groups"],
                "long_context": result["long_context"],
            },
            indent=2,
        ),
        flush=True,
    )


def main() -> None:
    """Expose immutable preparation, bounded measurement and CPU comparison phases."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--model", required=True)
    p.add_argument("--previous-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--books", type=int, default=24)
    p = commands.add_parser("collect")
    p.add_argument("--model", required=True)
    p.add_argument("--label", choices=("bf16", "h64"), required=True)
    p.add_argument("--profile", choices=("graphs", "eager"), required=True)
    p.add_argument("--mtp-tokens", type=int, choices=(0, 2), required=True)
    p.add_argument("--phase", choices=("diagnostics", "long"), required=True)
    p.add_argument("--checkpoint-sha256", required=True)
    p.add_argument("--protocol", required=True)
    p.add_argument("--output-dir", required=True)
    p = commands.add_parser("compare")
    p.add_argument("--protocol", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "collect":
        asyncio.run(_collect(args))
    else:
        compare(args)


if __name__ == "__main__":
    main()
