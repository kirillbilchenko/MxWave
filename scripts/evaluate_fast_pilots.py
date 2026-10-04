"""Freeze and measure a small MTP and long-context development pilot on Spark."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import itertools
import json
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import torch
from analyze_kl_prefixes import _hash_file, _paired_summary, _prefix_rows, _read_metrics
from evaluate_next_token_kl import _extract_full_logprobs, _model_vocab_size, _token_digest
from evaluate_quantization_controls import _load_protocol, _write_json, _write_protocol
from safetensors.torch import save_file

IMAGE = "sha256:785fd756b4ae1cacda7612cb859dbf227330e8ced9a6fdea42e9379128bebce0"
PG19_REVISION = "4d28bd77e66947ad3835cf78ed7aaeb4dd87ad8b"
PREFIXES = (512, 2048, 8192)
SEED = 20261004


def _source_digest() -> str:
    digest = hashlib.sha256()
    for name in (
        "evaluate_fast_pilots.py",
        "analyze_kl_prefixes.py",
        "evaluate_next_token_kl.py",
        "evaluate_quantization_controls.py",
    ):
        path = Path(__file__).with_name(name)
        digest.update(name.encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def _download(url: str, path: Path) -> bytes:
    if not path.exists():
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return path.read_bytes()


def _serving_cases() -> list[dict[str, Any]]:
    prompts = {
        "chat": [
            (
                "Explain how a home backup plan can recover from a lost laptop. Give a concrete "
                "plan, verification steps, and common mistakes in about 300 words."
            ),
            (
                "A small web service has slow requests after every deployment. Explain how to "
                "investigate cache warmup, database connections, and startup work in about 300 words."
            ),
            (
                "Explain to a new engineer how timeouts, retries, and idempotency interact. "
                "Use one concrete example and discuss failure cases in about 300 words."
            ),
        ],
        "math": [
            (
                "A shop raises a price by 25 percent and then discounts the new price by 20 percent. "
                "The original price is 80 dollars. Derive each price, the overall change, and why "
                "adding the percentages would be wrong. Explain step by step."
            ),
            (
                "A tank is filled by one pipe in 6 hours and another in 9 hours. A drain empties "
                "a full tank in 18 hours. Starting empty, all three are open. Derive the filling "
                "time, check your answer, and explain the assumptions step by step."
            ),
            (
                "A train travels 120 km at 60 km per hour, then 180 km at 90 km per hour. Derive "
                "the average speed. Explain why averaging the two speeds is wrong and show a "
                "second way to check the answer."
            ),
        ],
        "code": [
            (
                "Write Python merge_sorted(left, right) for two sorted integer lists. Return "
                "a new sorted list in linear time, preserve duplicates, and include a docstring "
                "and three short examples. Return code only."
            ),
            (
                "Write Python group_intervals(intervals) merging overlapping closed intervals. "
                "Handle empty input and touching boundaries. Include a docstring and three "
                "assert examples. Return code only."
            ),
            (
                "Write Python dedupe_events(events) for dictionaries with id and timestamp. "
                "Keep the newest event per id, break equal timestamps by keeping the first, "
                "and return the result sorted by id. Include a docstring and examples. Code only."
            ),
        ],
        "prose": [
            (
                "Write a 300-word fictional scene about a librarian discovering a handwritten "
                "map in a returned book. Use a calm tone, concrete details, and no supernatural events."
            ),
            (
                "Write a 300-word explanation of how a city can reduce summer heat using shade, "
                "trees, reflective roofs, and water. Explain the tradeoffs in plain language."
            ),
            (
                "Write a 300-word guide to reading a scientific experiment critically. Cover "
                "controls, uncertainty, selection bias, reproducibility, and limited conclusions."
            ),
        ],
    }
    return [
        {"id": f"{category}-{index}", "category": category, "content": content}
        for category, values in prompts.items()
        for index, content in enumerate(values)
    ]


def prepare(args: argparse.Namespace) -> None:
    """Freeze public book prefixes and chat-template prompts before inference."""
    from transformers import AutoTokenizer

    root = Path(args.output_dir)
    if (root / "protocol.json").exists() or (root / "protocol-tokens.safetensors").exists():
        raise FileExistsError("Pilot protocol already frozen")
    if not 8 <= args.books <= 16:
        raise ValueError("The development pilot requires 8 to 16 independent books")
    old, old_sha = _load_protocol(Path(args.previous_protocol))
    old_hashes = {
        item["token_ids_sha256"]
        for role in ("ppl_windows", "kl_windows", "kl_contexts")
        for item in old[role]
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    url = (
        "https://huggingface.co/datasets/deepmind/pg19/resolve/"
        f"{PG19_REVISION}/data/validation_files.txt"
    )
    source_list = _download(url, root / "dataset-source/validation_files.txt")
    windows, contexts, skipped = [], [], []
    for filename in source_list.decode().splitlines():
        if not filename.startswith("validation/") or ".." in filename:
            raise ValueError("Unexpected PG-19 manifest entry")
        book_url = f"https://storage.googleapis.com/deepmind-gutenberg/{filename}"
        data = _download(book_url, root / "dataset-source" / filename)
        text = data.decode("utf-8")
        raw = tokenizer.encode(text[:200000], add_special_tokens=False)
        if len(raw) < 1024 + max(PREFIXES):
            skipped.append(filename)
            continue
        tokens = raw[1024 : 1024 + max(PREFIXES)]
        index = len(windows)
        window = {
            "index": index,
            "book": filename,
            "source_url": book_url,
            "source_sha256": hashlib.sha256(data).hexdigest(),
            "source_bytes": len(data),
            "character_start": 0,
            "character_end": min(200000, len(text)),
            "token_start": 1024,
            "token_ids": tokens,
            "token_ids_sha256": _token_digest(tokens),
        }
        windows.append(window)
        for length in PREFIXES:
            values = tokens[:length]
            digest = _token_digest(values)
            if digest in old_hashes:
                raise ValueError("Pilot prefix overlaps earlier control inputs")
            contexts.append(
                {
                    "window_index": index,
                    "prefix_tokens": length,
                    "token_ids": values,
                    "token_ids_sha256": digest,
                }
            )
        if len(windows) == args.books:
            break
    if len(windows) != args.books:
        raise ValueError("Too few eligible PG-19 books")
    cases = _serving_cases()
    for case in cases:
        messages = [{"role": "user", "content": case.pop("content")}]
        case["messages"] = messages
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        case["token_ids"] = tokenizer.encode(rendered, add_special_tokens=False)
        case["token_ids_sha256"] = _token_digest(case["token_ids"])
    document = {
        "format": "mxwave-corrected-control-protocol-v2",
        "study": "fast-development-pilots-2026-10-04",
        "executed_source_sha256": _source_digest(),
        "runtime_image": IMAGE,
        "dataset": {
            "repository": "deepmind/pg19",
            "revision": PG19_REVISION,
            "split": "validation",
            "manifest_sha256": hashlib.sha256(source_list).hexdigest(),
            "selection": "first eligible books in pinned manifest; skip first 1024 tokens",
            "skipped_short_books": skipped,
        },
        "previous_protocol_sha256": old_sha,
        "reserved_validation": "WikiText final reserved windows untouched; different corpus",
        "ppl_windows": [],
        "kl_windows": windows,
        "kl_contexts": contexts,
        "serving_cases": cases,
        "serving": {
            "repetitions": 3,
            "max_tokens": 192,
            "temperature": 0,
            "seed": SEED,
            "ignore_eos": False,
            "enable_thinking": False,
            "concurrency": 1,
            "mtp_tokens": [0, 2],
            "warmup": "one 32-token completion per category",
        },
        "scope": "exploratory screens only; no final quality or promotion claim",
    }
    _write_protocol(root / "protocol.json", document)
    print(f"FROZEN {len(windows)} books, {len(contexts)} KL positions, {len(cases)} prompts")


def _engine_kwargs(model: str, label: str, mtp: int) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "load_format": "safetensors",
        "dtype": "bfloat16",
        "seed": SEED,
        "max_model_len": 8448,
        "max_num_seqs": 1,
        "max_num_batched_tokens": 4096,
        "enable_chunked_prefill": True,
        "enable_prefix_caching": False,
        "kv_cache_dtype": "bfloat16",
        "max_logprobs": -1,
        "gpu_memory_utilization": 0.65 if label == "bf16" else 0.45,
        "limit_mm_per_prompt": {"image": 0, "video": 0},
        "compilation_config": {"cudagraph_capture_sizes": [1, 2, 4]},
        "disable_log_stats": False,
    }
    if label != "bf16":
        kwargs["linear_backend"] = "marlin"
    if mtp:
        kwargs["speculative_config"] = {"method": "mtp", "num_speculative_tokens": mtp}
        kwargs["per_request_spec_decode_metrics"] = "summary"
    return kwargs


def _stream_summary(events: list[list[float]], tokens: list[int]) -> dict[str, Any]:
    if len(tokens) < 2 or not events:
        raise ValueError("Latency pilot requires at least two generated tokens")
    counts = [int(event[1]) for event in events]
    times = [event[0] for event in events]
    if (
        counts[-1] != len(tokens)
        or any(a >= b for a, b in itertools.pairwise(counts))
        or any(a > b for a, b in itertools.pairwise(times))
        or times[0] <= 0
        or times[-1] <= times[0]
    ):
        raise ValueError("Invalid cumulative streaming token/time sequence")
    return {
        "output_tokens": len(tokens),
        "ttft_seconds": times[0],
        "last_token_seconds": times[-1],
        "tpot_seconds": (times[-1] - times[0]) / (len(tokens) - counts[0]),
        "output_tokens_per_second": len(tokens) / times[-1],
        "stream_events": events,
    }


async def _generate(engine: Any, tokens: list[int], params: Any, request: str) -> Any:
    final = None
    async for result in engine.generate({"prompt_token_ids": tokens}, params, request):
        final = result
    if final is None or not final.finished or len(final.outputs) != 1:
        raise ValueError("Incomplete generation")
    return final.outputs[0]


async def _collect_async(args: argparse.Namespace) -> None:
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.logprobs import FlatLogprobs
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.metrics.reader import get_metrics_snapshot

    protocol, digest = _load_protocol(Path(args.protocol))
    root = Path(args.output_dir)
    name = f"{args.label}-mtp{args.mtp_tokens}"
    output = root / f"{name}.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite pilot: {output}")
    if protocol["executed_source_sha256"] != _source_digest():
        raise ValueError("Collector sources differ from frozen pilot")
    for case in protocol["serving_cases"]:
        if _token_digest(case["token_ids"]) != case["token_ids_sha256"]:
            raise ValueError("Frozen serving token hash mismatch")
    kwargs = _engine_kwargs(args.model, args.label, args.mtp_tokens)
    started = time.monotonic()
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**kwargs))
    report: dict[str, Any] = {
        "format": "mxwave-fast-pilot-measurement-v1",
        "label": args.label,
        "mtp_tokens": args.mtp_tokens,
        "protocol_sha256": digest,
        "executed_source_sha256": _source_digest(),
        "checkpoint_sha256": args.checkpoint_sha256,
        "runtime_image": IMAGE,
        "engine_kwargs": kwargs,
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "model_load_seconds": time.monotonic() - started,
        "serving": [],
    }
    try:
        if args.long_context:
            if args.mtp_tokens:
                raise ValueError("Quality collection requires MTP disabled")
            vocab, _ = _model_vocab_size(Path(args.model))
            matrix = torch.empty((len(protocol["kl_contexts"]), vocab), dtype=torch.float32)
            params = SamplingParams(
                max_tokens=1,
                temperature=1,
                seed=SEED,
                logprobs=-1,
                flat_logprobs=True,
                detokenize=False,
            )
            for index, context in enumerate(protocol["kl_contexts"]):
                result = await _generate(engine, context["token_ids"], params, f"kl-{index}")
                if not isinstance(result.logprobs, FlatLogprobs):
                    raise TypeError("Full-vocabulary collection requires FlatLogprobs")
                matrix[index] = _extract_full_logprobs(result.logprobs, vocab)
                print(f"{name}: KL {index + 1}/{len(matrix)}", flush=True)
            path = root / f"{args.label}-long-logprobs.safetensors"
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite distributions: {path}")
            save_file({"logprobs": matrix}, str(path), metadata={"protocol_sha256": digest})
            report["distributions"] = {"sha256": _hash_file(path), "positions": len(matrix)}
            del matrix
            _write_json(root / f"{name}-quality.json", report)
        if args.serving:
            settings = protocol["serving"]
            for index in (0, 3, 6, 9):
                await _generate(
                    engine,
                    protocol["serving_cases"][index]["token_ids"],
                    SamplingParams(max_tokens=32, temperature=0),
                    f"warmup-{index}",
                )
            for repetition in range(settings["repetitions"]):
                for case in protocol["serving_cases"]:
                    params = SamplingParams(
                        max_tokens=settings["max_tokens"],
                        temperature=0,
                        seed=SEED,
                        ignore_eos=False,
                    )
                    events, final = [], None
                    before = time.monotonic()
                    async for result in engine.generate(
                        {"prompt_token_ids": case["token_ids"]},
                        params,
                        f"{case['id']}-{repetition}",
                    ):
                        final = result
                        count = len(result.outputs[0].token_ids) if result.outputs else 0
                        if count and (not events or count > events[-1][1]):
                            events.append([time.monotonic() - before, count])
                    if final is None or not final.finished:
                        raise ValueError("Incomplete streaming request")
                    completion = final.outputs[0]
                    tokens = list(completion.token_ids)
                    spec = completion.spec_decode_metrics
                    acceptance = spec.to_dict() if spec is not None else None
                    if args.mtp_tokens and (
                        acceptance is None or acceptance["num_draft_tokens"] <= 0
                    ):
                        raise ValueError("MTP enabled but backend reports no proposed tokens")
                    report["serving"].append(
                        {
                            "case_id": case["id"],
                            "category": case["category"],
                            "repetition": repetition,
                            "prompt_sha256": case["token_ids_sha256"],
                            "prompt_tokens": len(case["token_ids"]),
                            "token_ids": tokens,
                            "token_ids_sha256": _token_digest(tokens),
                            "finish_reason": completion.finish_reason,
                            "speculative_decoding": acceptance,
                            **_stream_summary(events, tokens),
                        }
                    )
                    _write_json(root / f"{name}-progress.json", report)
                    print(f"{name}: serving {len(report['serving'])}/36", flush=True)
            report["backend_spec_counters"] = [
                {
                    "name": metric.name,
                    "value": getattr(metric, "value", None),
                    "values": getattr(metric, "values", None),
                }
                for metric in get_metrics_snapshot()
                if "spec_decode" in metric.name
            ]
        report["complete"] = True
        _write_json(output, report)
    finally:
        engine.shutdown()


def collect(args: argparse.Namespace) -> None:
    """Use the stock async runtime for exact distributions and colocated streaming times."""
    asyncio.run(_collect_async(args))


def _serving_comparison(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    a = {(item["case_id"], item["repetition"]): item for item in left["serving"]}
    b = {(item["case_id"], item["repetition"]): item for item in right["serving"]}
    if not a or a.keys() != b.keys():
        raise ValueError("Serving case/repetition pairing mismatch")
    for key in a:
        if a[key]["prompt_sha256"] != b[key]["prompt_sha256"]:
            raise ValueError("Serving prompt identity mismatch")
    rows = []
    for category in ("all", "chat", "math", "code", "prose"):
        keys = [key for key in a if category == "all" or a[key]["category"] == category]
        if not keys:
            continue
        off, on = [a[key] for key in keys], [b[key] for key in keys]
        proposed = sum(item["speculative_decoding"]["num_draft_tokens"] for item in on)
        accepted = sum(item["speculative_decoding"]["num_accepted_draft_tokens"] for item in on)
        steps = sum(item["speculative_decoding"]["num_spec_steps"] for item in on)
        rows.append(
            {
                "category": category,
                "requests": len(keys),
                "token_parity_count": sum(
                    a[key]["token_ids"] == b[key]["token_ids"] for key in keys
                ),
                "off_median_ttft_seconds": statistics.median(item["ttft_seconds"] for item in off),
                "on_median_ttft_seconds": statistics.median(item["ttft_seconds"] for item in on),
                "off_median_tpot_seconds": statistics.median(item["tpot_seconds"] for item in off),
                "on_median_tpot_seconds": statistics.median(item["tpot_seconds"] for item in on),
                "paired_median_decode_speedup": statistics.median(
                    a[key]["tpot_seconds"] / b[key]["tpot_seconds"] for key in keys
                ),
                "off_output_tokens_per_second": sum(item["output_tokens"] for item in off)
                / sum(item["last_token_seconds"] for item in off),
                "on_output_tokens_per_second": sum(item["output_tokens"] for item in on)
                / sum(item["last_token_seconds"] for item in on),
                "proposed_tokens": proposed,
                "accepted_tokens": accepted,
                "draft_acceptance_rate": accepted / proposed if proposed else None,
                "mean_acceptance_length": 1 + accepted / steps if steps else None,
            }
        )
    return {
        "groups": rows,
        "parity": "exact greedy token equality; any mismatch blocks deployment",
        "timing_scope": "colocated AsyncLLM stream, includes prefill, no network/proxy latency",
    }


def compare(args: argparse.Namespace) -> None:
    """Report paired book-prefix uncertainty and per-domain MTP latency/acceptance."""
    protocol, digest = _load_protocol(Path(args.protocol))
    root, output = Path(args.output_dir), Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite comparison: {output}")
    reports = {}
    for name in ("bf16-mtp0", "h64-mtp0", "h64-mtp2"):
        path = root / f"{name}.json"
        report = json.loads(path.read_text())
        if (
            not report.get("complete")
            or report["protocol_sha256"] != digest
            or report["executed_source_sha256"] != protocol["executed_source_sha256"]
        ):
            raise ValueError(f"Pilot report identity mismatch: {name}")
        reports[name] = report
    if reports["h64-mtp0"]["checkpoint_sha256"] != reports["h64-mtp2"]["checkpoint_sha256"]:
        raise ValueError("MTP measurements used different checkpoints")
    for label in ("bf16", "h64"):
        if (
            _hash_file(root / f"{label}-long-logprobs.safetensors")
            != (reports[f"{label}-mtp0"]["distributions"]["sha256"])
        ):
            raise ValueError("Long-context distribution hash mismatch")
    windows, prefixes, indices = _prefix_rows(protocol["kl_contexts"])
    kl, top1 = _read_metrics(
        root / "bf16-long-logprobs.safetensors",
        root / "h64-long-logprobs.safetensors",
        protocol_sha=digest,
        positions=len(protocol["kl_contexts"]),
    )
    samples = np.random.default_rng(SEED).integers(0, len(windows), size=(20000, len(windows)))
    result = {
        "format": "mxwave-fast-pilot-comparison-v1",
        "complete": True,
        "protocol_sha256": digest,
        "executed_source_sha256": protocol["executed_source_sha256"],
        "report_sha256": {name: _hash_file(root / f"{name}.json") for name in reports},
        "checkpoint_sha256": {name: item["checkpoint_sha256"] for name, item in reports.items()},
        "scope": "small exploratory development screen; no model promotion or accuracy conclusion",
        "long_context": {
            "dataset": protocol["dataset"],
            "uncertainty_unit": "book, prefixes travel together",
            "intervals": "nominal 95%; unadjusted; one next-token position per book/prefix",
            "prefixes": [
                {
                    "prefix_tokens": prefix,
                    "forward_kl": _paired_summary(kl[rows], samples),
                    "top1_agreement_count": int(top1[rows].sum()),
                    "kl_per_book": kl[rows].tolist(),
                }
                for prefix, rows in zip(prefixes, indices, strict=True)
            ],
            "longest_minus_shortest_kl": _paired_summary(kl[indices[-1]] - kl[indices[0]], samples),
        },
        "mtp": _serving_comparison(reports["h64-mtp0"], reports["h64-mtp2"]),
    }
    _write_json(output, result)
    print(json.dumps(result, indent=2))


def main() -> None:
    """Expose immutable preparation, serial collection, and paired comparison phases."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--model", required=True)
    preparation.add_argument("--previous-protocol", required=True)
    preparation.add_argument("--output-dir", required=True)
    preparation.add_argument("--books", type=int, default=12)
    collection = commands.add_parser("collect")
    collection.add_argument("--model", required=True)
    collection.add_argument("--label", choices=("bf16", "h64"), required=True)
    collection.add_argument("--mtp-tokens", type=int, choices=(0, 2), required=True)
    collection.add_argument("--checkpoint-sha256", required=True)
    collection.add_argument("--protocol", required=True)
    collection.add_argument("--output-dir", required=True)
    collection.add_argument("--long-context", action="store_true")
    collection.add_argument("--serving", action="store_true")
    comparison = commands.add_parser("compare")
    comparison.add_argument("--protocol", required=True)
    comparison.add_argument("--output-dir", required=True)
    comparison.add_argument("--output", required=True)
    args = parser.parse_args()
    {"prepare": prepare, "collect": collect, "compare": compare}[args.command](args)


if __name__ == "__main__":
    main()
