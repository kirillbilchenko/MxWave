"""Freeze and evaluate corrected MXFP4 controls on separate validation windows."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import statistics
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from evaluate_next_token_kl import _extract_full_logprobs, _model_vocab_size
from safetensors import safe_open
from safetensors.torch import save_file

from mxwave.checkpoint import _atomic_save_shard, _atomic_write_text, _file_integrity
from mxwave.core import QUANTIZATION_REVISION

DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
COMPACT_PROTOCOL_FORMAT = "mxwave-corrected-control-protocol-v2"


def _token_digest(tokens: list[int]) -> str:
    return hashlib.sha256(b"".join(struct.pack("<q", value) for value in tokens)).hexdigest()


def _source_digest() -> str:
    root = Path(__file__).resolve().parent.parent
    files = sorted((root / "mxwave").rglob("*.py")) + [
        Path(__file__).resolve(),
        root / "scripts/evaluate_next_token_kl.py",
    ]
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def _write_json(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(path, json.dumps(document, indent=2, sort_keys=True) + "\n")


def _write_protocol(path: Path, document: dict[str, Any]) -> None:
    """Freeze a small manifest and one hashed token vector, storing each window once."""
    token_path = path.with_name("protocol-tokens.safetensors")
    for artifact in (path, path.with_suffix(path.suffix + ".gz"), token_path):
        if artifact.exists():
            raise FileExistsError(f"Refusing to overwrite frozen protocol artifact: {artifact}")
    manifest = {**document, "format": COMPACT_PROTOCOL_FORMAT}
    tokens = []
    for role in ("ppl_windows", "kl_windows"):
        manifest[role] = []
        for window in document[role]:
            values = window["token_ids"]
            manifest[role].append(
                {
                    **{key: value for key, value in window.items() if key != "token_ids"},
                    "token_offset": len(tokens),
                    "token_count": len(values),
                }
            )
            tokens.extend(values)
    manifest["kl_contexts"] = [
        {key: value for key, value in context.items() if key != "token_ids"}
        for context in document["kl_contexts"]
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_save_shard({"token_ids": torch.tensor(tokens, dtype=torch.int64)}, token_path)
    manifest["token_data"] = {
        "filename": token_path.name,
        "sha256": _file_integrity(token_path).sha256,
        "tensor": "token_ids",
        "dtype": "int64",
        "count": len(tokens),
    }
    _write_json(path, manifest)


def _load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    """Load legacy JSON/gzip or compact manifests; hash the uncompressed JSON bytes."""
    raw = path.read_bytes()
    if path.suffix == ".gz":
        raw = gzip.decompress(raw)
    document = json.loads(raw)
    digest = hashlib.sha256(raw).hexdigest()
    format_name = document.get("format", "mxwave-corrected-control-protocol-v1")
    if format_name == "mxwave-corrected-control-protocol-v1":
        return document, digest
    if format_name != COMPACT_PROTOCOL_FORMAT:
        raise ValueError(f"Unsupported protocol format: {format_name}")
    reference = document["token_data"]
    filename = reference["filename"]
    if Path(filename).name != filename or filename in ("", ".", ".."):
        raise ValueError("Protocol token filename must refer to a sibling file")
    token_path = path.parent / filename
    if _file_integrity(token_path).sha256 != reference["sha256"]:
        raise ValueError("Protocol token sidecar hash mismatch")
    with safe_open(str(token_path), framework="pt", device="cpu") as source:
        if reference["tensor"] != "token_ids" or list(source.keys()) != ["token_ids"]:
            raise ValueError("Unexpected protocol token tensor")
        tokens = source.get_tensor("token_ids")
    if (
        reference["dtype"] != "int64"
        or tokens.dtype != torch.int64
        or tokens.ndim != 1
        or tokens.numel() != reference["count"]
        or bool((tokens < 0).any())
    ):
        raise ValueError("Unexpected protocol token shape, dtype or values")
    cursor = 0
    indices: set[int] = set()
    for role in ("ppl_windows", "kl_windows"):
        for window in document[role]:
            count = window["token_count"]
            if (
                window["index"] in indices
                or window["token_offset"] != cursor
                or not isinstance(count, int)
                or count < 2
                or cursor + count > tokens.numel()
            ):
                raise ValueError("Protocol windows overlap or have invalid token ranges")
            indices.add(window["index"])
            window["token_ids"] = tokens[cursor : cursor + count].tolist()
            if _token_digest(window["token_ids"]) != window["token_ids_sha256"]:
                raise ValueError("Protocol window token hash mismatch")
            cursor += count
    if cursor != tokens.numel():
        raise ValueError("Protocol token sidecar has unreferenced tokens")
    kl_windows = {window["index"]: window for window in document["kl_windows"]}
    for context in document["kl_contexts"]:
        window = kl_windows.get(context["window_index"])
        length = context["prefix_tokens"]
        if window is None or not isinstance(length, int) or not 0 < length <= window["token_count"]:
            raise ValueError("Protocol KL prefix has an invalid window or length")
        context["token_ids"] = window["token_ids"][:length]
        if _token_digest(context["token_ids"]) != context["token_ids_sha256"]:
            raise ValueError("Protocol KL prefix token hash mismatch")
    return document, digest


def prepare(args: argparse.Namespace) -> None:
    """Freeze PPL and KL windows before collecting any candidate result."""
    from huggingface_hub import hf_hub_download
    from pyarrow import parquet
    from transformers import AutoTokenizer

    root = Path(args.output_dir)
    if args.ppl_windows <= 0 or args.kl_windows <= 0:
        raise ValueError("Window counts must be positive")
    output = root / "protocol.json"
    for artifact in (output, root / "protocol.json.gz", root / "protocol-tokens.safetensors"):
        if artifact.exists():
            raise FileExistsError(f"Refusing to overwrite frozen protocol artifact: {artifact}")
    filename = "wikitext-2-raw-v1/validation-00000-of-00001.parquet"
    dataset_path = Path(
        hf_hub_download(
            "Salesforce/wikitext",
            filename,
            repo_type="dataset",
            revision=DATASET_REVISION,
            local_dir=str(root / "dataset-source"),
            token=False,
        )
    )
    rows = parquet.read_table(dataset_path, columns=["text"])["text"].to_pylist()
    corpus = "\n\n".join(rows)
    corpus_sha = hashlib.sha256(corpus.encode()).hexdigest()
    old_corpus = Path(args.previous_corpus).read_text()
    if corpus_sha == hashlib.sha256(old_corpus.encode()).hexdigest():
        raise ValueError("Evaluation corpus duplicates the historical test corpus")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    window_chars = 4096
    old_text_hashes: set[str] = set()
    old_token_hashes: set[str] = set()
    for start in range(0, len(old_corpus), window_chars):
        text = old_corpus[start : start + window_chars]
        old_text_hashes.add(hashlib.sha256(text.encode()).hexdigest())
        tokens = tokenizer.encode(text, add_special_tokens=False)
        old_token_hashes.add(_token_digest(tokens))
        if len(tokens) >= 512:
            old_token_hashes.add(_token_digest(tokens[:512]))
    windows = []
    for start in range(0, len(corpus), window_chars):
        text = corpus[start : start + window_chars]
        if not text.strip():
            continue
        tokens = tokenizer.encode(text, add_special_tokens=False)
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest in old_text_hashes or _token_digest(tokens) in old_token_hashes:
            raise ValueError("Validation window overlaps supplied historical evaluation")
        if not 2 <= len(tokens) < 4096:
            raise ValueError("Validation window has an unsupported token length")
        windows.append(
            {
                "index": len(windows),
                "character_start": start,
                "character_end": start + len(text),
                "text_sha256": digest,
                "token_ids_sha256": _token_digest(tokens),
                "token_ids": tokens,
            }
        )
    if len(windows) < args.ppl_windows + args.kl_windows:
        raise ValueError("Validation split has too few windows for the frozen protocol")
    ppl_windows = windows[: args.ppl_windows]
    kl_windows = windows[args.ppl_windows : args.ppl_windows + args.kl_windows]
    prefixes = [64, 128, 192, 256, 320, 384, 448, 512]
    contexts = []
    for window in kl_windows:
        if len(window["token_ids"]) < max(prefixes):
            raise ValueError("KL window is too short for every declared prefix")
        for length in prefixes:
            tokens = window["token_ids"][:length]
            if _token_digest(tokens) in old_token_hashes:
                raise ValueError("KL prefix overlaps supplied historical evaluation")
            contexts.append(
                {
                    "window_index": window["index"],
                    "prefix_tokens": length,
                    "token_ids": tokens,
                    "token_ids_sha256": _token_digest(tokens),
                }
            )
    document = {
        "format": COMPACT_PROTOCOL_FORMAT,
        "quantization_revision": QUANTIZATION_REVISION,
        "executed_source_sha256": _source_digest(),
        "dataset": {
            "repository": "Salesforce/wikitext",
            "configuration": "wikitext-2-raw-v1",
            "revision": DATASET_REVISION,
            "split": "validation",
            "parquet_sha256": _file_integrity(dataset_path).sha256,
            "corpus_sha256": corpus_sha,
            "row_joiner": "\n\n",
        },
        "historical_exclusion": {
            "corpus_sha256": hashlib.sha256(old_corpus.encode()).hexdigest(),
            "scope": "supplied historical WikiText test windows and 512-token prefixes",
            "limitations": "does not certify unknown prior use or semantic duplicates",
        },
        "controls": ["rtn", "mse", "norm", "h64", "bf16", "amd"],
        "conversion": {
            "policy": "qwen3.8-27b-compatible",
            "target_count": 400,
            "scale_percentile": 99.5,
            "mse_clip_depth": 4,
            "h64_calibration": "existing 64x512 Pile validation, absolute damping, no sink skip",
        },
        "runtime_image": args.runtime_image,
        "runtime": {
            "max_model_len": 4096,
            "max_num_seqs": 4,
            "max_num_batched_tokens": 4096,
            "prefix_caching": False,
            "mtp": False,
            "dtype": "bfloat16",
            "kv_cache_dtype": "bfloat16",
            "mxfp4_linear_backend": "marlin",
            "cudagraph_capture_sizes": [1, 2, 4],
        },
        "windowing": {
            "character_window_size": window_chars,
            "min_tokens": 2,
            "max_tokens": 4095,
            "kl_prefix_lengths": prefixes,
            "tokenizer_model": args.model,
            "add_special_tokens": False,
        },
        "ppl_windows": ppl_windows,
        "kl_windows": kl_windows,
        "kl_contexts": contexts,
        "kl_uncertainty_unit": "window; all eight prefixes resampled together",
        "remaining_windows": {
            "count": len(windows) - args.ppl_windows - args.kl_windows,
            "use": "reserved; not scored in this comparison",
        },
        "promotion": "fixed control comparison only; no model promotion or recipe tuning",
    }
    _write_protocol(output, document)
    print(f"Frozen {len(ppl_windows)} PPL windows and {len(contexts)} KL positions", flush=True)


def _prompt_nll(output: Any, tokens: list[int]) -> float:
    logprobs = output.prompt_logprobs
    if not isinstance(logprobs, list) or len(logprobs) != len(tokens):
        raise TypeError("Expected one prompt-logprob entry per frozen token")
    values = []
    for position, token in enumerate(tokens[1:], start=1):
        entry = logprobs[position]
        if entry is None or token not in entry:
            raise ValueError(f"Prompt token {position} has no likelihood")
        value = float(entry[token].logprob)
        if not math.isfinite(value):
            raise ValueError("Prompt likelihood is not finite")
        values.append(value)
    return -math.fsum(values)


def collect(args: argparse.Namespace) -> None:
    """Use one stock vLLM load for paired PPL, exact KL and decode measurements."""
    from vllm import LLM, SamplingParams
    from vllm.logprobs import FlatLogprobs

    protocol_path = Path(args.protocol)
    protocol, protocol_sha = _load_protocol(protocol_path)
    output_dir = Path(args.output_dir)
    report_path = output_dir / f"{args.label}.json"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite measurement: {report_path}")
    if protocol["executed_source_sha256"] != _source_digest():
        raise ValueError("Executed sources differ from the frozen protocol")
    kwargs: dict[str, Any] = {
        "model": args.model,
        "load_format": "safetensors",
        "dtype": "bfloat16",
        "seed": 20261003,
        "max_model_len": 4096,
        "max_num_seqs": 4,
        "max_num_batched_tokens": 4096,
        "enable_chunked_prefill": True,
        "enable_prefix_caching": False,
        "kv_cache_dtype": "bfloat16",
        "gpu_memory_utilization": 0.65 if args.label == "bf16" else 0.45,
        "max_logprobs": -1,
        "disable_log_stats": True,
        "limit_mm_per_prompt": {"image": 0, "video": 0},
        "compilation_config": {"cudagraph_capture_sizes": [1, 2, 4]},
    }
    if args.label != "bf16":
        kwargs["linear_backend"] = "marlin"
    started = time.monotonic()
    llm = LLM(**kwargs)
    load_seconds = time.monotonic() - started
    report: dict[str, Any] = {
        "format": "mxwave-corrected-control-measurement-v1",
        "label": args.label,
        "protocol_sha256": protocol_sha,
        "executed_source_sha256": _source_digest(),
        "checkpoint_sha256": args.checkpoint_sha256,
        "runtime_image": protocol["runtime_image"],
        "engine_kwargs": kwargs,
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "model_load_seconds": load_seconds,
        "ppl_windows": [],
    }
    params = SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=1, detokenize=False)
    for window in protocol["ppl_windows"]:
        tokens = window["token_ids"]
        result = llm.generate({"prompt_token_ids": tokens}, params, use_tqdm=False)
        nll = _prompt_nll(result[0], tokens)
        report["ppl_windows"].append(
            {
                "index": window["index"],
                "text_sha256": window["text_sha256"],
                "token_ids_sha256": window["token_ids_sha256"],
                "scored_tokens": len(tokens) - 1,
                "negative_log_likelihood": nll,
            }
        )
        if len(report["ppl_windows"]) % 16 == 0:
            print(f"{args.label}: PPL {len(report['ppl_windows'])} windows", flush=True)
    total_nll = math.fsum(item["negative_log_likelihood"] for item in report["ppl_windows"])
    total_tokens = sum(item["scored_tokens"] for item in report["ppl_windows"])
    report["ppl"] = {
        "perplexity": math.exp(total_nll / total_tokens),
        "scored_tokens": total_tokens,
        "negative_log_likelihood": total_nll,
    }
    _write_json(output_dir / f"{args.label}-ppl.json", report)
    contexts = protocol["kl_contexts"]
    vocab_size, _ = _model_vocab_size(Path(args.model))
    matrix = torch.empty((len(contexts), vocab_size), dtype=torch.float32)
    params = SamplingParams(
        max_tokens=1,
        temperature=1,
        seed=20261003,
        logprobs=-1,
        flat_logprobs=True,
        detokenize=False,
    )
    for index, context in enumerate(contexts):
        result = llm.generate({"prompt_token_ids": context["token_ids"]}, params, use_tqdm=False)
        probabilities = result[0].outputs[0].logprobs
        if not isinstance(probabilities, FlatLogprobs):
            raise TypeError("Full-vocabulary collection requires vLLM FlatLogprobs")
        matrix[index] = _extract_full_logprobs(probabilities, vocab_size)
        if (index + 1) % 32 == 0:
            print(f"{args.label}: KL {index + 1}/{len(contexts)} positions", flush=True)
    distributions_path = output_dir / f"{args.label}-logprobs.safetensors"
    save_file(
        {"logprobs": matrix},
        str(distributions_path),
        metadata={
            "protocol_sha256": report["protocol_sha256"],
            "label": args.label,
        },
    )
    report["distributions"] = {
        "positions": len(contexts),
        "vocab_size": vocab_size,
        "sha256": _file_integrity(distributions_path).sha256,
    }
    del matrix
    prompt = protocol["ppl_windows"][0]["token_ids"][:512]
    params = SamplingParams(max_tokens=128, temperature=0, ignore_eos=True, detokenize=False)
    timings = []
    for concurrency in (1, 4):
        prompts = [{"prompt_token_ids": prompt}] * concurrency
        llm.generate(prompts, params, use_tqdm=False)
        samples = []
        for _ in range(3):
            started = time.monotonic()
            result = llm.generate(prompts, params, use_tqdm=False)
            elapsed = time.monotonic() - started
            if any(len(item.outputs[0].token_ids) != 128 for item in result):
                raise ValueError("Decode measurement did not emit the fixed token budget")
            samples.append(elapsed)
        timings.append(
            {
                "concurrency": concurrency,
                "seconds": samples,
                "median_output_tokens_per_second": 128 * concurrency / statistics.median(samples),
                "scope": "includes 512-token prefill",
            }
        )
    report["decode"] = timings
    report["complete"] = True
    _write_json(report_path, report)
    print(f"{args.label}: PPL={report['ppl']['perplexity']:.9f}, complete", flush=True)


def compare(args: argparse.Namespace) -> None:
    """Compute paired window-bootstrap intervals from complete frozen measurements."""
    root = Path(args.output_dir)
    if args.bootstrap_iterations <= 0:
        raise ValueError("Bootstrap iterations must be positive")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite comparison: {output}")
    protocol, protocol_sha = _load_protocol(Path(args.protocol))
    reports = {}
    for label in protocol["controls"]:
        report = json.loads((root / f"{label}.json").read_text())
        if not report.get("complete") or report["protocol_sha256"] != protocol_sha:
            raise ValueError(f"Incomplete or mismatched measurement: {label}")
        if report["executed_source_sha256"] != protocol["executed_source_sha256"]:
            raise ValueError(f"Source mismatch: {label}")
        for actual, expected in zip(report["ppl_windows"], protocol["ppl_windows"], strict=True):
            if (
                actual["index"],
                actual["text_sha256"],
                actual["token_ids_sha256"],
                actual["scored_tokens"],
            ) != (
                expected["index"],
                expected["text_sha256"],
                expected["token_ids_sha256"],
                len(expected["token_ids"]) - 1,
            ):
                raise ValueError(f"PPL pairing mismatch: {label}")
        reports[label] = report
    rng = np.random.default_rng(20261003)
    ppl_sample = rng.integers(
        0,
        len(protocol["ppl_windows"]),
        size=(args.bootstrap_iterations, len(protocol["ppl_windows"])),
    )
    kl_indices = sorted({item["window_index"] for item in protocol["kl_contexts"]})
    kl_sample = rng.integers(0, len(kl_indices), size=(args.bootstrap_iterations, len(kl_indices)))
    result: dict[str, Any] = {
        "format": "mxwave-corrected-control-comparison-v1",
        "complete": True,
        "protocol_sha256": protocol_sha,
        "executed_source_sha256": protocol["executed_source_sha256"],
        "quantization_revision": protocol["quantization_revision"],
        "dataset": protocol["dataset"],
        "runtime": protocol["runtime"],
        "runtime_image": protocol["runtime_image"],
        "ppl_window_count": len(protocol["ppl_windows"]),
        "kl_window_count": len(kl_indices),
        "kl_position_count": len(protocol["kl_contexts"]),
        "evaluation_overlap": "PPL and KL use disjoint frozen validation windows",
        "reserved_windows": protocol["remaining_windows"],
        "historical_exclusion": protocol["historical_exclusion"],
        "bootstrap": {
            "iterations": args.bootstrap_iterations,
            "seed": 20261003,
            "unit": "window; all prefixes within a KL window travel together",
        },
        "results": {},
        "paired_comparisons": {},
    }
    kl_by_window = {}
    reference_path = root / "bf16-logprobs.safetensors"
    with safe_open(str(reference_path), framework="pt", device="cpu") as reference_file:
        reference = reference_file.get_slice("logprobs")
        for label, report in reports.items():
            distribution = root / f"{label}-logprobs.safetensors"
            if _file_integrity(distribution).sha256 != report["distributions"]["sha256"]:
                raise ValueError(f"Distribution hash mismatch: {label}")
            rows, agreements = [], []
            with safe_open(str(distribution), framework="pt", device="cpu") as candidate_file:
                candidate = candidate_file.get_slice("logprobs")
                if (
                    candidate.get_shape() != reference.get_shape()
                    or candidate.get_shape()[0] != len(protocol["kl_contexts"])
                    or candidate_file.metadata().get("protocol_sha256") != protocol_sha
                ):
                    raise ValueError(f"Distribution pairing mismatch: {label}")
                for start in range(0, len(protocol["kl_contexts"]), 16):
                    teacher = reference[start : start + 16].double()
                    values = candidate[start : start + 16].double()
                    rows.extend((teacher.exp() * (teacher - values)).sum(dim=1).tolist())
                    agreements.extend((teacher.argmax(dim=1) == values.argmax(dim=1)).tolist())
            kl_by_window[label] = np.array(
                [
                    np.mean(
                        [
                            value
                            for value, context in zip(rows, protocol["kl_contexts"], strict=True)
                            if context["window_index"] == index
                        ]
                    )
                    for index in kl_indices
                ]
            )
            result["results"][label] = {
                "ppl": report["ppl"],
                "mean_forward_kl": float(np.mean(rows)),
                "top1_agreement_count": sum(agreements),
                "decode": report["decode"],
                "checkpoint_sha256": report["checkpoint_sha256"],
                "report_sha256": _file_integrity(root / f"{label}.json").sha256,
                "kl_per_window": kl_by_window[label].tolist(),
            }
    pairs = [(label, "bf16") for label in ("rtn", "mse", "norm", "h64", "amd")]
    pairs += [
        ("mse", "rtn"),
        ("norm", "mse"),
        ("h64", "rtn"),
        ("h64", "mse"),
        ("h64", "norm"),
        ("h64", "amd"),
    ]
    for left, right in pairs:
        counts = np.array([item["scored_tokens"] for item in reports[left]["ppl_windows"]])
        left_nll = np.array(
            [item["negative_log_likelihood"] for item in reports[left]["ppl_windows"]]
        )
        right_nll = np.array(
            [item["negative_log_likelihood"] for item in reports[right]["ppl_windows"]]
        )
        delta = left_nll - right_nll
        samples = np.expm1(delta[ppl_sample].sum(axis=1) / counts[ppl_sample].sum(axis=1)) * 100
        kl_delta = kl_by_window[left] - kl_by_window[right]
        result["paired_comparisons"][f"{left}_vs_{right}"] = {
            "relative_perplexity_percent": math.expm1(delta.sum() / counts.sum()) * 100,
            "relative_perplexity_ci95_percent": np.percentile(samples, [2.5, 97.5]).tolist(),
            "forward_kl_difference": float(kl_delta.mean()),
            "forward_kl_difference_ci95": np.percentile(
                kl_delta[kl_sample].mean(axis=1), [2.5, 97.5]
            ).tolist(),
        }
    _write_json(output, result)
    print(json.dumps(result["results"], indent=2), flush=True)


def main() -> None:
    """Parse the preparation or single-model measurement command."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prep = subparsers.add_parser("prepare")
    prep.add_argument("--model", required=True)
    prep.add_argument("--previous-corpus", required=True)
    prep.add_argument("--output-dir", required=True)
    prep.add_argument("--runtime-image", required=True)
    prep.add_argument("--ppl-windows", type=int, default=128)
    prep.add_argument("--kl-windows", type=int, default=64)
    prep.set_defaults(handler=prepare)
    measure = subparsers.add_parser("collect")
    measure.add_argument("--model", required=True)
    measure.add_argument(
        "--label", choices=("bf16", "amd", "rtn", "mse", "norm", "h64"), required=True
    )
    measure.add_argument("--protocol", required=True)
    measure.add_argument("--output-dir", required=True)
    measure.add_argument("--checkpoint-sha256", required=True)
    measure.set_defaults(handler=collect)
    comparison = subparsers.add_parser("compare")
    comparison.add_argument("--protocol", required=True)
    comparison.add_argument("--output-dir", required=True)
    comparison.add_argument("--output", required=True)
    comparison.add_argument("--bootstrap-iterations", type=int, default=20000)
    comparison.set_defaults(handler=compare)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
