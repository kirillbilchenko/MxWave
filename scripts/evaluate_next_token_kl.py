"""Collect and compare exact next-token distributions with vLLM.

The collector requests the full vocabulary for one generated position per fixed
context.  It stores token-ID-aligned log probabilities in safetensors so that
comparisons do not depend on decoded token strings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _token_digest(token_ids: list[int]) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        digest.update(struct.pack("<q", token_id))
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _nonempty_chunks(text: str, chunk_characters: int) -> list[str]:
    if chunk_characters <= 0:
        raise ValueError("--chunk-characters must be positive")
    return [
        chunk
        for start in range(0, len(text), chunk_characters)
        if (chunk := text[start : start + chunk_characters]).strip()
    ]


def _evenly_spaced_indices(population: int, count: int) -> list[int]:
    if count <= 0:
        raise ValueError("--num-contexts must be positive")
    if count > population:
        raise ValueError(f"Requested {count} contexts from only {population} chunks")
    if count == 1:
        return [0]
    indices = [round(index * (population - 1) / (count - 1)) for index in range(count)]
    if len(set(indices)) != count:
        raise RuntimeError("Evenly spaced context selection produced duplicate indices")
    return indices


def prepare_contexts(args: argparse.Namespace) -> Path:
    """Create one immutable token-ID manifest shared by every model."""
    from transformers import AutoTokenizer

    model_path = Path(args.model)
    corpus_path = Path(args.corpus)
    output_path = Path(args.output)
    corpus_bytes = corpus_path.read_bytes()
    chunks = _nonempty_chunks(corpus_bytes.decode("utf-8"), args.chunk_characters)
    indices = _evenly_spaced_indices(len(chunks), args.num_contexts)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
    )

    contexts: list[dict[str, object]] = []
    for ordinal, chunk_index in enumerate(indices):
        chunk = chunks[chunk_index]
        raw_ids = tokenizer.encode(chunk, add_special_tokens=False)
        token_ids = [int(token_id) for token_id in raw_ids[: args.context_tokens]]
        if not token_ids:
            raise ValueError(f"Selected chunk {chunk_index} tokenized to an empty context")
        contexts.append(
            {
                "ordinal": ordinal,
                "chunk_index": chunk_index,
                "characters": len(chunk),
                "text_sha256": _sha256_bytes(chunk.encode()),
                "token_count": len(token_ids),
                "token_ids_sha256": _token_digest(token_ids),
                "token_ids": token_ids,
            }
        )

    manifest = {
        "format": "mxwave-next-token-contexts-v1",
        "corpus": str(corpus_path),
        "corpus_sha256": _sha256_bytes(corpus_bytes),
        "chunk_characters": args.chunk_characters,
        "available_nonempty_chunks": len(chunks),
        "selection": "evenly-spaced-inclusive-endpoints",
        "num_contexts": len(contexts),
        "context_tokens_max": args.context_tokens,
        "tokenizer_class": tokenizer.__class__.__name__,
        "tokenizer_vocab_size": len(tokenizer),
        "contexts": contexts,
    }
    _write_json_atomic(output_path, manifest)
    print(
        f"prepared {len(contexts)} contexts from {len(chunks)} chunks: {output_path}",
        flush=True,
    )
    return output_path


def _load_context_manifest(path: Path, limit: int | None) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    value = json.loads(raw)
    accepted_formats = {
        "mxwave-next-token-contexts-v1",
        # Frozen evaluation artifacts created before the project rename remain
        # byte-identical inputs and must not be silently rewritten.
        "mxstream-next-token-contexts-v1",
    }
    if not isinstance(value, dict) or value.get("format") not in accepted_formats:
        raise ValueError("Unsupported context manifest")
    contexts = value.get("contexts")
    if not isinstance(contexts, list) or not contexts:
        raise ValueError("Context manifest contains no contexts")
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        if limit > len(contexts):
            raise ValueError("--limit exceeds the context count")
        value = dict(value)
        value["contexts"] = contexts[:limit]
        value["num_contexts"] = limit
    return cast(dict[str, Any], value), _sha256_bytes(raw)


def _model_vocab_size(model_path: Path) -> tuple[int, str]:
    config_path = model_path / "config.json"
    raw = config_path.read_bytes()
    config = json.loads(raw)
    text_config = config.get("text_config")
    source = text_config if isinstance(text_config, dict) else config
    vocab_size = source.get("vocab_size")
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValueError("Model config has no positive vocab_size")
    return vocab_size, _sha256_bytes(raw)


def _extract_full_logprobs(flat: Any, vocab_size: int) -> torch.Tensor:
    if len(flat.start_indices) != 1 or len(flat.end_indices) != 1:
        raise ValueError("Expected exactly one generated-position logprob range")
    start = int(flat.start_indices[0])
    end = int(flat.end_indices[0])
    token_ids = torch.tensor(flat.token_ids[start:end], dtype=torch.int64)
    values = torch.tensor(flat.logprobs[start:end], dtype=torch.float32)
    if token_ids.numel() != values.numel():
        raise ValueError("Full-logprob token-ID and value counts differ")
    if token_ids.numel() not in (vocab_size, vocab_size + 1):
        raise ValueError(
            f"Expected {vocab_size} entries plus at most one sampled-token duplicate, "
            f"received {token_ids.numel()}"
        )
    if token_ids.min().item() != 0 or token_ids.max().item() != vocab_size - 1:
        raise ValueError("Full-logprob token IDs do not span the configured vocabulary")
    counts = torch.bincount(token_ids, minlength=vocab_size)
    if torch.any(counts == 0):
        raise ValueError("Full-logprob token IDs do not cover the complete vocabulary")
    duplicate_ids = torch.nonzero(counts > 1, as_tuple=False).flatten()
    if token_ids.numel() == vocab_size:
        if duplicate_ids.numel() != 0:
            raise ValueError("Full-logprob token IDs contain an unexpected duplicate")
    elif duplicate_ids.numel() != 1 or counts[duplicate_ids[0]].item() != 2:
        raise ValueError("Expected exactly one repeated sampled-token entry")
    for duplicate_id in duplicate_ids.tolist():
        duplicate_values = values[token_ids == duplicate_id]
        if not torch.allclose(duplicate_values, duplicate_values[0], rtol=0.0, atol=1e-7):
            raise ValueError("Repeated sampled-token log probabilities disagree")
    row = torch.empty(vocab_size, dtype=torch.float32)
    row[token_ids] = values
    if not torch.all(torch.isfinite(row)):
        raise ValueError("Full-logprob vector contains non-finite values")
    log_normalizer = float(torch.logsumexp(row.to(torch.float64), dim=0).item())
    if abs(log_normalizer) > 0.02:
        raise ValueError(f"Log probabilities are not normalized: logsumexp={log_normalizer}")
    return row


def collect_distributions(args: argparse.Namespace) -> Path:
    """Load one model and collect exact next-token log probabilities."""
    from vllm import LLM, SamplingParams
    from vllm.logprobs import FlatLogprobs

    model_path = Path(args.model)
    contexts_path = Path(args.contexts)
    output_path = Path(args.output)
    manifest, manifest_sha256 = _load_context_manifest(contexts_path, args.limit)
    contexts = cast(list[dict[str, Any]], manifest["contexts"])
    vocab_size, config_sha256 = _model_vocab_size(model_path)
    max_context = max(int(item["token_count"]) for item in contexts)
    max_model_len = max(args.max_model_len, max_context + 1)
    engine_kwargs: dict[str, object] = {
        "model": str(model_path),
        "load_format": "safetensors",
        "dtype": "bfloat16",
        "seed": args.seed,
        "max_model_len": max_model_len,
        "max_num_seqs": 1,
        "max_logprobs": -1,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": max(1024, max_context + 1),
        "kv_cache_dtype": "bfloat16",
        "disable_log_stats": True,
    }
    if args.linear_backend:
        engine_kwargs["linear_backend"] = args.linear_backend

    print(
        f"loading {args.model_label}: contexts={len(contexts)} vocab={vocab_size} "
        f"max_context={max_context}",
        flush=True,
    )
    started = time.monotonic()
    llm = LLM(**engine_kwargs)
    sampling = SamplingParams(
        max_tokens=1,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        seed=args.seed,
        logprobs=-1,
        flat_logprobs=True,
        detokenize=False,
        skip_special_tokens=False,
    )
    matrix = torch.empty((len(contexts), vocab_size), dtype=torch.float32)
    context_digests: list[str] = []
    per_context_seconds: list[float] = []

    for index, context in enumerate(contexts):
        token_ids = [int(value) for value in context["token_ids"]]
        expected_digest = str(context["token_ids_sha256"])
        if _token_digest(token_ids) != expected_digest:
            raise ValueError(f"Context {index} token digest does not match its manifest")
        item_started = time.monotonic()
        outputs = llm.generate(
            {"prompt_token_ids": token_ids},
            sampling,
            use_tqdm=False,
        )
        if len(outputs) != 1 or len(outputs[0].outputs) != 1:
            raise ValueError(f"Context {index} returned an unexpected output count")
        logprobs = outputs[0].outputs[0].logprobs
        if not isinstance(logprobs, FlatLogprobs):
            raise TypeError(f"Context {index} did not return FlatLogprobs")
        matrix[index] = _extract_full_logprobs(logprobs, vocab_size)
        elapsed = time.monotonic() - item_started
        per_context_seconds.append(elapsed)
        context_digests.append(expected_digest)
        if index == 0 or (index + 1) % 8 == 0 or index + 1 == len(contexts):
            print(
                f"{args.model_label}: context {index + 1}/{len(contexts)} elapsed={elapsed:.3f}s",
                flush=True,
            )

    selected_context_sha256 = _sha256_bytes("".join(context_digests).encode())
    metadata = {
        "format": "mxwave-next-token-logprobs-v1",
        "model_label": args.model_label,
        "model_config_sha256": config_sha256,
        "checkpoint_sha256": args.checkpoint_sha256,
        "contexts_manifest_sha256": manifest_sha256,
        "selected_context_sha256": selected_context_sha256,
        "num_contexts": str(len(contexts)),
        "vocab_size": str(vocab_size),
        "dtype": "float32",
        "seed": str(args.seed),
        "max_model_len": str(max_model_len),
        "gpu_memory_utilization": str(args.gpu_memory_utilization),
        "linear_backend": args.linear_backend or "auto",
        "runtime_image": args.runtime_image,
        "elapsed_seconds": f"{time.monotonic() - started:.6f}",
        "mean_context_seconds": f"{sum(per_context_seconds) / len(per_context_seconds):.6f}",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.incomplete")
    save_file({"logprobs": matrix}, temporary, metadata=metadata)
    temporary.replace(output_path)
    print(
        f"collected {args.model_label}: output={output_path} sha256={_sha256_file(output_path)}",
        flush=True,
    )
    return output_path


def _metadata(path: Path) -> dict[str, str]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return dict(handle.metadata() or {})


def _percentile(values: torch.Tensor, quantile: float) -> float:
    return float(torch.quantile(values, quantile).item())


def _summary(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(tensor.mean().item()),
        "median": _percentile(tensor, 0.5),
        "p95": _percentile(tensor, 0.95),
        "max": float(tensor.max().item()),
    }


def _distribution_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    ref = reference.to(torch.float64)
    cand = candidate.to(torch.float64)
    ref -= torch.logsumexp(ref, dim=0)
    cand -= torch.logsumexp(cand, dim=0)
    probability_ref = torch.exp(ref)
    probability_cand = torch.exp(cand)
    log_mixture = torch.logaddexp(ref, cand) - math.log(2.0)
    forward_kl = torch.sum(probability_ref * (ref - cand))
    reverse_kl = torch.sum(probability_cand * (cand - ref))
    js = 0.5 * (
        torch.sum(probability_ref * (ref - log_mixture))
        + torch.sum(probability_cand * (cand - log_mixture))
    )
    total_variation = 0.5 * torch.sum(torch.abs(probability_ref - probability_cand))
    ref_top5 = torch.topk(ref, k=5).indices
    cand_top5 = torch.topk(cand, k=5).indices
    overlap = len(set(ref_top5.tolist()) & set(cand_top5.tolist()))
    return {
        "forward_kl_nats": max(0.0, float(forward_kl.item())),
        "reverse_kl_nats": max(0.0, float(reverse_kl.item())),
        "jensen_shannon_nats": max(0.0, float(js.item())),
        "total_variation": float(total_variation.item()),
        "top1_agreement": bool(ref_top5[0].item() == cand_top5[0].item()),
        "top5_overlap": overlap,
        "reference_top1_token_id": int(ref_top5[0].item()),
        "candidate_top1_token_id": int(cand_top5[0].item()),
    }


def _bootstrap_mean_ci(values: list[float], seed: int, iterations: int) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=np.float64)
    for start in range(0, iterations, 1000):
        count = min(1000, iterations - start)
        indices = generator.integers(0, len(array), size=(count, len(array)))
        means[start : start + count] = array[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return [float(lower), float(upper)]


def compare_distributions(args: argparse.Namespace) -> Path:
    """Compare candidate distributions against a BF16 reference."""
    reference_path = Path(args.reference)
    reference_metadata = _metadata(reference_path)
    reference = load_file(reference_path)["logprobs"]
    context_protocol: dict[str, Any] | None = None
    if args.contexts:
        contexts_path = Path(args.contexts)
        contexts_raw = contexts_path.read_bytes()
        contexts_value = json.loads(contexts_raw)
        if not isinstance(contexts_value, dict):
            raise ValueError("Context manifest is not a JSON object")
        contexts_sha256 = _sha256_bytes(contexts_raw)
        if contexts_sha256 != reference_metadata.get("contexts_manifest_sha256"):
            raise ValueError("Context manifest hash differs from the reference metadata")
        raw_contexts = contexts_value.get("contexts")
        if not isinstance(raw_contexts, list) or len(raw_contexts) != reference.shape[0]:
            raise ValueError("Context manifest count differs from the reference tensor")
        if not all(isinstance(item, dict) for item in raw_contexts):
            raise ValueError("Context manifest contains a non-object context")
        context_protocol = {
            key: value for key, value in contexts_value.items() if key != "contexts"
        }
        context_protocol["manifest_sha256"] = contexts_sha256
        context_protocol["contexts"] = [
            {key: value for key, value in item.items() if key != "token_ids"}
            for item in raw_contexts
        ]
    candidates: list[tuple[str, Path]] = []
    for item in args.candidate:
        label, separator, raw_path = item.partition("=")
        if not separator or not label or not raw_path:
            raise ValueError("--candidate must use LABEL=PATH")
        candidates.append((label, Path(raw_path)))

    report_candidates: dict[str, Any] = {}
    per_candidate_forward: dict[str, list[float]] = {}
    for label, path in candidates:
        metadata = _metadata(path)
        candidate = load_file(path)["logprobs"]
        if candidate.shape != reference.shape:
            raise ValueError(f"{label}: tensor shape differs from the reference")
        for key in ("selected_context_sha256", "num_contexts", "vocab_size"):
            if metadata.get(key) != reference_metadata.get(key):
                raise ValueError(f"{label}: metadata field {key} differs from the reference")

        rows: list[dict[str, Any]] = []
        for index in range(reference.shape[0]):
            row = _distribution_metrics(reference[index], candidate[index])
            row["context_ordinal"] = index
            rows.append(row)
        forward = [float(row["forward_kl_nats"]) for row in rows]
        reverse = [float(row["reverse_kl_nats"]) for row in rows]
        js = [float(row["jensen_shannon_nats"]) for row in rows]
        tv = [float(row["total_variation"]) for row in rows]
        per_candidate_forward[label] = forward
        report_candidates[label] = {
            "artifact": str(path),
            "artifact_sha256": _sha256_file(path),
            "metadata": metadata,
            "forward_kl_nats": _summary(forward),
            "reverse_kl_nats": _summary(reverse),
            "jensen_shannon_nats": _summary(js),
            "total_variation": _summary(tv),
            "top1_agreement_rate": sum(bool(row["top1_agreement"]) for row in rows) / len(rows),
            "mean_top5_overlap": sum(int(row["top5_overlap"]) for row in rows) / len(rows),
            "contexts": rows,
        }

    paired: dict[str, Any] = {}
    labels = list(per_candidate_forward)
    for left_index, left in enumerate(labels):
        for right in labels[left_index + 1 :]:
            differences = [
                left_value - right_value
                for left_value, right_value in zip(
                    per_candidate_forward[left], per_candidate_forward[right], strict=True
                )
            ]
            paired[f"{left}_minus_{right}"] = {
                "mean_forward_kl_difference_nats": sum(differences) / len(differences),
                "bootstrap_95_ci": _bootstrap_mean_ci(
                    differences, args.bootstrap_seed, args.bootstrap_iterations
                ),
                "left_lower_contexts": sum(value < 0 for value in differences),
                "ties": sum(value == 0 for value in differences),
                "right_lower_contexts": sum(value > 0 for value in differences),
            }

    metric_definition = {
        "forward_kl_nats": "D_KL(P_BF16 || P_candidate)",
        "reverse_kl_nats": "D_KL(P_candidate || P_BF16)",
        "jensen_shannon_nats": "symmetric bounded divergence using the 50/50 mixture",
        "total_variation": "0.5 * sum(abs(P_BF16 - P_candidate))",
    }
    protocol_identity = {
        "reference_artifact_sha256": _sha256_file(reference_path),
        "reference_selected_context_sha256": reference_metadata.get(
            "selected_context_sha256"
        ),
        "context_manifest_sha256": reference_metadata.get("contexts_manifest_sha256"),
        "context_protocol_manifest_sha256": (
            context_protocol.get("manifest_sha256")
            if isinstance(context_protocol, dict)
            else None
        ),
        "num_contexts": reference_metadata.get("num_contexts"),
        "vocab_size": reference_metadata.get("vocab_size"),
        "metric_definition": metric_definition,
        "bootstrap_iterations": args.bootstrap_iterations,
        "bootstrap_seed": args.bootstrap_seed,
    }
    report = {
        "format": "mxwave-next-token-divergence-v1",
        "protocol_sha256": _sha256_bytes(
            json.dumps(protocol_identity, sort_keys=True, separators=(",", ":")).encode()
        ),
        "reference": {
            "artifact": str(reference_path),
            "artifact_sha256": _sha256_file(reference_path),
            "metadata": reference_metadata,
        },
        "context_protocol": context_protocol,
        "metric_definition": metric_definition,
        "candidates": report_candidates,
        "paired_candidate_comparisons": paired,
        "bootstrap": {
            "iterations": args.bootstrap_iterations,
            "seed": args.bootstrap_seed,
            "unit": "context",
        },
    }
    output_path = Path(args.output)
    _write_json_atomic(output_path, report)
    print(f"comparison report: {output_path}", flush=True)
    for label, candidate in report_candidates.items():
        mean_kl = candidate["forward_kl_nats"]["mean"]
        agreement = candidate["top1_agreement_rate"]
        print(f"{label}: mean_forward_kl={mean_kl:.9f} top1={agreement:.3%}", flush=True)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="prepare fixed token-ID contexts")
    prepare.add_argument("--model", required=True)
    prepare.add_argument("--corpus", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--num-contexts", type=int, default=128)
    prepare.add_argument("--context-tokens", type=int, default=512)
    prepare.add_argument("--chunk-characters", type=int, default=4096)
    prepare.set_defaults(handler=prepare_contexts)

    collect = subparsers.add_parser("collect", help="collect full next-token distributions")
    collect.add_argument("--model", required=True)
    collect.add_argument("--model-label", required=True)
    collect.add_argument("--contexts", required=True)
    collect.add_argument("--output", required=True)
    collect.add_argument("--runtime-image", required=True)
    collect.add_argument("--checkpoint-sha256", default="")
    collect.add_argument("--linear-backend", default="")
    collect.add_argument("--limit", type=int)
    collect.add_argument("--seed", type=int, default=20260916)
    collect.add_argument("--max-model-len", type=int, default=1024)
    collect.add_argument("--gpu-memory-utilization", type=float, default=0.65)
    collect.set_defaults(handler=collect_distributions)

    compare = subparsers.add_parser("compare", help="compare distributions with BF16")
    compare.add_argument("--reference", required=True)
    compare.add_argument("--contexts")
    compare.add_argument("--candidate", action="append", required=True)
    compare.add_argument("--output", required=True)
    compare.add_argument("--bootstrap-iterations", type=int, default=10000)
    compare.add_argument("--bootstrap-seed", type=int, default=20260916)
    compare.set_defaults(handler=compare_distributions)
    return parser


def main() -> int:
    """Run the selected command and return a process exit code."""
    try:
        args = build_parser().parse_args()
        handler = cast(Any, args.handler)
        handler(args)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"evaluation error: {exc}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
