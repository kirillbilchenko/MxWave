"""Command-line activation calibration for MxWave quantization."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import resource
import sys
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

import torch

from .calibration import (
    ActivationCollector,
    CalibrationObjective,
    attach_activation_hooks,
    save_calibration_data,
)
from .calibration_stream import (
    calibrate_decoder_sequentially,
    infer_sequential_decoder_layout,
)
from .engine import ModelPlan, QuantizeConfig, plan_model

_OBJECTIVE_CHOICES = ("mean-abs", "rms", "block-hessian")


def build_parser() -> argparse.ArgumentParser:
    """Build the ``mxwave-calibrate`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="mxwave-calibrate",
        description="Capture real per-module input statistics from a float checkpoint.",
    )
    parser.add_argument("--model-dir", required=True, help="BF16/FP16 source model")
    parser.add_argument("--corpus", required=True, help="UTF-8 text, JSON, or JSONL corpus")
    parser.add_argument("--output", required=True, help="Output calibration .safetensors")
    parser.add_argument(
        "--policy",
        choices=("auto", "qwen3.8-27b-mlp", "qwen3.8-27b-compatible", "all-linear"),
        default="auto",
        help="Must exactly match the later quantization policy",
    )
    parser.add_argument(
        "--statistics",
        default="mean-abs,rms",
        help="Comma-separated: mean-abs,rms,block-hessian",
    )
    parser.add_argument("--num-sequences", type=int, default=16)
    parser.add_argument(
        "--sequence-offset",
        type=int,
        default=0,
        help="Skip this many deterministically packed sequences before calibration",
    )
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--weight-loading",
        choices=("streaming", "resident"),
        default="streaming",
        help=(
            "stream one decoder layer at a time (default), or explicitly load the full "
            "model with resident"
        ),
    )
    parser.add_argument("--text-field", default="text", help="Text key for JSON/JSONL rows")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--attention-implementation",
        choices=("eager", "sdpa"),
        default="sdpa",
    )
    parser.add_argument("--hessian-damp", type=float, default=1e-6)
    parser.add_argument("--source-repository", default=None)
    parser.add_argument("--source-revision", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate policy, corpus, tokenizer, and module hooks without loading weights",
    )
    return parser


def _parse_objectives(value: str) -> tuple[CalibrationObjective, ...]:
    raw_items = [item.strip() for item in value.split(",") if item.strip()]
    if not raw_items:
        raise ValueError("--statistics must contain at least one objective")
    invalid = sorted(set(raw_items).difference(_OBJECTIVE_CHOICES))
    if invalid:
        raise ValueError(f"Unsupported calibration objectives: {invalid}")
    return tuple(cast(CalibrationObjective, item) for item in dict.fromkeys(raw_items))


def _extract_json_texts(value: object, text_field: str) -> list[str]:
    if isinstance(value, list):
        rows: Iterable[object] = value
    elif isinstance(value, dict) and isinstance(value.get("data"), list):
        rows = cast(list[object], value["data"])
    elif isinstance(value, dict) and isinstance(value.get("rows"), list):
        rows = cast(list[object], value["rows"])
    else:
        rows = [value]

    texts: list[str] = []
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("row"), dict):
            row = row["row"]
        if isinstance(row, str):
            text = row
        elif isinstance(row, dict):
            raw_text = row.get(text_field)
            if not isinstance(raw_text, str):
                raise TypeError(f"JSON row is missing string field {text_field!r}")
            text = raw_text
        else:
            raise TypeError("Corpus JSON rows must be strings or objects")
        if text.strip():
            texts.append(text)
    return texts


def _load_corpus(path: Path, text_field: str) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Calibration corpus is missing: {path}")
    if path.suffix.lower() == ".jsonl":
        texts: list[str] = []
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}") from exc
            texts.extend(_extract_json_texts(row, text_field))
        return texts
    if path.suffix.lower() == ".json":
        try:
            value = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON corpus: {path}") from exc
        return _extract_json_texts(value, text_field)
    return [paragraph for paragraph in path.read_text().split("\n\n") if paragraph.strip()]


def _tokenize_sequences(
    tokenizer: Any,
    texts: Iterable[str],
    *,
    num_sequences: int,
    sequence_length: int,
    sequence_offset: int = 0,
) -> list[list[int]]:
    if num_sequences <= 0:
        raise ValueError("num_sequences must be positive")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if sequence_offset < 0:
        raise ValueError("sequence_offset must be non-negative")
    required_sequences = sequence_offset + num_sequences
    sequences: list[list[int]] = []
    token_buffer: list[int] = []
    cursor = 0
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    for text in texts:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
        )
        if not isinstance(encoded, Mapping):
            raise TypeError("Tokenizer must return a mapping")
        raw_ids = encoded.get("input_ids")
        if not isinstance(raw_ids, list) or not all(isinstance(item, int) for item in raw_ids):
            raise TypeError("Tokenizer input_ids must be a flat integer list")
        token_buffer.extend(cast(list[int], raw_ids))
        if isinstance(eos_token_id, int):
            token_buffer.append(eos_token_id)
        while len(token_buffer) - cursor >= sequence_length:
            sequences.append(token_buffer[cursor : cursor + sequence_length])
            cursor += sequence_length
            if len(sequences) == required_sequences:
                return sequences[sequence_offset:]
        if cursor >= 4096:
            token_buffer = token_buffer[cursor:]
            cursor = 0
    available = sum(len(sequence) for sequence in sequences) + len(token_buffer) - cursor
    required = required_sequences * sequence_length
    raise ValueError(f"Corpus provides only {available} usable tokens; {required} are required")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sequence_sha256(sequences: Iterable[Iterable[int]]) -> str:
    digest = hashlib.sha256()
    for sequence in sequences:
        for token_id in sequence:
            digest.update(token_id.to_bytes(8, byteorder="little", signed=True))
    return digest.hexdigest()


def _optional_file_sha256(path: Path) -> str:
    return _sha256_file(path) if path.is_file() else ""


def _load_transformers() -> Any:
    try:
        return importlib.import_module("transformers")
    except ImportError as exc:
        raise RuntimeError(
            'Calibration dependencies are missing; install with pip install -e ".[calibrate]"'
        ) from exc


def _model_class(transformers: Any, config: Any) -> Any:
    raw_architectures = getattr(config, "architectures", None)
    architectures = raw_architectures if isinstance(raw_architectures, list) else []
    if any("ConditionalGeneration" in item for item in architectures if isinstance(item, str)):
        model_class = getattr(transformers, "AutoModelForImageTextToText", None)
        if model_class is None:
            raise RuntimeError("Transformers lacks AutoModelForImageTextToText for this model")
        return model_class
    return transformers.AutoModelForCausalLM


def _target_widths(plan: ModelPlan) -> dict[str, int]:
    return {
        item.info.name: item.info.shape[-1]
        for item in plan.tensors
        if item.quantized
    }


def _checkpoint_files(plan: ModelPlan) -> dict[str, Path]:
    """Map every checkpoint key to its already-validated shard path."""
    shards = {shard.path.name: shard.path for shard in plan.shards}
    return {item.info.name: shards[item.shard_name] for item in plan.tensors}


def _peak_rss_bytes() -> int:
    """Return the process peak RSS in bytes on Linux and macOS."""
    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(maximum if sys.platform == "darwin" else maximum * 1024)


def run(args: argparse.Namespace) -> Path:
    """Capture and save calibration statistics for parsed CLI arguments."""
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    objectives = _parse_objectives(args.statistics)
    model_dir = Path(args.model_dir)
    corpus_path = Path(args.corpus)
    output_path = Path(args.output)
    plan = plan_model(
        QuantizeConfig(
            model_dir=model_dir,
            device=device,
            policy=args.policy,
            method="mse",
            gamma_proxy=False,
            verbose=False,
        )
    )
    widths = _target_widths(plan)
    policy_name = plan.policy.name
    checkpoint_files = _checkpoint_files(plan)

    transformers = _load_transformers()
    model_config = transformers.AutoConfig.from_pretrained(
        model_dir,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=args.trust_remote_code,
    )
    texts = _load_corpus(corpus_path, args.text_field)
    sequences = _tokenize_sequences(
        tokenizer,
        texts,
        num_sequences=args.num_sequences,
        sequence_length=args.sequence_length,
        sequence_offset=args.sequence_offset,
    )

    if args.dry_run:
        try:
            accelerate = importlib.import_module("accelerate")
        except ImportError as exc:
            raise RuntimeError(
                'Calibration dependencies are missing; install with pip install -e ".[calibrate]"'
            ) from exc
        with accelerate.init_empty_weights(include_buffers=True):
            empty_model = _model_class(transformers, model_config).from_config(
                model_config,
                attn_implementation=args.attention_implementation,
            )
        collector = ActivationCollector(
            widths,
            objectives,
            hessian_damp=args.hessian_damp,
        )
        handles = attach_activation_hooks(empty_model, collector)
        for handle in handles:
            handle.remove()
        if args.weight_loading == "streaming":
            infer_sequential_decoder_layout(empty_model, widths, checkpoint_files)
        print(
            json.dumps(
                {
                    "policy": policy_name,
                    "targets": len(widths),
                    "objectives": list(objectives),
                    "sequences": len(sequences),
                    "sequence_offset": args.sequence_offset,
                    "sequence_length": args.sequence_length,
                    "tokens": len(sequences) * args.sequence_length,
                    "weight_loading": args.weight_loading,
                    "corpus_sha256": _sha256_file(corpus_path),
                    "token_ids_sha256": _sequence_sha256(sequences),
                },
                indent=2,
            )
        )
        return output_path

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    if args.weight_loading == "streaming":
        try:
            accelerate = importlib.import_module("accelerate")
        except ImportError as exc:
            raise RuntimeError(
                'Calibration dependencies are missing; install with pip install -e ".[calibrate]"'
            ) from exc
        with accelerate.init_empty_weights(include_buffers=True):
            empty_model = _model_class(transformers, model_config).from_config(
                model_config,
                attn_implementation=args.attention_implementation,
            )
        print(
            f"[mxwave] streaming {len(sequences)} sequences through one decoder layer "
            f"at a time on {device}"
        )
        streamed = calibrate_decoder_sequentially(
            empty_model,
            widths,
            objectives,
            sequences,
            checkpoint_files,
            device=device,
            dtype=dtype,
            batch_size=args.batch_size,
            hessian_damp=args.hessian_damp,
            progress=lambda completed, total: print(
                f"[mxwave] calibration layer {completed}/{total}"
            ),
        )
        statistics = streamed.statistics
        counts = streamed.observation_counts
        del empty_model, streamed
    else:
        model_kwargs: dict[str, Any] = {
            "config": model_config,
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
            "trust_remote_code": args.trust_remote_code,
            "attn_implementation": args.attention_implementation,
        }
        if device.type != "cpu":
            model_kwargs["device_map"] = {"": str(device)}
        print(f"[mxwave] loading the full float model on {device} (resident mode)")
        model = _model_class(transformers, model_config).from_pretrained(
            model_dir, **model_kwargs
        )
        if device.type == "cpu":
            model.to(device)
        model.eval()

        collector = ActivationCollector(
            widths,
            objectives,
            hessian_damp=args.hessian_damp,
        )
        handles = attach_activation_hooks(model, collector)
        try:
            with torch.inference_mode():
                for start in range(0, len(sequences), args.batch_size):
                    rows = sequences[start : start + args.batch_size]
                    input_ids = torch.tensor(rows, dtype=torch.long, device=device)
                    attention_mask = torch.ones_like(input_ids)
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        return_dict=False,
                    )
                    del output, input_ids, attention_mask
                    completed = min(start + len(rows), len(sequences))
                    print(f"[mxwave] calibration {completed}/{len(sequences)} sequences")
        finally:
            for handle in handles:
                handle.remove()
        statistics = collector.finalize()
        counts = collector.observation_counts
        del model, collector

    elapsed = time.monotonic() - started
    peak_accelerator = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    metadata = {
        "policy": policy_name,
        "source_repository": args.source_repository or "",
        "source_revision": args.source_revision or "",
        "num_sequences": str(len(sequences)),
        "sequence_offset": str(args.sequence_offset),
        "sequence_length": str(args.sequence_length),
        "num_tokens": str(len(sequences) * args.sequence_length),
        "batch_size": str(args.batch_size),
        "weight_loading": args.weight_loading,
        "text_field": args.text_field,
        "corpus_sha256": _sha256_file(corpus_path),
        "token_ids_sha256": _sequence_sha256(sequences),
        "config_sha256": _optional_file_sha256(model_dir / "config.json"),
        "index_sha256": _optional_file_sha256(model_dir / "model.safetensors.index.json"),
        "hessian_damp": str(args.hessian_damp),
        "minimum_observations": str(min(counts.values())),
        "maximum_observations": str(max(counts.values())),
        "capture_seconds": f"{elapsed:.6f}",
        "peak_process_rss_bytes": str(_peak_rss_bytes()),
        "peak_accelerator_memory_allocated_bytes": str(peak_accelerator),
    }
    save_calibration_data(output_path, statistics, metadata)
    print(
        f"[mxwave] saved {','.join(sorted(statistics))} for {len(widths)} targets "
        f"to {output_path}"
    )
    print(
        f"[mxwave] peak RSS {_peak_rss_bytes() / (1024**3):.2f} GiB; "
        f"peak accelerator allocation {peak_accelerator / (1024**3):.2f} GiB"
    )
    return output_path


def main(argv: list[str] | None = None) -> int:
    """Run calibration and return a process exit status."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except (FileNotFoundError, ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"mxwave: calibration error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
