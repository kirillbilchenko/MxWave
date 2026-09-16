"""Checkpoint assembly for vLLM compressed-tensors MXFP4 artifacts."""

from __future__ import annotations

import json
import os
import shutil
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .shard import ShardFile, shard_tensor_keys
from .verify import verify_config_coverage

__all__ = [
    "assemble_output_dir",
    "build_model_card",
    "build_quantization_config",
    "copy_model_assets",
    "record_runtime_validation",
    "verify_emitted_config",
]

# Canonical schema emitted by compressed-tensors 0.17.0 for the MXFP4 preset.
_MXFP4_GROUP: dict[str, Any] = {
    "targets": [],
    "weights": {
        "num_bits": 4,
        "type": "float",
        "symmetric": True,
        "group_size": 32,
        "strategy": "group",
        "block_structure": None,
        "dynamic": False,
        "actorder": None,
        "scale_dtype": "torch.uint8",
        "zp_dtype": None,
        "observer": "memoryless_minmax",
        "observer_kwargs": {},
    },
    "input_activations": {
        "num_bits": 4,
        "type": "float",
        "symmetric": True,
        "group_size": 32,
        "strategy": "group",
        "block_structure": None,
        "dynamic": True,
        "actorder": None,
        "scale_dtype": "torch.uint8",
        "zp_dtype": None,
        "observer": None,
        "observer_kwargs": {},
    },
    "output_activations": None,
}

_ASSET_SUFFIXES = frozenset(
    {
        ".json",
        ".jinja",
        ".model",
        ".py",
        ".tiktoken",
        ".txt",
    }
)
_SKIPPED_ASSETS = frozenset(
    {
        "config.json",
        "model.safetensors.index.json",
        "mxwave-manifest.json",
        "mxwave-run.json",
        # Pre-rename provenance is accepted as input history, never copied into
        # a newly produced MxWave checkpoint.
        "mxstream-manifest.json",
        "mxstream-run.json",
    }
)


def _atomic_write_text(path: Path, value: str) -> None:
    """Atomically replace a small text artifact in the output directory."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.incomplete")
    temporary.unlink(missing_ok=True)
    try:
        temporary.write_text(value)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def build_quantization_config(
    target_modules: list[str],
    ignored_modules: list[str],
    *,
    compression_ratio: float | None = None,
) -> dict[str, Any]:
    """Build the minimal compressed-tensors 0.17 MXFP4 configuration.

    Targets are concrete checkpoint module names. This avoids broad ``Linear``
    matching and remains compatible with vLLM's fused-module mapping.
    """
    if not target_modules:
        raise ValueError("At least one MXFP4 target module is required")
    overlap = sorted(set(target_modules).intersection(ignored_modules))
    if overlap:
        raise ValueError(f"Modules cannot be both targeted and ignored: {overlap[:5]}")

    group = {**_MXFP4_GROUP, "targets": sorted(set(target_modules))}
    return {
        "config_groups": {"group_0": group},
        "quant_method": "compressed-tensors",
        "kv_cache_scheme": None,
        "format": "mxfp4-pack-quantized",
        "quantization_status": "compressed",
        "global_compression_ratio": compression_ratio,
        "ignore": sorted(set(ignored_modules)),
    }


def _shard_data_size(path: Path) -> int:
    """Return the total tensor-data byte size of a safetensors shard."""
    with path.open("rb") as stream:
        raw_length = stream.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"Invalid safetensors file: {path}")
        header_length = struct.unpack("<Q", raw_length)[0]
        stream.seek(8 + header_length)
        return path.stat().st_size - stream.tell()


def _build_index(output_shards: list[Path]) -> dict[str, Any]:
    """Build an index from emitted keys, never from source-checkpoint names."""
    weight_map: dict[str, str] = {}
    total_size = 0
    for path in output_shards:
        shard = ShardFile(path=path, weight_map={})
        for key in shard_tensor_keys(shard):
            if key in weight_map:
                raise ValueError(f"Duplicate emitted tensor key: {key}")
            weight_map[key] = path.name
        total_size += _shard_data_size(path)
    return {"metadata": {"total_size": total_size}, "weight_map": weight_map}


def build_model_card(manifest: dict[str, Any]) -> str:
    """Render an evidence-oriented Hugging Face model card for an artifact."""
    source = manifest.get("source")
    source = source if isinstance(source, dict) else {}
    repository = source.get("repository")
    revision = source.get("revision")
    producer = manifest.get("producer")
    producer = producer if isinstance(producer, dict) else {}
    method = str(manifest.get("method", "unknown"))
    policy = str(manifest.get("policy", "unknown"))
    targets = manifest.get("target_tensors", "unknown")
    output_bytes = manifest.get("actual_output_bytes")
    output_gib = (
        f"{float(output_bytes) / (1024**3):.2f} GiB"
        if isinstance(output_bytes, int | float)
        else "unknown"
    )
    ratio = manifest.get("global_compression_ratio")
    ratio_text = f"{float(ratio):.3f}x" if isinstance(ratio, int | float) else "unknown"
    selection = str(manifest.get("weight_scale_selection", "unknown"))
    tensor_row_chunk_size = manifest.get("tensor_row_chunk_size", "unknown")
    sqnr_data = manifest.get("sqnr_db")
    sqnr_data = sqnr_data if isinstance(sqnr_data, dict) else {}
    sqnr_count = sqnr_data.get("count", 0)
    sqnr_minimum = sqnr_data.get("minimum")
    sqnr_mean = sqnr_data.get("mean")
    sqnr_result = (
        f"{float(sqnr_minimum):.2f} dB minimum / {float(sqnr_mean):.2f} dB mean "
        f"across {sqnr_count} bounded row samples"
        if isinstance(sqnr_minimum, int | float) and isinstance(sqnr_mean, int | float)
        else "not recorded"
    )
    activation = manifest.get("activation_calibration")
    activation = activation if isinstance(activation, dict) else {}
    weighted_sqnr_data = (
        manifest.get("calibration_weighted_sqnr_db")
        if activation
        else manifest.get("gamma_weighted_sqnr_db")
    )
    weighted_sqnr_data = weighted_sqnr_data if isinstance(weighted_sqnr_data, dict) else {}
    weighted_sqnr_count = weighted_sqnr_data.get("count", 0)
    weighted_sqnr_minimum = weighted_sqnr_data.get("minimum")
    weighted_sqnr_mean = weighted_sqnr_data.get("mean")
    weighted_sqnr_result = (
        f"{float(weighted_sqnr_minimum):.2f} dB minimum / "
        f"{float(weighted_sqnr_mean):.2f} dB mean across "
        f"{weighted_sqnr_count} bounded row samples"
        if isinstance(weighted_sqnr_minimum, int | float)
        and isinstance(weighted_sqnr_mean, int | float)
        else "not applicable"
    )
    title_source = repository.rsplit("/", 1)[-1] if isinstance(repository, str) else "Model"
    source_text = (
        f"`{repository}` at immutable revision `{revision}`"
        if isinstance(repository, str) and isinstance(revision, str)
        else "the local source checkpoint recorded in the manifest"
    )
    gamma = manifest.get("gamma_proxy")
    if activation:
        calibration_text = (
            f"corpus-derived `{activation.get('objective', 'unknown')}` inputs on "
            f"{activation.get('weighted_tensors', 0)} targets from "
            f"{activation.get('num_sequences', 'unknown')} sequences"
        )
        weighted_sqnr_label = "Calibration-weighted SQNR"
    elif isinstance(gamma, dict):
        calibration_text = (
            f"LayerNorm gamma proxy on {gamma.get('weighted_tensors', 0)} targets; "
            f"{gamma.get('unweighted_tensors', 0)} targets unweighted"
        )
        weighted_sqnr_label = "Gamma-weighted SQNR"
    else:
        calibration_text = "none"
        weighted_sqnr_label = "Weighted SQNR"
    runtime_validation = manifest.get("runtime_validation")
    runtime_validation = runtime_validation if isinstance(runtime_validation, dict) else {}
    runtime_status = runtime_validation.get("status")
    if runtime_status == "passed":
        runtime_bits = [
            str(value)
            for key in ("runtime", "hardware", "linear_backend", "execution_mode")
            if isinstance(value := runtime_validation.get(key), str) and value
        ]
        runtime_result = "Passed" + (f" — {', '.join(runtime_bits)}" if runtime_bits else "")
    else:
        runtime_result = "Not recorded by the conversion process"

    smoke = runtime_validation.get("smoke")
    if (
        isinstance(smoke, dict)
        and isinstance(smoke.get("passed"), int)
        and isinstance(smoke.get("total"), int)
    ):
        smoke_result = f"{smoke['passed']}/{smoke['total']} passed"
    else:
        smoke_result = "not recorded"

    evaluation = runtime_validation.get("evaluation")
    if (
        isinstance(evaluation, dict)
        and isinstance(evaluation.get("passed"), int)
        and isinstance(evaluation.get("total"), int)
    ):
        suite = evaluation.get("suite", "unspecified suite")
        suite_version = evaluation.get("suite_version")
        version_text = f" v{suite_version}" if isinstance(suite_version, int) else ""
        thinking_text = ", thinking disabled" if evaluation.get("enable_thinking") is False else ""
        run_id = evaluation.get("run_id")
        run_text = f", run `{run_id}`" if isinstance(run_id, str) and run_id else ""
        evaluation_result = (
            f"{evaluation['passed']}/{evaluation['total']} passed — "
            f"{suite}{version_text}{thinking_text}{run_text}"
        )
    else:
        evaluation_result = "not measured yet"

    frontmatter = ["---"]
    if isinstance(repository, str):
        frontmatter.append(f"base_model: {repository}")
    frontmatter.extend(
        [
            "library_name: transformers",
            "tags:",
            "- mxfp4",
            "- compressed-tensors",
            "- vllm",
            "- mxwave",
            "---",
        ]
    )
    body = [
        f"# {title_source} — MxWave {method.upper()} MXFP4",
        "",
        (
            "This checkpoint was generated by "
            f"MxWave `{producer.get('version', 'unknown')}` from {source_text}."
        ),
        "It targets vLLM's `compressed-tensors` `mxfp4-pack-quantized` format.",
        "",
        "## Quantization contract",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Policy | `{policy}` |",
        f"| Weight method | `{method}` |",
        f"| Scale selection | `{selection}` |",
        f"| Tensor row chunk size | {tensor_row_chunk_size} |",
        "| Weight format | OCP MXFP4 E2M1, E8M0 scale, block size 32 |",
        "| Input activations | Dynamic MXFP4, block size 32 |",
        f"| Quantized tensors | {targets} |",
        f"| Calibration | {calibration_text} |",
        f"| Artifact data size | {output_gib} |",
        f"| Overall source/data compression | {ratio_text} |",
        f"| Sampled weight SQNR | {sqnr_result} |",
        f"| {weighted_sqnr_label} | {weighted_sqnr_result} |",
        "",
        (
            "The exact targeted and ignored module lists, per-tensor SQNR samples, source "
            "identity, and byte counts are recorded in `mxwave-manifest.json`."
        ),
        "",
        "## Usage",
        "",
        "```bash",
        "vllm serve . --load-format safetensors --linear-backend marlin",
        "```",
        "",
        (
            "The command above is the conservative DGX Spark (SM121) path: Marlin consumes "
            "the packed MXFP4 weights but computes with unquantized activations (W4A16). "
            "Native W4A4 requires an MXFP4 backend that explicitly supports the installed "
            "GPU and vLLM build. Always verify the selected kernel in the startup log."
        ),
        "",
        "## Validation status",
        "",
        "| Check | Result |",
        "|---|---|",
        "| Safetensors structure, shapes, and dtypes | Passed during conversion |",
        "| Quantization target/ignore coverage | Passed during conversion |",
        f"| Bounded weight SQNR | {sqnr_result} |",
        f"| {weighted_sqnr_label} | {weighted_sqnr_result} |",
        f"| vLLM load and kernel selection | {runtime_result} |",
        f"| API smoke contract | {smoke_result} |",
        f"| Deterministic task screening | {evaluation_result} |",
        "| Perplexity and downstream task quality | Not measured yet |",
        "| Throughput and latency | Not measured yet |",
        "",
        (
            "Do not infer model quality from structural validation or weight SQNR alone. "
            "Compare deterministic perplexity and task results against the same source model "
            "and runtime settings before deployment."
        ),
        "",
        "## Limitations",
        "",
        (
            "- Only modules listed by the recorded policy are quantized; all others remain "
            "in their source dtype."
        ),
        (
            "- LayerNorm gamma is a low-cost activation-magnitude proxy, not corpus-derived "
            "calibration data."
            if not activation and isinstance(gamma, dict)
            else "- Calibration quality depends on the representativeness of the recorded corpus."
        ),
        "- Runtime compatibility depends on the exact vLLM/compressed-tensors build and GPU.",
        (
            "- This artifact is experimental until load, generation, perplexity, and task "
            "tests are recorded."
        ),
        "",
    ]
    return "\n".join(frontmatter + [""] + body)


def record_runtime_validation(
    output_dir: str | Path,
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    """Record post-conversion runtime evidence and regenerate the model card.

    The caller owns the measurements. This function only validates the minimal
    envelope and atomically keeps ``mxwave-manifest.json`` and ``README.md`` in sync.
    """
    output = Path(output_dir)
    manifest_path = output / "mxwave-manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"MxWave manifest is missing: {manifest_path}")
    raw_manifest = json.loads(manifest_path.read_text())
    if not isinstance(raw_manifest, dict):
        raise TypeError("MxWave manifest must contain a JSON object")
    normalized = dict(validation)
    if normalized.get("status") not in {"passed", "failed", "partial"}:
        raise ValueError("runtime validation status must be passed, failed, or partial")
    json.dumps(normalized)
    manifest: dict[str, Any] = raw_manifest
    manifest["runtime_validation"] = normalized
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    card_text = build_model_card(manifest)
    _atomic_write_text(manifest_path, manifest_text)
    _atomic_write_text(output / "README.md", card_text)
    return manifest


def copy_model_assets(model_dir: str | Path, output_dir: str | Path) -> list[str]:
    """Copy tokenizer, processor, generation, template, and remote-code assets."""
    source = Path(model_dir)
    output = Path(output_dir)
    copied: list[str] = []
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.name in _SKIPPED_ASSETS:
            continue
        if path.suffix.lower() not in _ASSET_SUFFIXES:
            continue
        relative = path.relative_to(source)
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.incomplete")
        temporary.unlink(missing_ok=True)
        try:
            shutil.copy2(path, temporary)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        copied.append(relative.as_posix())
    return copied


def assemble_output_dir(
    model_dir: str | Path,
    output_dir: str | Path,
    output_shards: list[Path],
    *,
    target_modules: list[str],
    ignored_modules: list[str],
    real_modules: list[str],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Assemble and atomically index a drop-in checkpoint.

    Coverage is checked before ``config.json`` is emitted, as required by the
    project's verification-first contract.
    """
    source = Path(model_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    gaps = verify_config_coverage(target_modules, ignored_modules, real_modules)
    if gaps:
        raise ValueError(f"Quantization config has uncovered modules: {gaps[:10]}")

    source_config_path = source / "config.json"
    if not source_config_path.is_file():
        raise FileNotFoundError(f"Source config.json is missing: {source_config_path}")
    raw_config = json.loads(source_config_path.read_text())
    if not isinstance(raw_config, dict):
        raise TypeError("Source config.json must contain a JSON object")
    config: dict[str, Any] = raw_config

    source_bytes = sum(path.stat().st_size for path in source.glob("*.safetensors"))
    output_bytes = sum(path.stat().st_size for path in output_shards)
    compression_ratio = (
        round(source_bytes / output_bytes, 4) if source_bytes > 0 and output_bytes > 0 else None
    )
    quantization_config = build_quantization_config(
        target_modules,
        ignored_modules,
        compression_ratio=compression_ratio,
    )
    config["quantization_config"] = quantization_config

    copied_assets = copy_model_assets(source, output)
    manifest["copied_assets"] = copied_assets
    manifest["actual_output_bytes"] = output_bytes
    manifest["global_compression_ratio"] = compression_ratio

    index = _build_index(output_shards)
    _atomic_write_text(output / "config.json", json.dumps(config, indent=2) + "\n")
    _atomic_write_text(
        output / "model.safetensors.index.json",
        json.dumps(index, indent=2) + "\n",
    )
    _atomic_write_text(
        output / "mxwave-manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    _atomic_write_text(output / "README.md", build_model_card(manifest))
    return quantization_config


def verify_emitted_config(
    output_dir: str | Path,
    real_modules: list[str],
) -> list[str]:
    """Verify that emitted targets/ignores cover every supplied real module."""
    config_path = Path(output_dir) / "config.json"
    if not config_path.exists():
        return real_modules
    raw_config = json.loads(config_path.read_text())
    if not isinstance(raw_config, dict):
        return real_modules
    raw_quantization = raw_config.get("quantization_config")
    if not isinstance(raw_quantization, dict):
        return real_modules
    raw_groups = raw_quantization.get("config_groups")
    targets: list[str] = []
    if isinstance(raw_groups, dict):
        for raw_group in raw_groups.values():
            if isinstance(raw_group, dict):
                raw_targets = raw_group.get("targets")
                if isinstance(raw_targets, list):
                    targets.extend(item for item in raw_targets if isinstance(item, str))
    raw_ignore = raw_quantization.get("ignore")
    ignore = (
        [item for item in raw_ignore if isinstance(item, str)]
        if isinstance(raw_ignore, list)
        else []
    )
    return verify_config_coverage(targets, ignore, real_modules)
