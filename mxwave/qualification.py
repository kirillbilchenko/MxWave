"""Qualification orchestration and checkpoint verification.

The qualifier intentionally separates model evidence from runtime orchestration.
MxWave verifies checkpoint structure, runs explicit argv-based evidence hooks,
imports versioned reports, and evaluates predeclared gates.  Platform projects
such as ``local-spark`` remain responsible for starting vLLM and collecting
hardware-specific latency, throughput, memory, and MTP evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import signal
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from .shard import ShardFile, TensorInfo, discover_shards, shard_tensor_info

SPEC_FORMAT = "mxwave-qualification-spec-v1"
REPORT_FORMAT = "mxwave-qualification-v1"
RUNTIME_EVIDENCE_FORMAT = "mxwave-runtime-evidence-v1"

EvidenceKind = Literal[
    "perplexity",
    "divergence",
    "serving",
    "runtime",
    "task",
    "generic",
]
GateOperator = Literal["<", "<=", ">", ">=", "==", "!="]
Decision = Literal["qualified-default", "qualified-optional", "rejected", "incomplete"]

_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}\Z")
_MODULE_NAME_PATTERN = re.compile(r"[A-Za-z0-9_.-]+\Z")
_DIGEST_PINNED_IMAGE_PATTERN = re.compile(r"\S+@sha256:[0-9a-fA-F]{64}\Z")
_SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}\Z")
_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}
_MODEL_ASSET_SUFFIXES = frozenset({".json", ".jinja", ".model", ".py", ".tiktoken", ".txt"})
_NON_ASSET_NAMES = frozenset(
    {
        "config.json",
        "model.safetensors.index.json",
        "mxwave-manifest.json",
        "mxwave-run.json",
    }
)

_EVIDENCE_FORMATS: dict[EvidenceKind, frozenset[str]] = {
    "perplexity": frozenset({"mxwave-api-perplexity-v1"}),
    "divergence": frozenset({"mxwave-next-token-divergence-v1"}),
    "serving": frozenset(
        {
            "mxwave-serving-qualification-collection-v1",
            "mxwave-serving-qualification-comparison-v1",
        }
    ),
    "runtime": frozenset({RUNTIME_EVIDENCE_FORMAT}),
    "task": frozenset(),
    "generic": frozenset(),
}

_SAFE_INHERITED_ENVIRONMENT = (
    "CUDA_VISIBLE_DEVICES",
    "HF_HOME",
    "HOME",
    "LD_LIBRARY_PATH",
    "PATH",
    "PYTHONPATH",
    "TRANSFORMERS_CACHE",
    "VIRTUAL_ENV",
)


@dataclass(frozen=True, slots=True)
class HookSpec:
    """One bounded subprocess that materializes an evidence artifact."""

    argv: tuple[str, ...]
    timeout_seconds: float
    cwd: Path | None
    inherit_environment: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvidenceBinding:
    """One evidence value frozen to checkpoint structure or a literal protocol value."""

    pointer: str
    structure_pointer: str | None
    literal: object
    uses_literal: bool


@dataclass(frozen=True, slots=True)
class EvidenceSpec:
    """A versioned evidence artifact required or optionally attached to a run."""

    evidence_id: str
    kind: EvidenceKind
    path: Path
    required: bool
    expected_format: str | None
    hook: HookSpec | None
    bindings: tuple[EvidenceBinding, ...]


@dataclass(frozen=True, slots=True)
class GateSpec:
    """A frozen scalar or boolean comparison over one evidence document."""

    gate_id: str
    evidence_id: str
    pointer: str
    operator: GateOperator
    threshold: object


@dataclass(frozen=True, slots=True)
class QualificationSpec:
    """Parsed qualification contract."""

    run_id: str
    model_dir: Path
    model_label: str
    hash_shards: bool
    evidence: tuple[EvidenceSpec, ...]
    gates: tuple[GateSpec, ...]
    required_capabilities: tuple[str, ...]
    success_decision: Literal["qualified-default", "qualified-optional"]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


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
    temporary = path.with_name(f".{path.name}.{os.getpid()}.incomplete")
    temporary.unlink(missing_ok=True)
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.incomplete")
    temporary.unlink(missing_ok=True)
    try:
        temporary.write_text(value)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json_object(path: Path) -> dict[str, Any]:
    raw: Any = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return cast(dict[str, Any], raw)


def _string_list(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError(f"{label} must be a list of strings")
    return tuple(cast(list[str], value))


def _resolve_path(raw: object, *, base: Path, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise TypeError(f"{label} must be a non-empty path string")
    path = Path(raw)
    return path if path.is_absolute() else base / path


def _parse_hook(raw: object, *, base: Path) -> HookSpec | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise TypeError("evidence hook must be an object")
    raw_argv = raw.get("argv")
    argv = _string_list(raw_argv, "evidence hook argv")
    if not argv or any(not item for item in argv):
        raise ValueError("evidence hook argv must contain non-empty arguments")
    timeout = raw.get("timeout_seconds", 3600.0)
    if not isinstance(timeout, int | float) or not 0 < float(timeout) <= 172800:
        raise ValueError("evidence hook timeout_seconds must be in (0, 172800]")
    raw_cwd = raw.get("cwd")
    cwd = None if raw_cwd is None else _resolve_path(raw_cwd, base=base, label="hook cwd")
    inherited = raw.get("inherit_environment", [])
    inherited_names = _string_list(inherited, "hook inherit_environment")
    if any("=" in name or not name for name in inherited_names):
        raise ValueError("hook environment names must be non-empty variable names")
    return HookSpec(
        argv=argv,
        timeout_seconds=float(timeout),
        cwd=cwd,
        inherit_environment=inherited_names,
    )


def _parse_bindings(raw: object, *, evidence_id: str) -> tuple[EvidenceBinding, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"evidence {evidence_id} bindings must be a non-empty list")
    bindings: list[EvidenceBinding] = []
    seen: set[str] = set()
    has_structure_binding = False
    for raw_binding in raw:
        if not isinstance(raw_binding, dict):
            raise TypeError(f"evidence {evidence_id} binding must be an object")
        pointer = raw_binding.get("pointer")
        if not isinstance(pointer, str) or not pointer.startswith("/"):
            raise ValueError(f"evidence {evidence_id} binding pointer must begin with /")
        if pointer in seen:
            raise ValueError(f"evidence {evidence_id} has duplicate binding {pointer!r}")
        seen.add(pointer)
        has_structure = "structure_pointer" in raw_binding
        has_literal = "value" in raw_binding
        if has_structure == has_literal:
            raise ValueError(
                f"evidence {evidence_id} binding {pointer!r} must have exactly one of "
                "structure_pointer or value"
            )
        structure_pointer: str | None = None
        literal: object = None
        if has_structure:
            raw_structure = raw_binding.get("structure_pointer")
            if not isinstance(raw_structure, str) or not raw_structure.startswith("/"):
                raise ValueError(
                    f"evidence {evidence_id} structure_pointer must begin with /"
                )
            structure_pointer = raw_structure
            has_structure_binding = True
        else:
            literal = raw_binding.get("value")
            if not isinstance(literal, bool | int | float | str) and literal is not None:
                raise TypeError(
                    f"evidence {evidence_id} literal binding values must be JSON scalars"
                )
        bindings.append(
            EvidenceBinding(
                pointer=pointer,
                structure_pointer=structure_pointer,
                literal=literal,
                uses_literal=has_literal,
            )
        )
    if not has_structure_binding:
        raise ValueError(f"evidence {evidence_id} must bind to the inspected checkpoint")
    if not any(
        binding.structure_pointer == "/checkpoint_sha256" for binding in bindings
    ):
        raise ValueError(
            f"evidence {evidence_id} must bind a value to /checkpoint_sha256"
        )
    protocol_bindings = [
        binding
        for binding in bindings
        if binding.pointer == "/protocol_sha256" and binding.uses_literal
    ]
    if len(protocol_bindings) != 1 or not _is_sha256(protocol_bindings[0].literal):
        raise ValueError(
            f"evidence {evidence_id} must freeze /protocol_sha256 to a SHA-256 literal"
        )
    return tuple(bindings)


def load_spec(path: str | Path) -> tuple[QualificationSpec, str, dict[str, Any]]:
    """Parse and validate a qualification specification."""
    spec_path = Path(path)
    raw_bytes = spec_path.read_bytes()
    raw: Any = json.loads(raw_bytes)
    if not isinstance(raw, dict) or raw.get("format") != SPEC_FORMAT:
        raise ValueError(f"Qualification spec must use format {SPEC_FORMAT!r}")
    document = cast(dict[str, Any], raw)
    run_id = document.get("run_id")
    if not isinstance(run_id, str) or _IDENTIFIER_PATTERN.fullmatch(run_id) is None:
        raise ValueError("run_id must be a safe identifier of at most 120 characters")
    model = document.get("model")
    if not isinstance(model, dict):
        raise TypeError("model must be an object")
    model_dir = _resolve_path(model.get("path"), base=spec_path.parent, label="model.path")
    model_label = model.get("label")
    if not isinstance(model_label, str) or not model_label:
        raise ValueError("model.label must be a non-empty string")
    hash_shards = model.get("hash_shards", False)
    if not isinstance(hash_shards, bool):
        raise TypeError("model.hash_shards must be boolean")
    if not hash_shards:
        raise ValueError("qualification runs require model.hash_shards=true")

    raw_evidence = document.get("evidence", [])
    if not isinstance(raw_evidence, list):
        raise TypeError("evidence must be a list")
    evidence: list[EvidenceSpec] = []
    seen_evidence: set[str] = set()
    for raw_item in raw_evidence:
        if not isinstance(raw_item, dict):
            raise TypeError("each evidence entry must be an object")
        evidence_id = raw_item.get("id")
        kind = raw_item.get("kind")
        if not isinstance(evidence_id, str) or _IDENTIFIER_PATTERN.fullmatch(evidence_id) is None:
            raise ValueError("evidence.id must be a safe identifier of at most 120 characters")
        if evidence_id in seen_evidence:
            raise ValueError(f"duplicate evidence id: {evidence_id}")
        seen_evidence.add(evidence_id)
        if kind not in _EVIDENCE_FORMATS:
            raise ValueError(f"unsupported evidence kind for {evidence_id}: {kind!r}")
        required = raw_item.get("required", True)
        if not isinstance(required, bool):
            raise TypeError(f"evidence {evidence_id} required must be boolean")
        expected_format = raw_item.get("expected_format")
        if expected_format is not None and not isinstance(expected_format, str):
            raise TypeError(f"evidence {evidence_id} expected_format must be a string")
        registered_formats = _EVIDENCE_FORMATS[cast(EvidenceKind, kind)]
        if (
            expected_format is not None
            and registered_formats
            and expected_format not in registered_formats
        ):
            raise ValueError(
                f"evidence {evidence_id} expected_format must be one of "
                f"{sorted(registered_formats)}"
            )
        evidence.append(
            EvidenceSpec(
                evidence_id=evidence_id,
                kind=cast(EvidenceKind, kind),
                path=_resolve_path(
                    raw_item.get("path"), base=spec_path.parent, label=f"evidence {evidence_id} path"
                ),
                required=required,
                expected_format=expected_format,
                hook=_parse_hook(raw_item.get("hook"), base=spec_path.parent),
                bindings=_parse_bindings(raw_item.get("bindings"), evidence_id=evidence_id),
            )
        )

    if not evidence:
        raise ValueError("qualification requires at least one evidence artifact")

    raw_gates = document.get("gates", [])
    if not isinstance(raw_gates, list):
        raise TypeError("gates must be a list")
    gates: list[GateSpec] = []
    seen_gates: set[str] = set()
    for raw_gate in raw_gates:
        if not isinstance(raw_gate, dict):
            raise TypeError("each gate must be an object")
        gate_id = raw_gate.get("id")
        evidence_id = raw_gate.get("evidence")
        pointer = raw_gate.get("pointer")
        operator = raw_gate.get("operator")
        if not isinstance(gate_id, str) or _IDENTIFIER_PATTERN.fullmatch(gate_id) is None:
            raise ValueError("gate.id must be a safe identifier of at most 120 characters")
        if gate_id in seen_gates:
            raise ValueError(f"duplicate gate id: {gate_id}")
        seen_gates.add(gate_id)
        if evidence_id != "structure" and evidence_id not in seen_evidence:
            raise ValueError(f"gate {gate_id} references unknown evidence {evidence_id!r}")
        if not isinstance(evidence_id, str):
            raise TypeError(f"gate {gate_id} evidence must be a string")
        if not isinstance(pointer, str) or (pointer and not pointer.startswith("/")):
            raise ValueError(f"gate {gate_id} pointer must be empty or begin with /")
        if operator not in {"<", "<=", ">", ">=", "==", "!="}:
            raise ValueError(f"gate {gate_id} has unsupported operator {operator!r}")
        threshold = raw_gate.get("threshold")
        if not isinstance(threshold, bool | int | float | str) and threshold is not None:
            raise TypeError(f"gate {gate_id} threshold must be a JSON scalar")
        if operator in {"<", "<=", ">", ">="} and (
            isinstance(threshold, bool) or not isinstance(threshold, int | float)
        ):
            raise TypeError(f"ordered gate {gate_id} threshold must be numeric")
        gates.append(
            GateSpec(
                gate_id=gate_id,
                evidence_id=evidence_id,
                pointer=pointer,
                operator=cast(GateOperator, operator),
                threshold=threshold,
            )
        )

    if not gates:
        raise ValueError("qualification requires at least one predeclared gate")
    capabilities = _string_list(
        document.get("required_capabilities", ["structural"]),
        "required_capabilities",
    )
    if not capabilities or len(set(capabilities)) != len(capabilities):
        raise ValueError("required_capabilities must be a non-empty unique list")
    if "structural" not in capabilities:
        raise ValueError("required_capabilities must include structural")
    required_quality_ids = {
        item.evidence_id
        for item in evidence
        if item.required and item.kind in {"perplexity", "divergence", "task"}
    }
    if not required_quality_ids:
        raise ValueError("qualification requires at least one required quality evidence artifact")
    if not any(gate.evidence_id in required_quality_ids for gate in gates):
        raise ValueError("qualification requires a gate over required quality evidence")
    success_decision = document.get("success_decision", "qualified-default")
    if success_decision not in {"qualified-default", "qualified-optional"}:
        raise ValueError("success_decision must be qualified-default or qualified-optional")
    parsed = QualificationSpec(
        run_id=run_id,
        model_dir=model_dir,
        model_label=model_label,
        hash_shards=hash_shards,
        evidence=tuple(evidence),
        gates=tuple(gates),
        required_capabilities=capabilities,
        success_decision=cast(
            Literal["qualified-default", "qualified-optional"], success_decision
        ),
    )
    return parsed, _sha256_bytes(raw_bytes), document


def _config_modules(
    config: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, str], str]:
    quantization = config.get("quantization_config")
    if not isinstance(quantization, dict):
        raise TypeError("config.json has no quantization_config object")
    if quantization.get("quant_method") != "compressed-tensors":
        raise ValueError("quantization_config.quant_method must be compressed-tensors")
    if quantization.get("quantization_status") != "compressed":
        raise ValueError("quantization_config.quantization_status must be compressed")
    checkpoint_format = quantization.get("format")
    if checkpoint_format not in {"mxfp4-pack-quantized", "mixed-precision"}:
        raise ValueError("qualification requires MxWave MXFP4 or mixed-precision format")
    groups = quantization.get("config_groups")
    if not isinstance(groups, dict) or not groups:
        raise TypeError("quantization_config has no config_groups object")
    targets: list[str] = []
    target_formats: dict[str, str] = {}
    observed_formats: set[str] = set()
    for group_name, raw_group in groups.items():
        if not isinstance(raw_group, dict):
            raise TypeError(f"quantization group {group_name!r} is not an object")
        raw_targets = raw_group.get("targets")
        if not isinstance(raw_targets, list) or not all(
            isinstance(item, str) for item in raw_targets
        ):
            raise TypeError(f"quantization group {group_name!r} targets must be strings")
        group_targets = cast(list[str], raw_targets)
        if (
            not group_targets
            or group_targets != sorted(set(group_targets))
            or any(_MODULE_NAME_PATTERN.fullmatch(item) is None for item in group_targets)
        ):
            raise ValueError(
                f"quantization group {group_name!r} targets must be sorted concrete names"
            )
        group_format = raw_group.get("format", checkpoint_format)
        if group_format not in {"mxfp4-pack-quantized", "float-quantized"}:
            raise ValueError(
                f"quantization group {group_name!r} has unsupported format {group_format!r}"
            )
        observed_formats.add(str(group_format))
        _validate_group_scheme(str(group_name), raw_group, str(group_format))
        for target in group_targets:
            if target in target_formats:
                raise ValueError(f"quantization config contains duplicate target {target!r}")
            target_formats[target] = str(group_format)
        targets.extend(group_targets)
    raw_ignore = quantization.get("ignore")
    ignore = _string_list(raw_ignore, "quantization_config.ignore")
    if tuple(ignore) != tuple(sorted(set(ignore))) or any(
        _MODULE_NAME_PATTERN.fullmatch(item) is None for item in ignore
    ):
        raise ValueError("quantization_config.ignore must contain sorted concrete names")
    if len(set(targets)) != len(targets):
        raise ValueError("quantization config contains duplicate target modules")
    if set(targets).intersection(ignore):
        raise ValueError("quantization targets and ignores overlap")
    if checkpoint_format == "mxfp4-pack-quantized" and observed_formats != {
        "mxfp4-pack-quantized"
    }:
        raise ValueError("pure MXFP4 checkpoints may only contain MXFP4 groups")
    if checkpoint_format == "mixed-precision" and observed_formats != {
        "mxfp4-pack-quantized",
        "float-quantized",
    }:
        raise ValueError("mixed-precision checkpoints require both MXFP4 and FP8 groups")
    return (
        tuple(sorted(targets)),
        tuple(sorted(ignore)),
        target_formats,
        str(checkpoint_format),
    )


def _require_scheme_fields(
    scheme: Mapping[str, Any], expected: Mapping[str, object], *, label: str
) -> None:
    for key, value in expected.items():
        actual = scheme.get(key)
        if isinstance(value, bool):
            matches = isinstance(actual, bool) and actual is value
        elif value is None:
            matches = actual is None
        else:
            matches = actual == value and not isinstance(actual, bool)
        if not matches:
            raise ValueError(f"{label}.{key} must be {value!r}, found {actual!r}")


def _validate_group_scheme(
    group_name: str, group: Mapping[str, Any], group_format: str
) -> None:
    weights = group.get("weights")
    if not isinstance(weights, dict):
        raise TypeError(f"quantization group {group_name!r} has no weights scheme")
    if group_format == "mxfp4-pack-quantized":
        _require_scheme_fields(
            weights,
            {
                "num_bits": 4,
                "type": "float",
                "symmetric": True,
                "group_size": 32,
                "strategy": "group",
                "dynamic": False,
                "scale_dtype": "torch.uint8",
            },
            label=f"config_groups.{group_name}.weights",
        )
        activations = group.get("input_activations")
        if not isinstance(activations, dict):
            raise TypeError(
                f"quantization group {group_name!r} has no input_activations scheme"
            )
        _require_scheme_fields(
            activations,
            {
                "num_bits": 4,
                "type": "float",
                "symmetric": True,
                "group_size": 32,
                "strategy": "group",
                "dynamic": True,
                "scale_dtype": "torch.uint8",
            },
            label=f"config_groups.{group_name}.input_activations",
        )
    else:
        _require_scheme_fields(
            weights,
            {
                "num_bits": 8,
                "type": "float",
                "symmetric": True,
                "group_size": None,
                "strategy": "channel",
                "dynamic": False,
            },
            label=f"config_groups.{group_name}.weights",
        )
        if group.get("input_activations") is not None:
            raise ValueError(f"FP8 group {group_name!r} must use weight-only activations")
    if group.get("output_activations") is not None:
        raise ValueError(f"quantization group {group_name!r} output_activations must be null")


def _checkpoint_headers(
    model_dir: Path,
) -> tuple[dict[str, TensorInfo], dict[str, str], tuple[Path, ...]]:
    shards, weight_map = discover_shards(model_dir)
    info_by_key: dict[str, TensorInfo] = {}
    actual_map: dict[str, str] = {}
    for shard in shards:
        resolved = shard.path.resolve()
        try:
            resolved.relative_to(model_dir.resolve())
        except ValueError as error:
            raise ValueError(f"Checkpoint shard escapes model directory: {shard.path}") from error
        if not shard.path.is_file():
            raise FileNotFoundError(f"Checkpoint shard is missing: {shard.path}")
        shard_info = shard_tensor_info(ShardFile(shard.path, {}))
        _validate_tensor_layout(shard.path, shard_info)
        for name, info in shard_info.items():
            if name in info_by_key:
                raise ValueError(f"Duplicate tensor {name!r} across checkpoint shards")
            info_by_key[name] = info
            actual_map[name] = shard.path.name
    if weight_map is None:
        weight_map = actual_map
    if set(weight_map) != set(actual_map):
        missing = sorted(set(weight_map).difference(actual_map))
        unindexed = sorted(set(actual_map).difference(weight_map))
        raise ValueError(
            f"Index/header disagreement: missing={missing[:5]}, unindexed={unindexed[:5]}"
        )
    disagreements = [
        name for name, filename in weight_map.items() if actual_map.get(name) != filename
    ]
    if disagreements:
        name = disagreements[0]
        raise ValueError(
            f"Index maps {name!r} to {weight_map[name]!r}, found in {actual_map[name]!r}"
        )
    return info_by_key, weight_map, tuple(shard.path for shard in shards)


def _validate_tensor_layout(path: Path, tensors: Mapping[str, TensorInfo]) -> None:
    cursor = 0
    ordered = sorted(tensors.values(), key=lambda item: (item.data_offsets[0], item.name))
    for info in ordered:
        start, end = info.data_offsets
        if start != cursor:
            raise ValueError(
                f"Safetensors data in {path} is overlapping or non-contiguous at {info.name!r}"
            )
        item_size = _DTYPE_BYTES.get(info.dtype)
        if item_size is None:
            raise ValueError(f"Unsupported safetensors dtype {info.dtype!r} in {path}")
        expected = math.prod(info.shape) * item_size
        if end - start != expected:
            raise ValueError(
                f"Tensor {info.name!r} has shape/dtype byte size {expected}, "
                f"but its offsets contain {end - start} bytes"
            )
        cursor = end


def verify_checkpoint(model_dir: str | Path, *, hash_shards: bool = False) -> dict[str, Any]:
    """Perform header-only whole-checkpoint verification.

    Tensor payloads are never materialized.  The verifier checks index/header
    agreement, manifest/config coverage identity, complete MXFP4 pairs, legal
    dtypes and shapes, and byte-accounting metadata.
    """
    root = Path(model_dir)
    config_path = root / "config.json"
    index_path = root / "model.safetensors.index.json"
    manifest_path = root / "mxwave-manifest.json"
    for path in (config_path, index_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required checkpoint file is missing: {path}")
    config = _read_json_object(config_path)
    index = _read_json_object(index_path)
    manifest = _read_json_object(manifest_path)
    targets, ignored, target_formats, checkpoint_format = _config_modules(config)
    manifest_targets = _string_list(manifest.get("target_modules"), "manifest target_modules")
    manifest_ignored = _string_list(
        manifest.get("ignored_modules"), "manifest ignored_modules"
    )
    if targets != tuple(sorted(manifest_targets)):
        raise ValueError("config and manifest target modules differ")
    if ignored != tuple(sorted(manifest_ignored)):
        raise ValueError("config and manifest ignored modules differ")
    if checkpoint_format == "mixed-precision":
        manifest_mxfp4 = _string_list(
            manifest.get("mxfp4_target_modules"), "manifest mxfp4_target_modules"
        )
        manifest_fp8 = _string_list(
            manifest.get("fp8_target_modules"), "manifest fp8_target_modules"
        )
        expected_mxfp4 = tuple(
            sorted(
                module
                for module, target_format in target_formats.items()
                if target_format == "mxfp4-pack-quantized"
            )
        )
        expected_fp8 = tuple(
            sorted(
                module
                for module, target_format in target_formats.items()
                if target_format == "float-quantized"
            )
        )
        if tuple(sorted(manifest_mxfp4)) != expected_mxfp4:
            raise ValueError("manifest MXFP4 targets differ from config groups")
        if tuple(sorted(manifest_fp8)) != expected_fp8:
            raise ValueError("manifest FP8 targets differ from config groups")
        for field, expected in (
            ("mxfp4_target_tensors", len(expected_mxfp4)),
            ("fp8_target_tensors", len(expected_fp8)),
            ("target_tensors", len(targets)),
        ):
            if manifest.get(field) != expected:
                raise ValueError(f"manifest {field} does not match checkpoint coverage")
        composition = manifest.get("composition")
        if not isinstance(composition, dict) or tuple(
            sorted(_string_list(composition.get("selected_modules"), "composition selected_modules"))
        ) != expected_fp8:
            raise ValueError("manifest composition selected modules differ from FP8 targets")

    info_by_key, weight_map, shards = _checkpoint_headers(root)
    for module in targets:
        raw_name = f"{module}.weight"
        packed_name = f"{module}.weight_packed"
        scale_name = f"{module}.weight_scale"
        if target_formats[module] == "mxfp4-pack-quantized":
            if raw_name in info_by_key:
                raise ValueError(f"Target module retains raw weight {raw_name!r}")
            if packed_name not in info_by_key or scale_name not in info_by_key:
                raise ValueError(f"Target module {module!r} has an incomplete MXFP4 pair")
            packed = info_by_key[packed_name]
            scale = info_by_key[scale_name]
            if packed.dtype != "U8" or scale.dtype != "U8":
                raise ValueError(
                    f"Target module {module!r} must use U8 packed weights and scales"
                )
            if len(packed.shape) != 2 or len(scale.shape) != 2:
                raise ValueError(f"Target module {module!r} packed tensors must be matrices")
            if packed.shape[0] != scale.shape[0] or packed.shape[1] != scale.shape[1] * 16:
                raise ValueError(
                    f"Target module {module!r} has incompatible packed/scale shapes "
                    f"{packed.shape} and {scale.shape}"
                )
        else:
            if packed_name in info_by_key:
                raise ValueError(f"FP8 target module {module!r} retains an MXFP4 packed weight")
            if raw_name not in info_by_key or scale_name not in info_by_key:
                raise ValueError(f"Target module {module!r} has an incomplete FP8 pair")
            weight = info_by_key[raw_name]
            scale = info_by_key[scale_name]
            if weight.dtype != "F8_E4M3" or scale.dtype != "F32":
                raise ValueError(
                    f"FP8 target module {module!r} must use F8_E4M3 weight and F32 scale"
                )
            if (
                len(weight.shape) != 2
                or len(scale.shape) != 2
                or weight.shape[0] != scale.shape[0]
                or scale.shape[1] != 1
            ):
                raise ValueError(
                    f"Target module {module!r} has incompatible FP8 weight/scale shapes "
                    f"{weight.shape} and {scale.shape}"
                )

    packed_bases = {
        name.removesuffix(".weight_packed")
        for name in info_by_key
        if name.endswith(".weight_packed")
    }
    expected_packed_bases = {
        module
        for module, target_format in target_formats.items()
        if target_format == "mxfp4-pack-quantized"
    }
    if packed_bases != expected_packed_bases:
        unexpected_packed = sorted(packed_bases.difference(expected_packed_bases))
        missing_packed = sorted(expected_packed_bases.difference(packed_bases))
        raise ValueError(
            f"MXFP4 packed-target disagreement: unexpected={unexpected_packed[:5]}, "
            f"missing={missing_packed[:5]}"
        )
    raw_weight_bases = {
        name.removesuffix(".weight")
        for name in info_by_key
        if name.endswith(".weight")
    }
    fp8_bases = {
        module
        for module, target_format in target_formats.items()
        if target_format == "float-quantized"
    }
    expected_raw_bases = fp8_bases.union(ignored)
    if raw_weight_bases != expected_raw_bases:
        unexpected_raw = sorted(raw_weight_bases.difference(expected_raw_bases))
        missing_raw = sorted(expected_raw_bases.difference(raw_weight_bases))
        raise ValueError(
            f"Raw-weight coverage disagreement: unexpected={unexpected_raw[:5]}, "
            f"missing={missing_raw[:5]}"
        )
    scale_bases = {
        name.removesuffix(".weight_scale")
        for name in info_by_key
        if name.endswith(".weight_scale")
    }
    if scale_bases != set(targets):
        unexpected_scales = sorted(scale_bases.difference(targets))
        missing_scales = sorted(set(targets).difference(scale_bases))
        raise ValueError(
            f"Weight-scale coverage disagreement: unexpected={unexpected_scales[:5]}, "
            f"missing={missing_scales[:5]}"
        )
    actual_weight_bases = packed_bases.union(raw_weight_bases)
    expected_weight_bases = set(targets).union(ignored)
    if actual_weight_bases != expected_weight_bases:
        raise ValueError("Checkpoint weight modules are not exactly covered by target and ignore")

    tensor_data_bytes = sum(
        info.data_offsets[1] - info.data_offsets[0] for info in info_by_key.values()
    )
    metadata = index.get("metadata")
    indexed_total = metadata.get("total_size") if isinstance(metadata, dict) else None
    if not isinstance(indexed_total, int) or indexed_total != tensor_data_bytes:
        raise ValueError(
            f"Index total_size {indexed_total!r} differs from tensor data bytes {tensor_data_bytes}"
        )
    shard_file_bytes = sum(path.stat().st_size for path in shards)
    manifest_output_bytes = manifest.get("actual_output_bytes")
    if not isinstance(manifest_output_bytes, int) or manifest_output_bytes != shard_file_bytes:
        raise ValueError(
            "Manifest actual_output_bytes differs from the emitted safetensors file sizes"
        )

    shard_records = []
    for shard in shards:
        record: dict[str, object] = {
            "name": shard.name,
            "bytes": shard.stat().st_size,
            "tensor_count": sum(1 for filename in weight_map.values() if filename == shard.name),
        }
        if hash_shards:
            record["sha256"] = _sha256_file(shard)
        shard_records.append(record)
    asset_records: list[dict[str, object]] = []
    raw_assets = manifest.get("copied_assets", [])
    if not isinstance(raw_assets, list) or not all(isinstance(item, str) for item in raw_assets):
        raise TypeError("manifest copied_assets must be a list of paths")
    declared_assets = sorted(cast(list[str], raw_assets))
    if len(set(declared_assets)) != len(declared_assets):
        raise ValueError("manifest copied_assets contains duplicates")
    discovered_assets = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.name not in _NON_ASSET_NAMES
        and path.suffix.lower() in _MODEL_ASSET_SUFFIXES
    )
    if declared_assets != discovered_assets:
        missing_assets = sorted(set(declared_assets).difference(discovered_assets))
        unlisted_assets = sorted(set(discovered_assets).difference(declared_assets))
        raise ValueError(
            f"Manifest asset coverage differs from disk: missing={missing_assets[:5]}, "
            f"unlisted={unlisted_assets[:5]}"
        )
    for raw_asset in declared_assets:
        asset = root / raw_asset
        resolved_asset = asset.resolve()
        try:
            resolved_asset.relative_to(root.resolve())
        except ValueError as error:
            raise ValueError(f"Checkpoint asset escapes model directory: {raw_asset}") from error
        if not asset.is_file():
            raise FileNotFoundError(f"Checkpoint asset is missing: {asset}")
        asset_records.append(
            {
                "name": Path(raw_asset).as_posix(),
                "bytes": asset.stat().st_size,
                "sha256": _sha256_file(asset) if hash_shards else None,
            }
        )
    config_sha256 = _sha256_file(config_path)
    index_sha256 = _sha256_file(index_path)
    manifest_sha256 = _sha256_file(manifest_path)
    checkpoint_sha256 = (
        _sha256_bytes(
            _canonical_json(
                {
                    "config_sha256": config_sha256,
                    "index_sha256": index_sha256,
                    "manifest_sha256": manifest_sha256,
                    "shards": shard_records,
                    "assets": asset_records,
                }
            )
        )
        if hash_shards
        else None
    )
    return {
        "status": "passed",
        "model_dir": str(root),
        "config_sha256": config_sha256,
        "index_sha256": index_sha256,
        "manifest_sha256": manifest_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "format": checkpoint_format,
        "target_modules": len(targets),
        "ignored_modules": len(ignored),
        "tensor_count": len(info_by_key),
        "shard_count": len(shards),
        "tensor_data_bytes": tensor_data_bytes,
        "shard_file_bytes": shard_file_bytes,
        "shards_hashed": hash_shards,
        "shards": shard_records,
        "assets_hashed": hash_shards,
        "assets": asset_records,
    }


def _replace_hook_placeholders(
    argv: Sequence[str],
    *,
    model_dir: Path,
    output_dir: Path,
    artifact: Path,
    structure: Mapping[str, Any],
) -> tuple[str, ...]:
    replacements = {
        "{model_dir}": str(model_dir),
        "{output_dir}": str(output_dir),
        "{artifact}": str(artifact),
        "{config_sha256}": str(structure["config_sha256"]),
        "{index_sha256}": str(structure["index_sha256"]),
        "{manifest_sha256}": str(structure["manifest_sha256"]),
        "{checkpoint_sha256}": str(structure["checkpoint_sha256"]),
    }
    return tuple(replacements.get(argument, argument) for argument in argv)


def _hook_environment(extra_names: Sequence[str]) -> dict[str, str]:
    names = set(_SAFE_INHERITED_ENVIRONMENT).union(extra_names)
    return {name: value for name in names if (value := os.environ.get(name)) is not None}


def run_hook(
    hook: HookSpec,
    *,
    evidence_id: str,
    artifact: Path,
    model_dir: Path,
    output_dir: Path,
    structure: Mapping[str, Any],
) -> dict[str, Any]:
    """Run one evidence hook without a shell and capture immutable logs."""
    argv = _replace_hook_placeholders(
        hook.argv,
        model_dir=model_dir,
        output_dir=output_dir,
        artifact=artifact,
        structure=structure,
    )
    log_path = output_dir / "logs" / f"{evidence_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    timed_out = False
    return_code: int | None = None
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            argv,
            cwd=hook.cwd,
            env=_hook_environment(hook.inherit_environment),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            shell=False,
            start_new_session=True,
        )
        try:
            return_code = process.wait(timeout=hook.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                return_code = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                return_code = process.wait(timeout=10)
    elapsed = time.monotonic() - started
    result = {
        "argv": list(argv),
        "cwd": str(hook.cwd) if hook.cwd is not None else None,
        "elapsed_seconds": elapsed,
        "timeout_seconds": hook.timeout_seconds,
        "timed_out": timed_out,
        "return_code": return_code,
        "log": str(log_path),
        "log_sha256": _sha256_file(log_path),
    }
    return result


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_PATTERN.fullmatch(value) is not None


def _strict_equal(left: object, right: object) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left is right
    if isinstance(left, int | float) and isinstance(right, int | float):
        return float(left) == float(right)
    return type(left) is type(right) and left == right


def _validate_metric_summary(value: object, *, label: str) -> None:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    for field in ("mean", "median", "p95", "max"):
        metric = value.get(field)
        if (
            isinstance(metric, bool)
            or not isinstance(metric, int | float)
            or not math.isfinite(float(metric))
            or float(metric) < 0
        ):
            raise ValueError(f"{label}.{field} must be a finite non-negative number")


def _validate_evidence(document: dict[str, Any], spec: EvidenceSpec) -> None:
    actual_format = document.get("format")
    accepted = _EVIDENCE_FORMATS[spec.kind]
    if spec.expected_format is not None:
        accepted = frozenset({spec.expected_format})
    if accepted and actual_format not in accepted:
        raise ValueError(
            f"Evidence {spec.evidence_id!r} has format {actual_format!r}; "
            f"expected one of {sorted(accepted)}"
        )
    if not _is_sha256(document.get("protocol_sha256")):
        raise ValueError(f"Evidence {spec.evidence_id!r} has no valid protocol_sha256")
    if spec.kind == "perplexity":
        perplexity = document.get("perplexity")
        scored = document.get("scored_tokens")
        chunks = document.get("chunks")
        num_chunks = document.get("num_chunks")
        if (
            not isinstance(perplexity, int | float)
            or not math.isfinite(float(perplexity))
            or float(perplexity) <= 0
            or not isinstance(scored, int)
            or scored <= 0
            or not isinstance(num_chunks, int)
            or isinstance(num_chunks, bool)
            or num_chunks <= 1
            or not isinstance(chunks, list)
            or len(chunks) != num_chunks
            or not isinstance(document.get("dataset_revision"), str)
            or not document["dataset_revision"]
            or not _is_sha256(document.get("dataset_sha256"))
            or not _is_sha256(document.get("corpus_sha256"))
        ):
            raise ValueError(f"Evidence {spec.evidence_id!r} has invalid perplexity metrics")
        chunk_scored = 0
        chunk_nll = 0.0
        for expected_index, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                raise TypeError(f"Evidence {spec.evidence_id!r} has an invalid PPL chunk")
            value = chunk.get("scored_tokens")
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"Evidence {spec.evidence_id!r} has invalid chunk tokens")
            if not _is_sha256(chunk.get("text_sha256")) or not _is_sha256(
                chunk.get("token_sha256")
            ):
                raise ValueError(f"Evidence {spec.evidence_id!r} has invalid chunk hashes")
            if chunk.get("index") != expected_index:
                raise ValueError(f"Evidence {spec.evidence_id!r} PPL chunk indices disagree")
            nll = chunk.get("negative_log_likelihood")
            if (
                isinstance(nll, bool)
                or not isinstance(nll, int | float)
                or not math.isfinite(float(nll))
                or float(nll) < 0
            ):
                raise ValueError(f"Evidence {spec.evidence_id!r} has invalid chunk NLL")
            chunk_scored += value
            chunk_nll += float(nll)
        if chunk_scored != scored:
            raise ValueError(f"Evidence {spec.evidence_id!r} PPL token totals disagree")
        total_nll = document.get("negative_log_likelihood")
        mean_nll = document.get("mean_negative_log_likelihood")
        if (
            isinstance(total_nll, bool)
            or not isinstance(total_nll, int | float)
            or isinstance(mean_nll, bool)
            or not isinstance(mean_nll, int | float)
            or not math.isclose(float(total_nll), chunk_nll, rel_tol=1e-12, abs_tol=1e-9)
            or not math.isclose(
                float(mean_nll), chunk_nll / scored, rel_tol=1e-12, abs_tol=1e-12
            )
            or not math.isclose(
                float(perplexity), math.exp(float(mean_nll)), rel_tol=1e-12, abs_tol=1e-12
            )
        ):
            raise ValueError(f"Evidence {spec.evidence_id!r} PPL aggregates disagree")
    elif spec.kind == "divergence":
        candidates = document.get("candidates")
        protocol = document.get("context_protocol")
        if (
            not isinstance(candidates, dict)
            or not candidates
            or not isinstance(protocol, dict)
            or not _is_sha256(protocol.get("manifest_sha256"))
        ):
            raise ValueError(f"Evidence {spec.evidence_id!r} has no divergence candidates")
        for label, candidate in candidates.items():
            if not isinstance(label, str) or not isinstance(candidate, dict):
                raise TypeError(f"Evidence {spec.evidence_id!r} has an invalid candidate")
            contexts = candidate.get("contexts")
            metadata = candidate.get("metadata")
            if not isinstance(contexts, list) or not contexts or not isinstance(metadata, dict):
                raise ValueError(f"Divergence candidate {label!r} is incomplete")
            for metric in (
                "forward_kl_nats",
                "reverse_kl_nats",
                "jensen_shannon_nats",
                "total_variation",
            ):
                _validate_metric_summary(candidate.get(metric), label=f"{label}.{metric}")
            agreement = candidate.get("top1_agreement_rate")
            if (
                isinstance(agreement, bool)
                or not isinstance(agreement, int | float)
                or not 0 <= float(agreement) <= 1
            ):
                raise ValueError(f"Divergence candidate {label!r} has invalid top1 agreement")
    elif spec.kind == "serving":
        if actual_format == "mxwave-serving-qualification-collection-v1":
            results = document.get("results")
            if document.get("complete") is not True or not isinstance(results, list) or not results:
                raise ValueError(
                    f"Evidence {spec.evidence_id!r} is not a complete serving collection"
                )
        else:
            gates = document.get("gates")
            if not isinstance(gates, dict) or not all(
                isinstance(gates.get(field), bool)
                for field in ("pass", "token_parity", "long_context_semantic_accuracy")
            ):
                raise ValueError(
                    f"Evidence {spec.evidence_id!r} has invalid serving comparison gates"
                )
        if not _is_sha256(document.get("manifest_sha256")):
            raise ValueError(f"Evidence {spec.evidence_id!r} has no serving manifest hash")
    elif spec.kind == "runtime":
        runtime = document.get("runtime")
        if not isinstance(runtime, dict):
            raise ValueError(f"Evidence {spec.evidence_id!r} has no runtime identity")


def _validate_bindings(
    document: dict[str, Any],
    spec: EvidenceSpec,
    structure: Mapping[str, Any],
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for binding in spec.bindings:
        actual = _pointer(document, binding.pointer)
        expected = (
            binding.literal
            if binding.uses_literal
            else _pointer(structure, cast(str, binding.structure_pointer))
        )
        if not _strict_equal(actual, expected):
            raise ValueError(
                f"Evidence {spec.evidence_id!r} binding {binding.pointer!r} "
                f"is {actual!r}, expected {expected!r}"
            )
        results.append(
            {
                "pointer": binding.pointer,
                "structure_pointer": binding.structure_pointer,
                "expected": expected,
                "pass": True,
            }
        )
    return results


def _positive_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(float(value))
        and float(value) > 0
    )


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _runtime_capabilities(document: Mapping[str, Any]) -> dict[str, bool]:
    runtime = document.get("runtime")
    load = document.get("load")
    memory = document.get("memory")
    performance = document.get("performance")
    mtp = document.get("mtp")
    smoke = document.get("smoke")
    runtime = runtime if isinstance(runtime, dict) else {}
    load = load if isinstance(load, dict) else {}
    memory = memory if isinstance(memory, dict) else {}
    performance = performance if isinstance(performance, dict) else {}
    mtp = mtp if isinstance(mtp, dict) else {}
    smoke = smoke if isinstance(smoke, dict) else {}
    raw_samples = performance.get("samples")
    samples = (
        [item for item in raw_samples if isinstance(item, dict)]
        if isinstance(raw_samples, list)
        else []
    )
    image = runtime.get("image")
    pinned_image = (
        isinstance(image, str) and _DIGEST_PINNED_IMAGE_PATTERN.fullmatch(image) is not None
    )
    kernel_backend = load.get("kernel_backend")
    stock_load = (
        runtime.get("name") == "vllm"
        and runtime.get("stock") is True
        and pinned_image
        and load.get("status") == "passed"
        and isinstance(kernel_backend, str)
        and bool(kernel_backend.strip())
    )
    measured_samples = [
        sample
        for sample in samples
        if _positive_int(sample.get("concurrency"))
        and _positive_number(sample.get("ttft_seconds"))
        and _positive_number(sample.get("output_tokens_per_second"))
    ]
    ttft = bool(measured_samples)
    decode = bool(measured_samples)
    concurrency = any(int(sample["concurrency"]) > 1 for sample in measured_samples)
    proposed = mtp.get("proposed_tokens")
    accepted = mtp.get("accepted_tokens")
    mtp_acceptance = (
        mtp.get("enabled") is True
        and isinstance(proposed, int)
        and not isinstance(proposed, bool)
        and proposed > 0
        and isinstance(accepted, int)
        and not isinstance(accepted, bool)
        and 0 < accepted <= proposed
        and mtp.get("token_parity") is True
    )
    return {
        "stock_vllm_load": stock_load,
        "fresh_runtime_cache_startup": _positive_number(
            load.get("fresh_runtime_cache_startup_seconds")
        ),
        "warm_restart": _positive_number(load.get("warm_restart_seconds")),
        "peak_host_memory": _positive_number(memory.get("peak_host_used_bytes")),
        "peak_accelerator_memory": _positive_number(
            memory.get("peak_accelerator_bytes")
        ),
        "ttft": ttft,
        "decode_throughput": decode,
        "concurrency": concurrency,
        "mtp_acceptance": mtp_acceptance,
        "critical_smoke": smoke.get("pass") is True,
        "colocated_runtime_client": runtime.get("client_location") == "on-device",
    }


def _load_evidence(
    spec: EvidenceSpec, structure: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, object]]:
    document = _read_json_object(spec.path)
    _validate_evidence(document, spec)
    bindings = _validate_bindings(document, spec, structure)
    return document, {
        "id": spec.evidence_id,
        "kind": spec.kind,
        "path": str(spec.path),
        "sha256": _sha256_file(spec.path),
        "format": document.get("format"),
        "required": spec.required,
        "status": "present",
        "bindings": bindings,
    }


def _pointer(document: object, pointer: str) -> object:
    if not pointer:
        return document
    current = document
    for raw_part in pointer.removeprefix("/").split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            raise KeyError(f"JSON pointer {pointer!r} is missing at {part!r}")
    return current


def _compare(actual: object, operator: GateOperator, threshold: object) -> bool:
    if operator in {"==", "!="}:
        equal = _strict_equal(actual, threshold)
        return equal if operator == "==" else not equal
    if (
        isinstance(actual, bool)
        or isinstance(threshold, bool)
        or not isinstance(actual, int | float)
        or not isinstance(threshold, int | float)
    ):
        raise TypeError("Ordered qualification gates require numeric values")
    left = float(actual)
    right = float(threshold)
    if not math.isfinite(left) or not math.isfinite(right):
        raise ValueError("Qualification gates require finite numeric values")
    if operator == "<":
        return left < right
    if operator == "<=":
        return left <= right
    if operator == ">":
        return left > right
    return left >= right


def _capabilities(
    structure: Mapping[str, Any],
    evidence: Mapping[str, tuple[EvidenceKind, dict[str, Any]]],
) -> dict[str, bool]:
    result = {"structural": structure.get("status") == "passed"}
    for kind, document in evidence.values():
        inferred: dict[str, bool] = {}
        if kind == "perplexity":
            inferred["perplexity"] = True
        elif kind == "divergence":
            inferred["distribution_divergence"] = True
            protocol = document.get("context_protocol")
            inferred["multi_position_divergence"] = (
                isinstance(protocol, dict)
                and isinstance(protocol.get("positions_per_context"), int)
                and int(protocol["positions_per_context"]) > 1
            )
        elif kind == "serving":
            gates = document.get("gates")
            inferred["critical_smoke"] = (
                isinstance(gates, dict) and gates.get("pass") is True
            )
            inferred["mtp_token_parity"] = (
                isinstance(gates, dict) and gates.get("token_parity") is True
            )
            inferred["long_context"] = (
                isinstance(gates, dict)
                and gates.get("long_context_semantic_accuracy") is True
            )
        elif kind == "runtime":
            inferred.update(_runtime_capabilities(document))
        for name, value in inferred.items():
            result[name] = result.get(name, False) or value
    return dict(sorted(result.items()))


def _render_markdown(report: Mapping[str, Any]) -> str:
    decision = report.get("decision", "unknown")
    lines = [
        "# MxWave qualification report",
        "",
        f"- Run: `{report.get('run_id', 'unknown')}`",
        f"- Model: `{report.get('model_label', 'unknown')}`",
        f"- Decision: **{decision}**",
        f"- Complete: `{str(report.get('complete', False)).lower()}`",
        "",
        "## Gates",
        "",
        "| Gate | Actual | Operator | Threshold | Result |",
        "|---|---:|:---:|---:|:---:|",
    ]
    raw_gates = report.get("gates")
    if isinstance(raw_gates, list):
        for raw in raw_gates:
            if not isinstance(raw, dict):
                continue
            lines.append(
                f"| {raw.get('id')} | `{raw.get('actual')}` | {raw.get('operator')} | "
                f"`{raw.get('threshold')}` | {'pass' if raw.get('pass') else 'FAIL'} |"
            )
    lines.extend(
        [
            "",
            "## Required capabilities",
            "",
            "| Capability | Result |",
            "|---|:---:|",
        ]
    )
    required = report.get("required_capabilities")
    capabilities = report.get("capabilities")
    if isinstance(required, list) and isinstance(capabilities, dict):
        for name in required:
            lines.append(f"| {name} | {'pass' if capabilities.get(name) else 'MISSING'} |")
    lines.extend(
        [
            "",
            "A missing capability makes the result incomplete; it is never treated as a pass.",
            "",
        ]
    )
    return "\n".join(lines)


def run_qualification(
    spec_path: str | Path,
    output_dir: str | Path,
    *,
    resume: bool = False,
) -> tuple[dict[str, Any], Path]:
    """Run a qualification contract and atomically emit JSON and Markdown reports."""
    spec, spec_sha256, raw_spec = load_spec(spec_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "qualification.json"
    markdown_path = output / "qualification.md"
    if report_path.exists():
        if not resume:
            raise FileExistsError(f"Qualification report already exists: {report_path}")
        existing = _read_json_object(report_path)
        if existing.get("spec_sha256") != spec_sha256:
            raise ValueError("Existing qualification report belongs to a different spec")

    started_at = _utc_now()
    structure = verify_checkpoint(spec.model_dir, hash_shards=spec.hash_shards)
    documents: dict[str, dict[str, Any]] = {}
    typed_documents: dict[str, tuple[EvidenceKind, dict[str, Any]]] = {}
    evidence_records: list[dict[str, object]] = []
    hook_records: dict[str, object] = {}
    missing_required: list[str] = []
    errors: list[dict[str, str]] = []
    for item in spec.evidence:
        try:
            should_run_hook = item.hook is not None and (not resume or not item.path.is_file())
            if item.hook is not None and resume and item.path.is_file():
                try:
                    _load_evidence(item, structure)
                except (KeyError, OSError, TypeError, ValueError):
                    should_run_hook = True
            if item.hook is not None and should_run_hook:
                item.path.unlink(missing_ok=True)
                hook_record = run_hook(
                    item.hook,
                    evidence_id=item.evidence_id,
                    artifact=item.path,
                    model_dir=spec.model_dir,
                    output_dir=output,
                    structure=structure,
                )
                hook_records[item.evidence_id] = hook_record
                if hook_record["timed_out"]:
                    raise RuntimeError(
                        f"Evidence hook {item.evidence_id!r} timed out after "
                        f"{hook_record['elapsed_seconds']:.1f}s"
                    )
                if hook_record["return_code"] != 0:
                    raise RuntimeError(
                        f"Evidence hook {item.evidence_id!r} failed with exit code "
                        f"{hook_record['return_code']}"
                    )
            if not item.path.is_file():
                if item.required:
                    missing_required.append(item.evidence_id)
                evidence_records.append(
                    {
                        "id": item.evidence_id,
                        "kind": item.kind,
                        "path": str(item.path),
                        "required": item.required,
                        "status": "missing",
                    }
                )
                continue
            document, record = _load_evidence(item, structure)
            documents[item.evidence_id] = document
            typed_documents[item.evidence_id] = (item.kind, document)
            evidence_records.append(record)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            if item.required:
                missing_required.append(item.evidence_id)
            errors.append(
                {
                    "evidence": item.evidence_id,
                    "type": type(error).__name__,
                    "message": str(error),
                }
            )
            evidence_records.append(
                {
                    "id": item.evidence_id,
                    "kind": item.kind,
                    "path": str(item.path),
                    "required": item.required,
                    "status": "invalid",
                }
            )

    checkpoint_stable = False
    try:
        post_structure = verify_checkpoint(spec.model_dir, hash_shards=True)
        checkpoint_stable = _strict_equal(
            post_structure.get("checkpoint_sha256"), structure.get("checkpoint_sha256")
        )
        if not checkpoint_stable:
            errors.append(
                {
                    "evidence": "structure",
                    "type": "CheckpointChanged",
                    "message": "checkpoint fingerprint changed while evidence was collected",
                }
            )
    except (OSError, TypeError, ValueError) as error:
        errors.append(
            {
                "evidence": "structure",
                "type": type(error).__name__,
                "message": f"post-evidence checkpoint verification failed: {error}",
            }
        )

    gate_results: list[dict[str, object]] = []
    gate_failure = False
    gate_incomplete = False
    for gate in spec.gates:
        source: object = structure if gate.evidence_id == "structure" else documents.get(
            gate.evidence_id
        )
        try:
            if source is None:
                raise KeyError(f"evidence {gate.evidence_id!r} is unavailable")
            actual = _pointer(source, gate.pointer)
            passed = _compare(actual, gate.operator, gate.threshold)
            gate_failure = gate_failure or not passed
            gate_results.append(
                {
                    "id": gate.gate_id,
                    "evidence": gate.evidence_id,
                    "pointer": gate.pointer,
                    "actual": actual,
                    "operator": gate.operator,
                    "threshold": gate.threshold,
                    "pass": passed,
                }
            )
        except (KeyError, TypeError, ValueError) as error:
            gate_incomplete = True
            gate_results.append(
                {
                    "id": gate.gate_id,
                    "evidence": gate.evidence_id,
                    "pointer": gate.pointer,
                    "operator": gate.operator,
                    "threshold": gate.threshold,
                    "pass": False,
                    "error": str(error),
                }
            )

    capabilities = _capabilities(structure, typed_documents)
    missing_capabilities = [
        name for name in spec.required_capabilities if not capabilities.get(name, False)
    ]
    complete = (
        not missing_required and not missing_capabilities and not errors and not gate_incomplete
    )
    decision: Decision
    if gate_failure:
        decision = "rejected"
    elif not complete:
        decision = "incomplete"
    else:
        decision = spec.success_decision
    report: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "run_id": spec.run_id,
        "model_label": spec.model_label,
        "model_dir": str(spec.model_dir),
        "started_at": started_at,
        "finished_at": _utc_now(),
        "spec": str(Path(spec_path)),
        "spec_sha256": spec_sha256,
        "spec_format": raw_spec.get("format"),
        "structure": structure,
        "checkpoint_stable": checkpoint_stable,
        "evidence": evidence_records,
        "hooks": hook_records,
        "gates": gate_results,
        "capabilities": capabilities,
        "required_capabilities": list(spec.required_capabilities),
        "missing_capabilities": missing_capabilities,
        "missing_required_evidence": sorted(set(missing_required)),
        "errors": errors,
        "complete": complete,
        "decision": decision,
    }
    _write_json_atomic(report_path, report)
    _write_text_atomic(markdown_path, _render_markdown(report))
    return report, report_path


__all__ = [
    "REPORT_FORMAT",
    "RUNTIME_EVIDENCE_FORMAT",
    "SPEC_FORMAT",
    "QualificationSpec",
    "load_spec",
    "run_qualification",
    "verify_checkpoint",
]
