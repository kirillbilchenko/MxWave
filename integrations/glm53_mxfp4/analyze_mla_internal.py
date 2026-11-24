"""Independently inspect raw MLA operands using NumPy; no Torch or GPU initialization."""

from __future__ import annotations

import gzip
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

MAX_BYTES = 16 * 1024**2
SIZES = {
    "torch.bfloat16": 2,
    "torch.float32": 4,
    "torch.float16": 2,
    "torch.float64": 8,
    "torch.int64": 8,
    "torch.int32": 4,
    "torch.int16": 2,
    "torch.int8": 1,
    "torch.uint8": 1,
    "torch.bool": 1,
    "torch.float8_e4m3fn": 1,
}
EVENTS = [
    "mla/input",
    "kv_write/input",
    "kv_write/before",
    "kv_write/after",
    "query_bmm/input",
    "query_bmm/output",
    "sparse_convert/input",
    "sparse_convert/output",
    "attention/input",
    "attention/selected_cache",
    "attention/output",
    "value_bmm/input",
    "value_bmm/output",
    "mla/output",
]


def read_artifact(directory: Path, artifact: dict[str, Any]) -> bytes:
    """Check both compressed and exact raw bytes of a bounded, explicit tensor basename."""
    name = artifact["path"]
    if Path(name).name != name or not name.endswith(".bin.gz"):
        raise ValueError("Artifact must be an owned basename")
    path = directory / name
    compressed = path.read_bytes()
    if (
        path.is_symlink()
        or len(compressed) != artifact["compressed_bytes"]
        or hashlib.sha256(compressed).hexdigest() != artifact["compressed_sha256"]
    ):
        raise ValueError("Compressed artifact changed")
    with gzip.open(path, "rb") as handle:
        raw = handle.read(MAX_BYTES + 1)
    expected = math.prod(artifact["shape"]) * SIZES[artifact["dtype"]]
    if (
        len(raw) > MAX_BYTES
        or len(raw) != expected
        or len(raw) != artifact["bytes"]
        or hashlib.sha256(raw).hexdigest() != artifact["sha256"]
    ):
        raise ValueError("Raw artifact shape, dtype or bytes changed")
    return raw


def raw_metrics(current: bytes, reference: bytes, dtype: str, *, numerical: bool) -> dict[str, Any]:
    """Recompute exact bit changes and finite BF16/F32 errors without the recorder's math."""
    size = SIZES[dtype]
    if len(current) != len(reference) or len(current) > MAX_BYTES or len(current) % size:
        raise ValueError("Require equally sized bounded logical tensors")
    changed = np.frombuffer(current, np.uint8) != np.frombuffer(reference, np.uint8)
    count = len(current) // size
    value: dict[str, Any] = {
        "bytes_exact": not bool(changed.any()),
        "changed_bytes": int(changed.sum()),
        "elements": count,
        "changed_elements_by_bits": int(changed.reshape(-1, size).any(axis=1).sum()),
    }
    if not numerical:
        return value
    if dtype == "torch.bfloat16":
        arrays = [
            (np.frombuffer(raw, "<u2").astype(np.uint32) << 16).view("<f4").astype(np.float64)
            for raw in (current, reference)
        ]
    elif dtype == "torch.float32":
        arrays = [np.frombuffer(raw, "<f4").astype(np.float64) for raw in (current, reference)]
    else:
        raise ValueError("Numerical operands must use the recorded BF16/F32 metric contract")
    left, right = arrays
    finite = np.isfinite(left) & np.isfinite(right)
    pairs = int(finite.sum())
    error = left[finite] - right[finite]
    energy = float(np.dot(error, error))
    reference_energy = float(np.dot(right[finite], right[finite]))
    value.update(
        finite_pairs=pairs,
        nonfinite_pairs=count - pairs,
        max_absolute_error_finite=float(np.max(np.abs(error))) if pairs else None,
        rms_error_finite=math.sqrt(energy / pairs) if pairs else None,
        reference_rms_finite=math.sqrt(reference_energy / pairs) if pairs else None,
        relative_l2_finite=math.sqrt(energy / reference_energy)
        if reference_energy
        else (0.0 if not energy else None),
        sqnr_db_finite=10 * math.log10(reference_energy / energy)
        if energy and reference_energy
        else None,
    )
    return value


def verify_metrics(actual: dict[str, Any], recorded: dict[str, Any]) -> None:
    """Keep all discrete results exact; allow only CPU float64 reduction roundoff."""
    if set(actual) != set(recorded) - {"metric_kind"}:
        raise ValueError("Metric fields differ")
    for name, expected in actual.items():
        observed = recorded[name]
        if isinstance(expected, float):
            if not isinstance(observed, (int, float)) or not math.isclose(
                expected, observed, rel_tol=1e-10, abs_tol=1e-9
            ):
                raise ValueError("Recorded finite error metric changed: " + name)
        elif type(expected) is not type(observed) or expected != observed:
            raise ValueError("Recorded bit/count decision changed: " + name)


def analyze_worker(worker: dict[str, Any], directory: Path) -> dict[str, Any]:
    """Locate the first changed actual operand/result in every one of five fixed replays."""
    if worker["status"] != "complete" or len(worker["replays"]) != 5:
        raise ValueError("Require five complete fixed replays")
    if worker["served_iteration"] != 4 or not worker["final_state_not_rewound"]:
        raise ValueError("Fifth fixed replay must supply the output")
    private = worker["workspace_policy"] == "restore_private"
    references: list[list[bytes]] = []
    reference_descriptors: list[list[dict[str, Any]]] = []
    hidden_reference: list[bytes] = []
    rows = []
    for iteration, replay in enumerate(worker["replays"]):
        if replay["index"] != iteration or not replay["initial_bytes_exact"]:
            raise ValueError("Initial complete byte restore failed")
        observation = replay["observation"]
        if private and not observation["private_before_original_exact"]:
            raise ValueError("Declared private restore failed")
        if observation["constants_after"] != worker["constants_before"]:
            raise ValueError("Actual BMM weights or original layout changed")
        events = observation["mla_internal_events"]
        if [event["name"] for event in events] != EVENTS:
            raise ValueError("Actual MLA operation chain is incomplete")
        first = None
        results = []
        for event_index, event in enumerate(events):
            if iteration == 0:
                references.append([])
                reference_descriptors.append([])
            elif len(event["tensors"]) != len(references[event_index]):
                raise ValueError("Internal operand count changed between fixed replays")
            tensors = []
            for tensor_index, tensor in enumerate(event["tensors"]):
                descriptor = {
                    "path": tensor["path"],
                    "shape": tensor["artifact"]["shape"],
                    "dtype": tensor["artifact"]["dtype"],
                }
                if iteration == 0:
                    reference_descriptors[event_index].append(descriptor)
                elif descriptor != reference_descriptors[event_index][tensor_index]:
                    raise ValueError("Internal operand path, shape or dtype changed")
                raw = read_artifact(directory, tensor["artifact"])
                if iteration == 0:
                    references[event_index].append(raw)
                metric = raw_metrics(
                    raw,
                    references[event_index][tensor_index],
                    tensor["artifact"]["dtype"],
                    numerical=tensor["metrics"]["metric_kind"] == "numerical",
                )
                verify_metrics(metric, tensor["metrics"])
                tensors.append({"path": tensor["path"], **metric})
                if first is None and not metric["bytes_exact"]:
                    first = event["name"]
            results.append({"name": event["name"], "tensors": tensors})
        if first != observation["first_changed_internal_event"]:
            raise ValueError("First changed internal event disagrees with retained raw bytes")
        hidden = []
        for index, tensor in enumerate(observation["hidden"]):
            raw = read_artifact(directory, tensor["artifact"])
            if iteration == 0:
                hidden_reference.append(raw)
            metric = raw_metrics(
                raw, hidden_reference[index], tensor["artifact"]["dtype"], numerical=True
            )
            verify_metrics(metric, tensor["metrics"])
            hidden.append(metric)
        active = observation["active_kda_post"]
        if len(active) != 68 or active != sorted(
            active, key=lambda x: (x["layer"], x["cache_index"])
        ):
            raise ValueError("Missing or unordered active KDA state observations")
        for item in active:
            if item["metrics"]["bytes_exact"] != (
                item["current"]["sha256"] == item["reference"]["sha256"]
            ):
                raise ValueError("Active-state bit decision disagrees with complete logical hashes")
        changed = next((item for item in active if not item["metrics"]["bytes_exact"]), None)
        saved = observation["first_changed_active_cache"]
        if bool(changed) != bool(saved):
            raise ValueError("Earliest active state evidence absent")
        if saved:
            if (saved["layer"], saved["cache_index"]) != (changed["layer"], changed["cache_index"]):
                raise ValueError("First changed active layer differs")
            raw = [read_artifact(directory, saved[key]) for key in ("current", "reference")]
            metric = raw_metrics(*raw, saved["current"]["dtype"], numerical=True)
            verify_metrics(metric, changed["metrics"])
        rows.append(
            {
                "iteration": iteration,
                "first_changed_internal_event": first,
                "events": results,
                "hidden": hidden,
                "first_changed_active_layer": changed["layer"] if changed else None,
                "private_before_original_exact": observation["private_before_original_exact"],
            }
        )
    return {
        "rank": worker["rank"],
        "label": worker["label"],
        "workspace_policy": worker["workspace_policy"],
        "replays": rows,
        "private_workspace_image_limit": "Temporary full private images are runtime byte-checked, not archived.",
        "qualification_attempted": False,
        "quality_qualified": False,
    }
