"""Break down retained full-vocabulary KL by prefix without running inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from evaluate_quantization_controls import _load_protocol, _write_json
from safetensors import safe_open


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def _paired_summary(values: np.ndarray, samples: np.ndarray) -> dict[str, Any]:
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Expected finite values, one per independent window")
    return {
        "mean": float(values.mean()),
        "ci95": np.percentile(values[samples].mean(axis=1), [2.5, 97.5]).tolist(),
        "window_count": len(values),
    }


def _prefix_rows(contexts: list[dict[str, Any]]) -> tuple[list[int], list[int], np.ndarray]:
    windows = sorted({item["window_index"] for item in contexts})
    prefixes = sorted({item["prefix_tokens"] for item in contexts})
    mapping = {(item["window_index"], item["prefix_tokens"]): i for i, item in enumerate(contexts)}
    if len(mapping) != len(contexts) or len(contexts) != len(windows) * len(prefixes):
        raise ValueError("Contexts must contain exactly one row for every window/prefix pair")
    try:
        indices = np.array([[mapping[window, prefix] for window in windows] for prefix in prefixes])
    except KeyError as error:
        raise ValueError("Missing window/prefix pair") from error
    return windows, prefixes, indices


def _read_metrics(
    reference: Path, candidate: Path, *, protocol_sha: str, positions: int
) -> tuple[np.ndarray, np.ndarray]:
    with (
        safe_open(str(reference), framework="pt", device="cpu") as teacher_file,
        safe_open(str(candidate), framework="pt", device="cpu") as candidate_file,
    ):
        teacher = teacher_file.get_slice("logprobs")
        student = candidate_file.get_slice("logprobs")
        if (
            teacher.get_shape() != student.get_shape()
            or len(teacher.get_shape()) != 2
            or teacher.get_shape()[0] != positions
            or any(
                (source.metadata() or {}).get("protocol_sha256") != protocol_sha
                for source in (teacher_file, candidate_file)
            )
        ):
            raise ValueError("Distribution shape or protocol identity mismatch")
        divergence: list[float] = []
        agreement: list[bool] = []
        for start in range(0, positions, 16):
            left = teacher[start : start + 16].double()
            right = student[start : start + 16].double()
            if not torch.isfinite(left).all() or not torch.isfinite(right).all():
                raise ValueError("Non-finite log probabilities")
            if any(
                bool((torch.logsumexp(matrix, dim=1).abs() > 0.02).any())
                for matrix in (left, right)
            ):
                raise ValueError("Log probabilities are not normalized")
            divergence.extend((left.exp() * (left - right)).sum(dim=1).tolist())
            agreement.extend((left.argmax(dim=1) == right.argmax(dim=1)).tolist())
    return np.array(divergence), np.array(agreement)


def analyze(args: argparse.Namespace) -> None:
    """Verify retained evidence and resample independent windows within each prefix."""
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite analysis: {output}")
    if args.bootstrap_iterations <= 0:
        raise ValueError("Bootstrap iterations must be positive")
    protocol, digest = _load_protocol(Path(args.protocol))
    root = Path(args.measurements)
    windows, prefixes, indices = _prefix_rows(protocol["kl_contexts"])
    labels = ["bf16", *args.labels]
    if len(set(labels)) != len(labels) or "h64" not in labels:
        raise ValueError("Provide unique candidate labels including h64, excluding bf16")
    provenance = {}
    for label in labels:
        path = root / f"{label}.json"
        report = json.loads(path.read_text())
        matrix = root / f"{label}-logprobs.safetensors"
        if (
            not report.get("complete")
            or report["label"] != label
            or report["protocol_sha256"] != digest
            or report["executed_source_sha256"] != protocol["executed_source_sha256"]
            or _hash_file(matrix) != report["distributions"]["sha256"]
        ):
            raise ValueError(f"Retained measurement identity mismatch: {label}")
        provenance[label] = {
            "report_sha256": _hash_file(path),
            "distributions_sha256": report["distributions"]["sha256"],
            "checkpoint_sha256": report["checkpoint_sha256"],
        }
    rng = np.random.default_rng(args.seed)
    samples = rng.integers(0, len(windows), size=(args.bootstrap_iterations, len(windows)))
    kl, top1 = {}, {}
    for label in args.labels:
        kl[label], top1[label] = _read_metrics(
            root / "bf16-logprobs.safetensors",
            root / f"{label}-logprobs.safetensors",
            protocol_sha=digest,
            positions=len(protocol["kl_contexts"]),
        )
    result: dict[str, Any] = {
        "format": "mxwave-retained-kl-prefix-analysis-v1",
        "complete": True,
        "protocol_sha256": digest,
        "collection_source_sha256": protocol["executed_source_sha256"],
        "analysis_source_sha256": _hash_file(Path(__file__)),
        "provenance": provenance,
        "scope": "exploratory breakdown of already observed validation data; no new inference",
        "bootstrap": {
            "unit": "whole window; identical sampled windows for every prefix and model",
            "iterations": args.bootstrap_iterations,
            "seed": args.seed,
            "intervals": "nominal 95%; not corrected for multiple comparisons",
        },
        "windows": windows,
        "prefixes": [],
        "longest_minus_shortest": {},
        "global": {},
    }
    for prefix, rows in zip(prefixes, indices, strict=True):
        result["prefixes"].append(
            {
                "prefix_tokens": prefix,
                "results": {
                    label: {
                        "forward_kl": _paired_summary(kl[label][rows], samples),
                        "top1_agreement_count": int(top1[label][rows].sum()),
                    }
                    for label in args.labels
                },
                "h64_minus_control": {
                    label: _paired_summary(kl["h64"][rows] - kl[label][rows], samples)
                    for label in args.labels
                    if label != "h64"
                },
            }
        )
    for label in args.labels:
        result["longest_minus_shortest"][label] = _paired_summary(
            kl[label][indices[-1]] - kl[label][indices[0]], samples
        )
        result["global"][label] = {
            "mean_forward_kl": float(kl[label].mean()),
            "top1_agreement_count": int(top1[label].sum()),
            "kl_per_position": kl[label].tolist(),
        }
    _write_json(output, result)
    print(json.dumps({"prefixes": result["prefixes"]}))


def main() -> None:
    """Parse paths to retained distributions and emit a compact derived report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--measurements", required=True)
    parser.add_argument("--labels", nargs="+", default=["h64", "amd", "mse", "rtn", "norm"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261004)
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
