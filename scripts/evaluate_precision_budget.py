"""Select and validate precision candidates against a frozen reference and baseline."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file


def _metadata(path: Path) -> dict[str, str]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return dict(handle.metadata() or {})


def _read_json_object(path: Path) -> dict[str, Any]:
    value: Any = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return cast(dict[str, Any], value)


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _selected_context_digest(contexts_path: Path, count: int) -> str:
    manifest = _read_json_object(contexts_path)
    raw_contexts = manifest.get("contexts")
    if not isinstance(raw_contexts, list) or count > len(raw_contexts):
        raise ValueError("Context manifest has too few contexts")
    digests: list[str] = []
    for raw_context in raw_contexts[:count]:
        if not isinstance(raw_context, dict):
            raise TypeError("Context manifest contains a non-object context")
        digest = raw_context.get("token_ids_sha256")
        if not isinstance(digest, str):
            raise TypeError("Context manifest is missing a token digest")
        digests.append(digest)
    return hashlib.sha256("".join(digests).encode()).hexdigest()


def _load_logprobs(
    path: Path,
    *,
    expected_contexts_sha256: str,
    expected_rows: int | None = None,
) -> tuple[torch.Tensor, dict[str, str]]:
    metadata = _metadata(path)
    if metadata.get("contexts_manifest_sha256") != expected_contexts_sha256:
        raise ValueError(f"{path}: context-manifest hash mismatch")
    values = load_file(path)["logprobs"]
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(f"{path}: expected a non-empty logprob matrix")
    if expected_rows is not None and values.shape[0] != expected_rows:
        raise ValueError(f"{path}: expected {expected_rows} rows, found {values.shape[0]}")
    if metadata.get("num_contexts") != str(values.shape[0]):
        raise ValueError(f"{path}: metadata row count mismatch")
    return values, metadata


def _forward_kl_rows(reference: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError("Reference and candidate logprob matrices must have equal shapes")
    ref = reference.to(torch.float64)
    cand = candidate.to(torch.float64)
    ref -= torch.logsumexp(ref, dim=1, keepdim=True)
    cand -= torch.logsumexp(cand, dim=1, keepdim=True)
    values = torch.sum(torch.exp(ref) * (ref - cand), dim=1)
    return torch.clamp(values, min=0.0)


def _bootstrap_mean_ci(
    values: list[float],
    *,
    seed: int,
    iterations: int,
) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=np.float64)
    for start in range(0, iterations, 1000):
        count = min(1000, iterations - start)
        indices = generator.integers(0, len(array), size=(count, len(array)))
        means[start : start + count] = array[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return [float(lower), float(upper)]


def _mean(values: torch.Tensor) -> float:
    return float(values.mean().item())


def _candidate_path(candidate_dir: Path, bucket_name: str) -> Path:
    return candidate_dir / f"{bucket_name}-logprobs.safetensors"


def _bucket_records(plan: dict[str, Any]) -> list[dict[str, Any]]:
    raw_buckets = plan.get("buckets")
    if not isinstance(raw_buckets, list):
        raise TypeError("Precision plan has no bucket list")
    buckets: list[dict[str, Any]] = []
    for raw_bucket in raw_buckets:
        if not isinstance(raw_bucket, dict):
            raise TypeError("Precision plan contains a non-object bucket")
        name = raw_bucket.get("name")
        premium = raw_bucket.get("premium_bytes")
        modules = raw_bucket.get("selected_modules")
        if (
            not isinstance(name, str)
            or not isinstance(premium, int)
            or premium <= 0
            or not isinstance(modules, list)
            or not all(isinstance(module, str) for module in modules)
        ):
            raise ValueError("Precision plan contains an invalid bucket")
        buckets.append(cast(dict[str, Any], raw_bucket))
    return buckets


def _selection_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("select", help="rank fixed single-bucket screens")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--contexts", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split-size", type=int, default=16)
    parser.add_argument("--max-selected-buckets", type=int, default=4)
    parser.add_argument("--max-premium-bytes", type=int, default=1024**3)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260918)
    parser.set_defaults(handler=select_candidates)


def _final_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("final", help="score one combined candidate on holdout")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--contexts", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--selection-contexts", type=int, default=32)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260918)
    parser.set_defaults(handler=validate_final_candidate)


def build_parser() -> argparse.ArgumentParser:
    """Build the precision-budget evaluator parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    _selection_parser(subparsers)
    _final_parser(subparsers)
    return parser


def select_candidates(args: argparse.Namespace) -> Path:
    """Rank single-bucket candidates on two disjoint fixed splits."""
    if args.split_size <= 0:
        raise ValueError("--split-size must be positive")
    if args.max_selected_buckets <= 0:
        raise ValueError("--max-selected-buckets must be positive")
    if args.max_premium_bytes <= 0:
        raise ValueError("--max-premium-bytes must be positive")
    if args.bootstrap_iterations <= 0:
        raise ValueError("--bootstrap-iterations must be positive")

    contexts_path = Path(args.contexts)
    contexts_sha256 = _sha256_file(contexts_path)
    screen_rows = args.split_size * 2
    expected_selected_digest = _selected_context_digest(contexts_path, screen_rows)
    reference, _reference_metadata = _load_logprobs(
        Path(args.reference),
        expected_contexts_sha256=contexts_sha256,
    )
    baseline, _baseline_metadata = _load_logprobs(
        Path(args.baseline),
        expected_contexts_sha256=contexts_sha256,
    )
    if reference.shape != baseline.shape or reference.shape[0] < screen_rows:
        raise ValueError("Frozen reference/baseline matrices cannot supply the screen")
    reference_screen = reference[:screen_rows]
    baseline_screen = baseline[:screen_rows]
    baseline_kl = _forward_kl_rows(reference_screen, baseline_screen)

    plan_path = Path(args.plan)
    plan = _read_json_object(plan_path)
    if plan.get("format") != "mxwave-precision-budget-plan-v1":
        raise ValueError("Unsupported precision-budget plan")
    candidate_dir = Path(args.candidate_dir)
    records: list[dict[str, Any]] = []
    for bucket in _bucket_records(plan):
        name = cast(str, bucket["name"])
        candidate_path = _candidate_path(candidate_dir, name)
        candidate, metadata = _load_logprobs(
            candidate_path,
            expected_contexts_sha256=contexts_sha256,
            expected_rows=screen_rows,
        )
        if metadata.get("selected_context_sha256") != expected_selected_digest:
            raise ValueError(f"{name}: selected context digest mismatch")
        candidate_kl = _forward_kl_rows(reference_screen, candidate)
        delta = candidate_kl - baseline_kl
        first = delta[: args.split_size]
        second = delta[args.split_size :]
        first_mean = _mean(first)
        second_mean = _mean(second)
        pooled_mean = _mean(delta)
        premium_bytes = cast(int, bucket["premium_bytes"])
        eligible = first_mean < 0.0 and second_mean < 0.0
        conservative_gain = min(-first_mean, -second_mean) if eligible else 0.0
        efficiency = conservative_gain / (premium_bytes / 1024**3)
        records.append(
            {
                "name": name,
                "artifact": str(candidate_path),
                "artifact_sha256": _sha256_file(candidate_path),
                "premium_bytes": premium_bytes,
                "selected_modules": bucket["selected_modules"],
                "baseline_forward_kl_nats": {
                    "split_a_mean": _mean(baseline_kl[: args.split_size]),
                    "split_b_mean": _mean(baseline_kl[args.split_size :]),
                    "pooled_mean": _mean(baseline_kl),
                },
                "candidate_forward_kl_nats": {
                    "split_a_mean": _mean(candidate_kl[: args.split_size]),
                    "split_b_mean": _mean(candidate_kl[args.split_size :]),
                    "pooled_mean": _mean(candidate_kl),
                },
                "candidate_minus_baseline_forward_kl_nats": {
                    "split_a_mean": first_mean,
                    "split_b_mean": second_mean,
                    "pooled_mean": pooled_mean,
                    "bootstrap_95_ci": _bootstrap_mean_ci(
                        [float(value) for value in delta.tolist()],
                        seed=args.bootstrap_seed,
                        iterations=args.bootstrap_iterations,
                    ),
                    "candidate_lower_contexts": int((delta < 0).sum().item()),
                    "ties": int((delta == 0).sum().item()),
                    "baseline_lower_contexts": int((delta > 0).sum().item()),
                },
                "eligible": eligible,
                "conservative_gain_nats": conservative_gain,
                "conservative_gain_per_gib": efficiency,
            }
        )

    eligible = [record for record in records if record["eligible"]]
    best: tuple[float, int, tuple[str, ...], tuple[dict[str, Any], ...]] | None = None
    for count in range(1, min(args.max_selected_buckets, len(eligible)) + 1):
        for combination in itertools.combinations(eligible, count):
            premium = sum(int(record["premium_bytes"]) for record in combination)
            if premium > args.max_premium_bytes:
                continue
            names = tuple(sorted(str(record["name"]) for record in combination))
            gain = sum(float(record["conservative_gain_nats"]) for record in combination)
            score = (gain, -premium, tuple(reversed(names)), combination)
            if best is None or score[:3] > best[:3]:
                best = score
    selected_records = best[3] if best is not None else ()
    selected = [str(record["name"]) for record in selected_records]
    selected_modules: set[str] = set()
    premium_bytes = 0
    for record in selected_records:
        modules = cast(list[str], record["selected_modules"])
        if selected_modules.intersection(modules):
            raise ValueError("Eligible precision buckets overlap unexpectedly")
        selected_modules.update(modules)
        premium_bytes += int(record["premium_bytes"])

    report = {
        "format": "mxwave-precision-budget-selection-v1",
        "plan": str(plan_path),
        "plan_file_sha256": _sha256_file(plan_path),
        "plan_sha256": plan.get("plan_sha256"),
        "reference": str(Path(args.reference)),
        "baseline": str(Path(args.baseline)),
        "contexts": str(contexts_path),
        "contexts_sha256": contexts_sha256,
        "screen_contexts": screen_rows,
        "split_size": args.split_size,
        "selection_rule": (
            "negative candidate-minus-baseline mean forward KL on both fixed splits; "
            "choose up to the bucket cap by maximum summed smaller-split gain under "
            "the byte budget"
        ),
        "max_selected_buckets": args.max_selected_buckets,
        "max_premium_bytes": args.max_premium_bytes,
        "candidates": records,
        "selected_buckets": selected,
        "selected_modules": sorted(selected_modules),
        "selected_premium_bytes": premium_bytes,
        "selection_passed": bool(selected),
    }
    output = Path(args.output)
    _write_json_atomic(output, report)
    print(
        f"selection: eligible={len(eligible)}/{len(records)} selected={selected} "
        f"premium_bytes={premium_bytes} report={output}",
        flush=True,
    )
    return output


def validate_final_candidate(args: argparse.Namespace) -> Path:
    """Validate a combined candidate only on the untouched context suffix."""
    if args.selection_contexts <= 0:
        raise ValueError("--selection-contexts must be positive")
    if args.bootstrap_iterations <= 0:
        raise ValueError("--bootstrap-iterations must be positive")
    contexts_path = Path(args.contexts)
    contexts_sha256 = _sha256_file(contexts_path)
    reference, _reference_metadata = _load_logprobs(
        Path(args.reference), expected_contexts_sha256=contexts_sha256
    )
    baseline, _baseline_metadata = _load_logprobs(
        Path(args.baseline), expected_contexts_sha256=contexts_sha256
    )
    candidate, _candidate_metadata = _load_logprobs(
        Path(args.candidate), expected_contexts_sha256=contexts_sha256
    )
    if reference.shape != baseline.shape or reference.shape != candidate.shape:
        raise ValueError("Final reference, baseline, and candidate shapes differ")
    holdout_rows = reference.shape[0] - args.selection_contexts
    if holdout_rows < 2 or holdout_rows % 2:
        raise ValueError("Final holdout must contain an even number of contexts")
    reference_holdout = reference[args.selection_contexts :]
    baseline_holdout = baseline[args.selection_contexts :]
    candidate_holdout = candidate[args.selection_contexts :]
    baseline_kl = _forward_kl_rows(reference_holdout, baseline_holdout)
    candidate_kl = _forward_kl_rows(reference_holdout, candidate_holdout)
    delta = candidate_kl - baseline_kl
    half = holdout_rows // 2
    first_mean = _mean(delta[:half])
    second_mean = _mean(delta[half:])
    pooled_mean = _mean(delta)
    reference_top1 = reference_holdout.argmax(dim=1)
    baseline_agreement = int((baseline_holdout.argmax(dim=1) == reference_top1).sum().item())
    candidate_agreement = int((candidate_holdout.argmax(dim=1) == reference_top1).sum().item())
    passed = (
        first_mean < 0.0 and second_mean < 0.0 and candidate_agreement >= baseline_agreement - 1
    )
    selection_path = Path(args.selection)
    selection = _read_json_object(selection_path)
    report = {
        "format": "mxwave-precision-budget-final-v1",
        "selection": str(selection_path),
        "selection_sha256": _sha256_file(selection_path),
        "selected_buckets": selection.get("selected_buckets"),
        "selected_premium_bytes": selection.get("selected_premium_bytes"),
        "selection_contexts_excluded": args.selection_contexts,
        "holdout_contexts": holdout_rows,
        "holdout_halves": [half, half],
        "candidate_minus_baseline_forward_kl_nats": {
            "first_half_mean": first_mean,
            "second_half_mean": second_mean,
            "pooled_mean": pooled_mean,
            "bootstrap_95_ci": _bootstrap_mean_ci(
                [float(value) for value in delta.tolist()],
                seed=args.bootstrap_seed,
                iterations=args.bootstrap_iterations,
            ),
            "candidate_lower_contexts": int((delta < 0).sum().item()),
            "ties": int((delta == 0).sum().item()),
            "baseline_lower_contexts": int((delta > 0).sum().item()),
        },
        "forward_kl_nats": {
            "baseline_holdout_mean": _mean(baseline_kl),
            "candidate_holdout_mean": _mean(candidate_kl),
        },
        "bf16_top1_agreement": {
            "baseline": baseline_agreement,
            "candidate": candidate_agreement,
            "total": holdout_rows,
        },
        "promotion_rule": (
            "lower mean forward KL than baseline on both untouched halves and no more "
            "than one lost BF16 top-1 agreement"
        ),
        "promotion_passed": passed,
    }
    output = Path(args.output)
    _write_json_atomic(output, report)
    print(
        f"final: passed={passed} delta={pooled_mean:+.9g} "
        f"top1={candidate_agreement}/{baseline_agreement} report={output}",
        flush=True,
    )
    return output


def main() -> int:
    """Run selection or final validation and return a process status."""
    try:
        args = build_parser().parse_args()
        handler = cast(Any, args.handler)
        handler(args)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"precision-budget evaluation error: {exc}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
