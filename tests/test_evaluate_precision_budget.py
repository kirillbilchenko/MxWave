"""Tests for held-out measured-precision selection and promotion gates."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import torch
from safetensors.torch import save_file


def _load_script() -> ModuleType:
    path = Path("scripts/evaluate_precision_budget.py")
    spec = importlib.util.spec_from_file_location("evaluate_precision_budget", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _context_files(tmp_path: Path) -> tuple[Path, str, str, str]:
    contexts = [
        {"ordinal": index, "token_ids_sha256": hashlib.sha256(str(index).encode()).hexdigest()}
        for index in range(8)
    ]
    path = tmp_path / "contexts.json"
    path.write_text(json.dumps({"format": "mxstream-next-token-contexts-v1", "contexts": contexts}))
    file_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    first_four = hashlib.sha256(
        "".join(item["token_ids_sha256"] for item in contexts[:4]).encode()
    ).hexdigest()
    all_eight = hashlib.sha256(
        "".join(item["token_ids_sha256"] for item in contexts).encode()
    ).hexdigest()
    return path, file_digest, first_four, all_eight


def _save_logprobs(
    path: Path,
    values: torch.Tensor,
    *,
    contexts_sha256: str,
    selected_digest: str,
) -> None:
    save_file(
        {"logprobs": values},
        path,
        metadata={
            "contexts_manifest_sha256": contexts_sha256,
            "selected_context_sha256": selected_digest,
            "num_contexts": str(values.shape[0]),
            "vocab_size": str(values.shape[1]),
        },
    )


def test_selection_requires_both_splits_and_final_uses_only_holdout(tmp_path: Path) -> None:
    module = _load_script()
    contexts, contexts_sha, first_four, all_eight = _context_files(tmp_path)
    reference = torch.log_softmax(torch.tensor([[4.0, 2.0, 1.0, 0.0]] * 8), dim=1)
    baseline = torch.log_softmax(torch.tensor([[3.0, 2.5, 1.0, 0.0]] * 8), dim=1)
    good = torch.log_softmax(torch.tensor([[3.6, 2.2, 1.0, 0.0]] * 4), dim=1)
    bad_rows = [[3.6, 2.2, 1.0, 0.0]] * 2 + [[2.5, 3.0, 1.0, 0.0]] * 2
    bad = torch.log_softmax(torch.tensor(bad_rows), dim=1)
    reference_path = tmp_path / "reference.safetensors"
    baseline_path = tmp_path / "baseline.safetensors"
    candidates = tmp_path / "candidates"
    candidates.mkdir()
    _save_logprobs(
        reference_path,
        reference,
        contexts_sha256=contexts_sha,
        selected_digest=all_eight,
    )
    _save_logprobs(
        baseline_path,
        baseline,
        contexts_sha256=contexts_sha,
        selected_digest=all_eight,
    )
    _save_logprobs(
        candidates / "good-logprobs.safetensors",
        good,
        contexts_sha256=contexts_sha,
        selected_digest=first_four,
    )
    _save_logprobs(
        candidates / "bad-logprobs.safetensors",
        bad,
        contexts_sha256=contexts_sha,
        selected_digest=first_four,
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "format": "mxwave-precision-budget-plan-v1",
                "plan_sha256": "a" * 64,
                "buckets": [
                    {
                        "name": "good",
                        "premium_bytes": 100,
                        "selected_modules": ["model.good"],
                    },
                    {
                        "name": "bad",
                        "premium_bytes": 100,
                        "selected_modules": ["model.bad"],
                    },
                ],
            }
        )
    )
    selection_path = tmp_path / "selection.json"
    module.select_candidates(
        argparse.Namespace(
            reference=str(reference_path),
            baseline=str(baseline_path),
            contexts=str(contexts),
            plan=str(plan_path),
            candidate_dir=str(candidates),
            output=str(selection_path),
            split_size=2,
            max_selected_buckets=2,
            max_premium_bytes=1000,
            bootstrap_iterations=100,
            bootstrap_seed=7,
        )
    )
    selection = json.loads(selection_path.read_text())
    assert selection["selected_buckets"] == ["good"]
    assert {item["name"]: item["eligible"] for item in selection["candidates"]} == {
        "good": True,
        "bad": False,
    }

    combined = torch.log_softmax(torch.tensor([[3.6, 2.2, 1.0, 0.0]] * 8), dim=1)
    combined_path = tmp_path / "combined.safetensors"
    _save_logprobs(
        combined_path,
        combined,
        contexts_sha256=contexts_sha,
        selected_digest=all_eight,
    )
    final_path = tmp_path / "final.json"
    module.validate_final_candidate(
        argparse.Namespace(
            reference=str(reference_path),
            baseline=str(baseline_path),
            candidate=str(combined_path),
            contexts=str(contexts),
            selection=str(selection_path),
            output=str(final_path),
            selection_contexts=4,
            bootstrap_iterations=100,
            bootstrap_seed=7,
        )
    )
    final = json.loads(final_path.read_text())
    assert final["promotion_passed"] is True
    assert final["holdout_contexts"] == 4
    assert final["candidate_minus_h64_forward_kl_nats"]["pooled_mean"] < 0
