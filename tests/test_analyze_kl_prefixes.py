"""Check prefix pairing and exact-distribution analysis without model inference."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file


def _script(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parent.parent / "scripts"))
    return importlib.import_module("analyze_kl_prefixes")


def test_prefix_pairing_uses_window_identity_not_row_order(monkeypatch: pytest.MonkeyPatch) -> None:
    script = _script(monkeypatch)
    contexts = [
        {"window_index": 20, "prefix_tokens": 512},
        {"window_index": 10, "prefix_tokens": 64},
        {"window_index": 20, "prefix_tokens": 64},
        {"window_index": 10, "prefix_tokens": 512},
    ]
    windows, prefixes, rows = script._prefix_rows(contexts)
    assert windows == [10, 20]
    assert prefixes == [64, 512]
    assert rows.tolist() == [[1, 2], [3, 0]]
    with pytest.raises(ValueError, match="exactly one"):
        script._prefix_rows(contexts + contexts[:1])
    with pytest.raises(ValueError, match="exactly one"):
        script._prefix_rows(contexts[:-1])


def test_distribution_metrics_match_analytic_kl_and_reject_wrong_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _script(monkeypatch)
    teacher, candidate = tmp_path / "teacher.safetensors", tmp_path / "candidate.safetensors"
    left = torch.tensor([[0.6, 0.3, 0.1], [0.1, 0.2, 0.7]], dtype=torch.float64)
    right = torch.tensor([[0.5, 0.25, 0.25], [0.4, 0.5, 0.1]], dtype=torch.float64)
    for path, values in ((teacher, left), (candidate, right)):
        save_file({"logprobs": values.log()}, str(path), metadata={"protocol_sha256": "frozen"})
    kl, agreement = script._read_metrics(teacher, candidate, protocol_sha="frozen", positions=2)
    np.testing.assert_allclose(kl, (left * (left.log() - right.log())).sum(dim=1).numpy())
    assert agreement.tolist() == [True, False]
    with pytest.raises(ValueError, match="identity"):
        script._read_metrics(teacher, candidate, protocol_sha="other", positions=2)
    save_file({"logprobs": right.log() + 1}, str(candidate), metadata={"protocol_sha256": "frozen"})
    with pytest.raises(ValueError, match="normalized"):
        script._read_metrics(teacher, candidate, protocol_sha="frozen", positions=2)


def test_paired_summary_resamples_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    script = _script(monkeypatch)
    values = np.array([-0.01, -0.03])
    samples = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
    summary = script._paired_summary(values, samples)
    assert summary["mean"] == pytest.approx(-0.02)
    np.testing.assert_allclose(summary["ci95"], np.percentile([-0.01, -0.03, -0.02, -0.02], [2.5, 97.5]))
    assert summary["window_count"] == 2
