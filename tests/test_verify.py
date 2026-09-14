"""Unit tests for the verification-first contract."""

from __future__ import annotations

import torch

from mxstream.verify import sqnr, verify_config_coverage


def test_sqnr_returns_positive_db_for_close_tensors():
    x = torch.randn(64)
    # small noise
    x_hat = x + 0.01 * torch.randn_like(x)
    assert sqnr(x, x_hat) > 20.0


def test_sqnr_is_very_high_for_perfect_reconstruction():
    x = torch.randn(64)
    # noise floor clamped → very large but finite value
    assert sqnr(x, x) > 100.0


def test_config_coverage_finds_uncovered_modules():
    targets = ["re:.*layers\\..*\\..*_proj$"]
    ignore = ["lm_head"]
    real = [
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.mlp.gate_proj",
        "model.lm_head",
        "model.embed_tokens",
    ]
    uncovered = verify_config_coverage(targets, ignore, real)
    assert "model.embed_tokens" in uncovered
    assert "model.lm_head" not in uncovered


def test_config_coverage_empty_when_all_covered():
    targets = ["re:.*"]
    real = ["model.layers.0.self_attn.q_proj"]
    assert verify_config_coverage(targets, [], real) == []
