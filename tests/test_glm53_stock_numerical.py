"""Verify fixed-input evidence keeps complete distributions and exact tensor bits."""

from __future__ import annotations

import math

import pytest
import torch

from deployment.glm53.numerical_client import compare, distribution
from deployment.glm53.numerical_worker import tensor_record


def test_full_distribution_and_normalized_kl():
    response = {"choices": [{"logprobs": {"top_logprobs": [
        {"token_id:1": math.log(.75), "token_id:0": math.log(.25)},
    ]}}]}
    values = distribution(response, 2)
    assert values == [math.log(.25), math.log(.75)]
    with pytest.raises(ValueError, match="Incomplete"):
        distribution(response, 3)
    offset = compare(values, [x + .125 for x in values])
    assert not offset["logprobs_equal"] and abs(offset["kl"]) < 1e-14
    changed = compare(values, list(reversed(values)))
    assert not changed["kl_pass_1e_5"] and not changed["argmax_equal"]


def test_activation_hash_handles_noncontiguous_bf16_and_signed_zero():
    value = torch.arange(600, dtype=torch.bfloat16).reshape(100, 6)[:, ::2]
    assert tensor_record(value, full=True) == tensor_record(value.contiguous(), full=True)
    changed = value.clone()
    changed[0, 0] = -0.0
    first, second = tensor_record(value, full=True), tensor_record(changed, full=True)
    assert first["full_sha256"] != second["full_sha256"]
    assert first["tail_sha256"] == second["tail_sha256"]
