"""Ensure held-out prompt groups are never reused as calibration examples."""

from __future__ import annotations

import hashlib

import pytest

from integrations.glm53_mxfp4.prepare_drafter_dataset import allocate, group_key
from integrations.glm53_mxfp4.verify_drafter_dataset import token_digest


def test_literal_and_numeric_instruction_variants_stay_in_one_group() -> None:
    assert group_key('Sort 12 records named "alpha"') == group_key('Sort 99 records named "beta"')
    assert group_key("Explain merge sort") != group_key("Explain database transactions")


def test_split_allocator_keeps_groups_disjoint_and_is_order_independent() -> None:
    rows = []
    for i, text in enumerate(["First", "Second", "Third", "Fourth"]):
        for variant in range(2):
            rows.append({
                "id": f"{i}-{variant}", "domain": "code", "group_sha256": group_key(text),
                "prompt_sha256": hashlib.sha256(f"{text}-{variant}".encode()).hexdigest(),
            })
    quotas = {name: {"code": 1} for name in ("train", "validation", "test")}
    result = allocate(rows, quotas)
    assert result == allocate(rows[::-1], quotas)
    selected = [row for split in result.values() for row in split]
    assert len({row["group_sha256"] for row in selected}) == 3
    with pytest.raises(ValueError, match="Insufficient"):
        allocate(rows[:2], quotas)


def test_token_identity_rejects_batch_encoding_keys() -> None:
    assert token_digest([1, 2]) != token_digest([2, 1])
    with pytest.raises(ValueError, match="integer"):
        token_digest(["input_ids", "attention_mask"])  # type: ignore[list-item]
