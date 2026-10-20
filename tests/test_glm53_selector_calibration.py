"""Check the selector pilot's objective, state alignment and data separation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from integrations.glm53_mxfp4.selector_calibration import (
    build_dataset,
    covered_loss,
    greedy_tokens,
    prefix_lengths,
    teacher_scores,
)
from integrations.glm53_mxfp4.selector_calibration_protocol import cases


def test_protocol_splits_have_no_shared_prompts() -> None:
    rows = cases()
    assert [sum(row["split"] == name for row in rows)
            for name in ("train", "validation", "test")] == [32, 8, 8]
    assert len({row["text_sha256"] for row in rows}) == 48


def test_uncovered_gold_is_excluded_without_changing_candidates() -> None:
    scores = torch.tensor([[[0.0, 2.0], [1.0, 0.0]]], requires_grad=True)
    ids = torch.tensor([[[4, 7], [8, 9]]])
    loss = covered_loss(scores, ids, torch.tensor([[7, 12]]))
    loss.backward()
    assert scores.grad is not None
    assert torch.equal(scores.grad[0, 1], torch.zeros(2))
    assert scores.grad[0, 0, 1] < 0
    with pytest.raises(ValueError, match="no covered"):
        covered_loss(scores, ids, torch.tensor([[12, 12]]))


def test_greedy_path_conditions_on_own_mistake() -> None:
    projection = torch.tensor([[1.0]], requires_grad=True)
    hidden = torch.ones(1, 2, 1)
    predecessor = torch.tensor([[1.0], [-1.0], [0.0], [1.0]])
    successor = torch.tensor([[0.0], [1.0], [0.0], [1.0]])
    candidates = torch.tensor([[[1, 2], [0, 3]]])
    unary = torch.zeros(1, 2, 2)
    anchors = torch.tensor([0])
    gold = torch.tensor([[2, 3]])
    predicted = greedy_tokens(
        hidden, projection, predecessor, successor, candidates, unary, anchors,
    )
    assert predicted.tolist() == [[1, 0]]
    assert prefix_lengths(predicted, gold).tolist() == [0]
    score = teacher_scores(
        hidden, projection, predecessor, successor, candidates, unary, gold, anchors,
    )
    covered_loss(score, candidates, gold).backward()
    assert projection.grad is not None and torch.isfinite(projection.grad).all()


def test_prefix_length_stops_after_first_mismatch() -> None:
    predicted = torch.tensor([[1, 0, 3, 4], [1, 2, 3, 4]])
    gold = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]])
    assert prefix_lengths(predicted, gold).tolist() == [1, 4]


@pytest.mark.parametrize("wrong_anchor", [False, True])
def test_dataset_aligns_absolute_positions_and_excludes_tail(
    tmp_path: Path, wrong_anchor: bool,
) -> None:
    folder = tmp_path / "ct-r0"
    folder.mkdir()
    case = {"id": "fixture", "split": "train", "domain": "code", "prompt_ids": [1, 2, 3]}
    (tmp_path / "plan.json").write_text(json.dumps({"calibration_cases": [case]}))
    (tmp_path / "ct-measurement.json").write_text(json.dumps({
        "status": "complete", "cases": [{
            "case": "fixture", "policy": "capture", "result": {"token_ids": [5] + [0] * 7},
        }],
    }))
    for index, start in enumerate((4, 11)):
        data = {
            "ids": torch.arange(16).repeat(7, 1), "scores": torch.zeros(7, 16, 16),
            "proposal": torch.zeros(7, dtype=torch.int64),
            "sample_positions": torch.arange(start, start + 7),
            "anchors": torch.tensor([99 if wrong_anchor else 5]),
            "hidden": torch.ones(7, 4), "values": torch.zeros(7, 16),
        }
        path = folder / f"calibration-fixture-{index:04d}.safetensors"
        save_file(data, str(path))
        path.with_suffix(".json").write_text(json.dumps({
            "payload_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "tensor_dtypes": {name: str(value.dtype) for name, value in data.items()},
        }))
    if wrong_anchor:
        with pytest.raises(ValueError, match="Anchor/history mismatch"):
            build_dataset(tmp_path)
    else:
        rows, coverage = build_dataset(tmp_path)
        assert len(rows) == 1 and rows[0]["gold"].tolist() == [0] * 7
        assert coverage == {"complete_blocks": 1, "incomplete_tail_frames": 1}
