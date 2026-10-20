from __future__ import annotations

import itertools
from types import SimpleNamespace

import pytest
import torch

from integrations.glm53_mxfp4.confidence_verification import (
    HistoricalConfidence,
    allocate_prefixes,
    choose_budget,
)
from integrations.glm53_mxfp4.drafter_confidence import (
    ConfidenceHead,
    ConfidenceObserver,
    acceptance_metrics,
    conditional_labels,
    confidence_loss,
)
from integrations.glm53_mxfp4.train_drafter_context_pilot import Pilot


def test_labels_censor_after_first_rejection() -> None:
    proposals = torch.tensor([[1, 9, 3, 4], [1, 2, 3, 4], [0, 2, 3, 4]])
    target = torch.tensor([[1, 2, 3, 4]]).expand_as(proposals)
    labels, observed = conditional_labels(proposals, target)
    assert observed.tolist() == [[True, True, False, False], [True] * 4,
                                 [True, False, False, False]]
    assert labels[:, 0].tolist() == [1.0, 1.0, 0.0]
    logits = torch.zeros(3, 4, requires_grad=True)
    confidence_loss(logits, proposals, target).backward()
    assert torch.equal(logits.grad[~observed], torch.zeros(5))
    # Two of three first positions succeed: a shared zero logit must move upward.
    # Dividing each block by its observed prefix length would reverse this sign.
    assert float(logits.grad[:, 0].sum()) < 0
    report = acceptance_metrics(logits, proposals, target)
    assert report["observed_positions"] == 7
    assert report["conditional_brier"] == 0.25


def test_head_trains_without_updating_drafter_or_selector() -> None:
    torch.manual_seed(42)
    hidden = torch.randn(3, 7, 8, requires_grad=True)
    predecessor = torch.randn(3, 7, 2, requires_grad=True)
    head = ConfidenceHead(8, 2)
    tokens = torch.ones(3, 7, dtype=torch.long)
    initial = confidence_loss(head(hidden, predecessor), tokens, tokens).item()
    optimizer = torch.optim.SGD(head.parameters(), lr=0.1)
    for _ in range(5):
        optimizer.zero_grad()
        loss = confidence_loss(head(hidden, predecessor), tokens, tokens)
        loss.backward()
        optimizer.step()
    assert confidence_loss(head(hidden, predecessor), tokens, tokens).item() < initial
    assert hidden.grad is None and predecessor.grad is None
    reloaded = ConfidenceHead(8, 2)
    reloaded.load_state_dict(head.state_dict())
    assert torch.equal(head(hidden, predecessor), reloaded(hidden, predecessor))


def test_observer_freezes_validation_and_refuses_test_data() -> None:
    head = ConfidenceHead(4, 2)
    observer = ConfidenceObserver(head, learning_rate=0.01)
    hidden = torch.ones(1, 3, 4, requires_grad=True)
    previous = torch.ones(1, 3, 2, requires_grad=True)
    tokens = torch.ones(1, 3, dtype=torch.long)
    row = {"id": "fixture", "split": "train", "domain": "code"}
    # The observer also works when the drafter capture itself runs without gradients.
    with torch.no_grad():
        observer(row, hidden, previous, tokens, tokens)
    assert torch.count_nonzero(head.weight) == 6
    assert hidden.grad is None and previous.grad is None
    before = head.weight.detach().clone()
    observer(row | {"split": "validation"}, hidden, previous, tokens, tokens)
    assert torch.equal(before, head.weight)
    with pytest.raises(ValueError, match="sealed"):
        observer(row | {"split": "test"}, hidden, previous, tokens, tokens)
    assert len(observer.rows) == 2


def test_pilot_observer_preserves_proposals_loss_and_context_gradient() -> None:
    class Selector:
        predecessor_codebook = torch.ones(16, 2)

        def score_candidates(self, **kwargs: torch.Tensor) -> torch.Tensor:
            return kwargs["unary_logits"]

        def greedy_path(self, **kwargs: torch.Tensor) -> torch.Tensor:
            return kwargs["candidate_ids"][..., 0]

    torch.manual_seed(42)
    pilot = Pilot.__new__(Pilot)
    pilot.feature_store = None
    pilot.fp32_derivatives = False
    pilot.head = torch.randn(16, 4)
    pilot.model = SimpleNamespace(candidate_selector=Selector())
    hidden = torch.randn(1, 7, 4, requires_grad=True)
    pilot.forward = lambda _: hidden
    sample = {"id": "fixture", "split": "train", "domain": "code", "anchors": [0],
              "tensors": {"input_ids": torch.arange(8)}}
    pilot.confidence_observer = None
    original_loss, original_metrics = pilot.objective(sample)
    original_loss.backward()
    original_gradient = hidden.grad.clone()
    hidden.grad = None
    pilot.confidence_observer = ConfidenceObserver(ConfidenceHead(4, 2), learning_rate=0.01)
    observed_loss, observed_metrics = pilot.objective(sample)
    observed_loss.backward()
    assert torch.equal(original_loss, observed_loss)
    assert original_metrics == observed_metrics
    assert torch.equal(original_gradient, hidden.grad)
    assert len(pilot.confidence_observer.rows) == 1


def test_allocation_matches_exhaustive_prefix_optimizer() -> None:
    probabilities = [[0.8, 0.9, 0.3], [], [0.6, 0.7], [0.0, 1.0]]
    prefix_values = [[1.0] for _ in probabilities]
    for values, row in zip(prefix_values, probabilities, strict=True):
        for probability in row:
            values.append(values[-1] * probability)
    for budget in range(4, 12):
        lengths = allocate_prefixes(probabilities, total_query_rows=budget)
        actual = sum(sum(v[1:length]) for v, length in zip(prefix_values, lengths, strict=True))
        possible = [
            sum(sum(v[1:length]) for v, length in zip(prefix_values, candidate, strict=True))
            for candidate in itertools.product(*(range(1, len(row) + 2) for row in probabilities))
            if sum(candidate) == budget
        ]
        assert sum(lengths) == budget and lengths[1] == 1
        assert actual == pytest.approx(max(possible))


def test_stable_ties_and_no_future_position_effect_on_earlier_choice() -> None:
    assert allocate_prefixes([[1, 1], [1, 1]], total_query_rows=4) == (3, 1)
    for tail in (0.0, 0.1, 1.0):
        lengths = allocate_prefixes([[0.9, tail], [0.8, 0.5]], total_query_rows=3)
        assert lengths == (2, 1)


def test_budget_requires_older_confidence_and_real_costs() -> None:
    history = HistoricalConfidence(2, ((0.9, 0.9, 0.1),))
    result = choose_budget(history, current_decision=4,
                           measured_step_ms={1: 10.0, 2: 10.0, 3: 10.0, 4: 20.0})
    assert result.total_query_rows == 3
    assert result.expected_emitted_tokens == pytest.approx(2.71)
    assert choose_budget(history, current_decision=4, measured_step_ms={1: 10.0}).total_query_rows == 1
    with pytest.raises(ValueError, match="two decisions"):
        choose_budget(history, current_decision=3, measured_step_ms={3: 10.0})
    with pytest.raises(ValueError, match="No measured"):
        choose_budget(history, current_decision=4, measured_step_ms={8: 10.0})


@pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_probability_is_rejected(probability: float) -> None:
    with pytest.raises(ValueError, match="probabilities"):
        allocate_prefixes([[probability]], total_query_rows=1)


def test_missing_drafts_and_invalid_costs_are_rejected() -> None:
    with pytest.raises(ValueError, match="Budget"):
        allocate_prefixes([[]], total_query_rows=2)
    with pytest.raises(ValueError, match="cost|Cost"):
        choose_budget(HistoricalConfidence(0, ((0.5,),)), current_decision=2,
                      measured_step_ms={2: float("nan")})
