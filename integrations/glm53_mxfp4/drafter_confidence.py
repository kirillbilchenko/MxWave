"""Experimental acceptance head; it observes the drafter without changing its proposals.

Design reference: DeepInfer 774ce244b87cc0eb8d4b94ab9f5de2ab22049f01.
Independently authored; provenance and limits are recorded in CONTRIBUTING.md.
This module does not install a serving hook or replace target verification.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class ConfidenceHead(nn.Module):
    """Predict conditional greedy acceptance from detached draft/selector features."""

    def __init__(self, hidden_size: int, predecessor_rank: int) -> None:
        super().__init__()
        if hidden_size < 1 or predecessor_rank < 1:
            raise ValueError("Confidence feature dimensions must be positive")
        self.hidden_size = hidden_size
        self.predecessor_rank = predecessor_rank
        self.weight = nn.Parameter(torch.zeros(1, hidden_size + predecessor_rank))

    def forward(self, hidden: torch.Tensor, predecessor: torch.Tensor) -> torch.Tensor:
        """Return FP32 logits; the auxiliary loss cannot update drafter features."""
        if (hidden.ndim != 3 or predecessor.ndim != 3
                or hidden.shape[:-1] != predecessor.shape[:-1]
                or hidden.shape[-1] != self.hidden_size
                or predecessor.shape[-1] != self.predecessor_rank):
            raise ValueError("Expected matching [blocks, draft positions, features]")
        features = torch.cat((hidden.detach().float(), predecessor.detach().float()), dim=-1)
        return F.linear(features, self.weight).squeeze(-1)


def conditional_labels(
    proposals: torch.Tensor, target_tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Label greedy acceptance only while every earlier proposal matches the target.

    Later gold-sequence tokens are not counterfactual target labels after a rejected
    proposal. Include the first rejection and censor all positions after it.
    """
    if (proposals.ndim != 2 or proposals.shape != target_tokens.shape
            or min(proposals.shape) < 1
            or proposals.dtype != torch.long or target_tokens.dtype != torch.long
            or proposals.device != target_tokens.device):
        raise ValueError("Require nonempty matching int64 token blocks on one device")
    agreement = proposals == target_tokens
    observed = torch.cat(
        (torch.ones_like(agreement[:, :1]), agreement[:, :-1].long().cumprod(-1).bool()),
        dim=-1,
    )
    return agreement.float(), observed


def confidence_loss(
    logits: torch.Tensor, proposals: torch.Tensor, target_tokens: torch.Tensor,
) -> torch.Tensor:
    """Compute conditional BCE with equal weight per independently proposed block."""
    labels, observed = conditional_labels(proposals, target_tokens)
    if logits.shape != labels.shape:
        raise ValueError("Confidence logits must match the proposal block")
    element_loss = F.binary_cross_entropy_with_logits(logits.float(), labels, reduction="none")
    if not torch.isfinite(logits).all():
        raise ValueError("Non-finite confidence logits")
    element_loss = torch.where(observed, element_loss, 0.0)
    return (element_loss.sum(-1) / observed.sum(-1)).mean()


def acceptance_metrics(
    logits: torch.Tensor, proposals: torch.Tensor, target_tokens: torch.Tensor,
) -> dict[str, float | int]:
    """Report conditional calibration and uncensored prefix prediction errors."""
    labels, observed = conditional_labels(proposals, target_tokens)
    if logits.shape != labels.shape:
        raise ValueError("Confidence logits must match the proposal block")
    probabilities = logits.detach().float().sigmoid()
    if not torch.isfinite(probabilities).all():
        raise ValueError("Non-finite confidence prediction")
    prefixes = probabilities.cumprod(-1)
    actual_prefixes = labels.long().cumprod(-1).float()
    error = probabilities[observed] - labels[observed]
    return {
        "blocks": logits.shape[0],
        "observed_positions": int(observed.sum()),
        "conditional_brier": float(error.square().mean()),
        "prefix_brier": float((prefixes - actual_prefixes).square().mean()),
        "predicted_accepted_length": float(prefixes.sum(-1).mean()),
        "actual_accepted_length": float(actual_prefixes.sum(-1).mean()),
    }


class ConfidenceObserver:
    """Opt-in training observer for Pilot.objective, with a separate optimizer."""

    def __init__(self, head: ConfidenceHead, *, learning_rate: float) -> None:
        self.head = head
        self.optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=0)
        self.rows: list[dict[str, Any]] = []

    def __call__(
        self, sample: dict[str, Any], hidden: torch.Tensor, predecessor: torch.Tensor,
        proposals: torch.Tensor, target_tokens: torch.Tensor,
    ) -> None:
        """Train on training prompts only; validation records scores without updates."""
        if sample["split"] not in ("train", "validation"):
            raise ValueError("Confidence training forbids sealed test data")
        training = sample["split"] == "train"
        with torch.set_grad_enabled(training):
            logits = self.head(hidden, predecessor)
            loss = confidence_loss(logits, proposals, target_tokens)
            record = {
                "id": sample["id"], "split": sample["split"], "domain": sample["domain"],
                "loss_before_update": float(loss.detach()),
                **acceptance_metrics(logits, proposals, target_tokens),
            }
            if training:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.head.parameters(), 1.0, error_if_nonfinite=True)
                self.optimizer.step()
        self.rows.append(record)
