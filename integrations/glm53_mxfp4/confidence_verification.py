"""CPU reference for confidence-scheduled prefix verification, disabled in serving.

The hardware cost table must be measured on this target/runtime. Choose the
total budget from older confidence, then allocate using current confidence;
current future-position confidence must not select its own total budget.
DeepInfer's public design is cited in CONTRIBUTING.md; no upstream code is copied.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


def ranked_prefixes(
    probabilities: Sequence[Sequence[float]],
) -> list[tuple[float, int, int]]:
    """Order marginal prefix acceptance by probability with stable request/position ties."""
    if not probabilities:
        raise ValueError("At least one request is required")
    values = []
    for request, row in enumerate(probabilities):
        prefix = 1.0
        for position, probability in enumerate(row):
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("Conditional probabilities must be finite and in [0, 1]")
            prefix *= probability
            values.append((prefix, request, position))
    return sorted(values, key=lambda item: (-item[0], item[1], item[2]))


def allocate_prefixes(
    probabilities: Sequence[Sequence[float]], *, total_query_rows: int,
) -> tuple[int, ...]:
    """Allocate a fixed budget; each request gets its base row and a contiguous draft prefix.

    A row with no available drafts must be empty, including newly admitted requests.
    Zero-confidence drafts remain available; zero confidence is not an availability mask.
    """
    ranked = ranked_prefixes(probabilities)
    requests = len(probabilities)
    if (isinstance(total_query_rows, bool) or not isinstance(total_query_rows, int)
            or not requests <= total_query_rows <= requests + len(ranked)):
        raise ValueError("Budget exceeds available query rows")
    lengths = [1] * requests
    for _, request, position in ranked[:total_query_rows - requests]:
        if position != lengths[request] - 1:
            raise AssertionError("Selected drafts must be contiguous prefixes")
        lengths[request] += 1
    return tuple(lengths)


@dataclass(frozen=True)
class HistoricalConfidence:
    """Confidence from an older decision, remapped to the current active requests."""

    decision: int
    probabilities: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class VerificationBudget:
    """Offline scheduling estimate, not a measured throughput result."""

    total_query_rows: int
    expected_emitted_tokens: float
    measured_step_ms: float

    @property
    def estimated_tokens_per_ms(self) -> float:
        """Return the modelled throughput at this measured hardware operating point."""
        return self.expected_emitted_tokens / self.measured_step_ms


def choose_budget(
    history: HistoricalConfidence, *, current_decision: int,
    measured_step_ms: Mapping[int, float],
) -> VerificationBudget:
    """Choose among measured costs using confidence at least two decisions old.

    Times must include draft, target, sampler and scheduling work for the actual
    graph/padding policy. There is no interpolation or extrapolation to unmeasured
    widths. This CPU oracle does not authorize asynchronous runtime integration.
    """
    if history.decision < 0 or current_decision - history.decision < 2:
        raise ValueError("Budget selection requires confidence at least two decisions old")
    ranked = ranked_prefixes(history.probabilities)
    requests = len(history.probabilities)
    capacity = requests + len(ranked)
    if not measured_step_ms:
        raise ValueError("A measured cost table is required")
    for rows, cost in measured_step_ms.items():
        if (isinstance(rows, bool) or not isinstance(rows, int) or rows < 1
                or not math.isfinite(cost) or cost <= 0):
            raise ValueError("Cost table requires positive integer widths and finite positive times")
    best = None
    expected = float(requests)
    for rows in range(requests, capacity + 1):
        if rows > requests:
            expected += ranked[rows - requests - 1][0]
        if rows not in measured_step_ms:
            continue
        candidate = VerificationBudget(rows, expected, measured_step_ms[rows])
        if best is None or candidate.estimated_tokens_per_ms > best.estimated_tokens_per_ms:
            best = candidate
    if best is None:
        raise ValueError("No measured width fits the available requests and drafts")
    return best
