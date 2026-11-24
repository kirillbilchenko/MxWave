"""Validate and execute the frozen one-pass context-projection training schedule."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TrainingSchedule:
    """Explicit warmup/cosine schedule with a fixed budget and validation checkpoints."""

    maximum_steps: int
    warmup_steps: int
    peak_rate: float
    final_rate: float
    validation_steps: tuple[int, ...]

    @classmethod
    def from_protocol(cls, protocol: dict[str, Any]) -> TrainingSchedule:
        """Reject ambiguous or unsupported training settings before loading large tensors."""
        optimizer = protocol["optimizer"]
        fixed = {
            "name": "AdamW", "master_dtype": "float32", "forward_weight_dtype": "bfloat16",
            "weight_decay": 0, "gradient_clip_l2": 1, "seed": 42,
        }
        if any(optimizer.get(key) != value for key, value in fixed.items()):
            raise ValueError("Unsupported frozen optimizer policy")
        maximum, warmup = optimizer["maximum_steps"], optimizer["warmup_steps"]
        if (type(maximum) is not int or type(warmup) is not int
                or not 0 < warmup < maximum <= 4000):
            raise ValueError("Expected 1 <= warmup < maximum <= 4000 updates")
        peak, final = optimizer["peak_learning_rate"], optimizer["final_learning_rate"]
        if not (math.isfinite(peak) and math.isfinite(final) and 0 < final <= peak):
            raise ValueError("Learning rates must be finite, positive and decreasing")
        steps = tuple(protocol["validation"]["steps"])
        if (any(type(step) is not int for step in steps)
                or steps != tuple(sorted(set(steps))) or not steps
                or steps[0] != 0 or steps[-1] != maximum):
            raise ValueError("Validation steps must increase from zero to the update limit")
        if protocol["trainable"] != ["fc.weight"]:
            raise ValueError("Only the context projection may train")
        return cls(maximum, warmup, peak, final, steps)

    def learning_rate(self, step: int) -> float:
        """Return the original pilot arithmetic generalized to the frozen update budget."""
        if not 1 <= step <= self.maximum_steps:
            raise ValueError("Training step exceeds the frozen schedule")
        if step <= self.warmup_steps:
            return self.peak_rate * step / self.warmup_steps
        return self.final_rate + 0.5 * (self.peak_rate - self.final_rate) * (
            1 + math.cos(math.pi * (step - self.warmup_steps)
                         / (self.maximum_steps - self.warmup_steps))
        )

    def evaluate_after(self, step: int, eligible_prompts: int) -> bool:
        """Evaluate scheduled checkpoints and the last eligible step without replacing short outputs."""
        last = min(self.maximum_steps, eligible_prompts)
        if not 1 <= step <= last:
            raise ValueError("Evaluation step lies outside the actual one-pass run")
        return step in self.validation_steps or step == last
