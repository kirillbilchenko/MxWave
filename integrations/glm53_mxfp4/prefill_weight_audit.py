"""Observe eager per-forward weight identities only in untimed diagnostic requests."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from typing import Any


class WeightAudit:
    """Call the exact saved model/apply methods and restore both when observation stops."""

    def __init__(self, model: Any, runtime: Any, *, capture: Callable[[], bool],
                 compiling: Callable[[], bool]) -> None:
        self.model, self.runtime = model, runtime
        self.original_forward, self.original_apply = model.forward, runtime.apply
        self.capture, self.compiling = capture, compiling
        self.events: list[dict[str, Any]] = []
        self.current: list[dict[str, Any]] | None = None
        self.active = False

    def start(self) -> None:
        """Install metadata observation after timing and graph capture have already finished."""
        if self.active:
            raise RuntimeError("Weight observation is already active")
        if self.runtime.apply != self.original_apply or self.model.forward != self.original_forward:
            raise RuntimeError("Model/apply methods changed before weight observation")

        def forward(*args: Any, **kwargs: Any) -> Any:
            if self.compiling() or self.capture():
                return self.original_forward(*args, **kwargs)
            if self.current is not None or len(self.events) >= 256:
                raise RuntimeError("Nested model forward or weight-observation bound exceeded")
            self.current = []
            try:
                return self.original_forward(*args, **kwargs)
            finally:
                rows = self.current
                counts = Counter(r["prefix"] for r in rows)
                self.events.append({"forward_id": len(self.events), "calls": rows,
                    "duplicate_weight_prefixes": {k: n for k, n in counts.items() if n > 1}})
                self.current = None

        def apply(layer: Any, x: Any) -> Any:
            if self.current is not None:
                metadata = layer.mxwave_kda_qualification
                width = int(layer.mxwave_kda_metadata[1])
                tokens = x.numel() // width
                self.current.append({"prefix": metadata["prefix"], "rows": tokens,
                    "mode": "reconstruction" if tokens > self.runtime.FUSED_MAX_TOKENS else "fused"})
            return self.original_apply(layer, x)

        self.wrapped_forward, self.wrapped_apply = forward, apply
        self.model.forward, self.runtime.apply = forward, apply
        self.active = True

    def stop(self) -> dict[str, Any]:
        """Remove observation exactly; never claim Python records count captured graph replays."""
        if (not self.active or self.current is not None
                or self.model.forward is not self.wrapped_forward
                or self.runtime.apply is not self.wrapped_apply):
            raise RuntimeError("Weight-observation ownership or closed-forward contract changed")
        self.model.forward, self.runtime.apply = self.original_forward, self.original_apply
        self.active = False
        return {"status": "complete", "active": False, "events": self.events,
            "saved_methods_restored": True, "during_unprofiled_timing": False,
            "scope": "Eager Python model/apply calls; captured graph replays are excluded."}
