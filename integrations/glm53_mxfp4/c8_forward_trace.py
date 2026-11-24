"""Observe CPU scheduler and prepared-batch metadata without copying GPU tensors."""

from __future__ import annotations

import functools
import hashlib
import json
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any


class ForwardTrace:
    """Wrap existing runner calls and retain their actual CPU batch order and descriptor."""

    def __init__(self, runner: Any, annotation: Callable[[str], Any] = nullcontext) -> None:
        self.runner = runner
        self.annotation = annotation
        self.enabled = False
        self.label = ""
        self.events: list[dict[str, Any]] = []
        self.requests: dict[str, dict[str, Any]] = {}
        self.current: dict[str, Any] | None = None
        original_execute, original_prepare = runner.execute_model, runner.prepare_inputs

        @functools.wraps(original_execute)
        def execute(scheduled: Any, *args: Any, **kwargs: Any) -> Any:
            if not self.enabled or not scheduled.total_num_scheduled_tokens:
                return original_execute(scheduled, *args, **kwargs)
            if self.current is not None or len(self.events) >= 512:
                raise RuntimeError("Overlapping or excessive C8 trace forwards")
            for request in scheduled.scheduled_new_reqs:
                ids = request.prompt_token_ids
                if ids is None:
                    raise ValueError("This diagnostic covers text-token prompts only")
                self.requests[request.req_id] = {
                    "prompt_sha256": hashlib.sha256(
                        json.dumps(ids, separators=(",", ":")).encode()
                    ).hexdigest(),
                    "prompt_tokens": len(ids),
                }
            counts = scheduled.num_scheduled_tokens
            for req_id in counts:
                if req_id not in self.requests:
                    raise ValueError("Trace started with an existing unobserved request")
            event = {
                "sequence": len(self.events),
                "total_scheduled_tokens": int(scheduled.total_num_scheduled_tokens),
                "scheduled_order": [
                    {"request_id": req_id, "scheduled_tokens": int(count), **self.requests[req_id]}
                    for req_id, count in counts.items()
                ],
            }
            self.events.append(event)
            self.current = event
            try:
                with self.annotation(
                    f"mxwave_c8_{self.label}_step_{event['sequence']}"
                ):
                    result = original_execute(scheduled, *args, **kwargs)
                event["status"] = "complete"
                return result
            except BaseException as exc:
                event.update(status="failed", error=repr(exc))
                raise
            finally:
                self.current = None

        @functools.wraps(original_prepare)
        def prepare(*args: Any, **kwargs: Any) -> Any:
            batch = original_prepare(*args, **kwargs)
            if self.enabled and self.current is not None:
                descriptor = args[2] if len(args) > 2 else kwargs["batch_desc"]
                counts = {
                    row["request_id"]: row["scheduled_tokens"]
                    for row in self.current["scheduled_order"]
                }
                req_ids = list(batch.req_ids)
                self.current["prepared"] = {
                    "num_reqs": int(batch.num_reqs),
                    "num_tokens": int(batch.num_tokens),
                    "num_tokens_after_padding": int(batch.num_tokens_after_padding),
                    "has_prefill": bool(batch.has_prefill),
                    "request_state_indices": batch.idx_mapping_np.tolist(),
                    "prefill_lengths": batch.prefill_len_np.tolist(),
                    "row_order": [
                        {
                            "request_id": req_id,
                            "scheduled_tokens": counts[req_id],
                            **self.requests[req_id],
                        }
                        for req_id in req_ids
                    ],
                    "descriptor": {
                        "mode": str(descriptor.cg_mode),
                        "num_tokens": int(descriptor.num_tokens),
                        "num_reqs": (
                            None if descriptor.num_reqs is None else int(descriptor.num_reqs)
                        ),
                        "max_query_len": (
                            None
                            if descriptor.max_query_len is None
                            else int(descriptor.max_query_len)
                        ),
                    },
                }
            return batch

        runner.execute_model = execute
        runner.prepare_inputs = prepare

    def configure(self, enabled: bool, label: str) -> dict[str, Any]:
        """Arm only between completed groups; disarming preserves all recorded metadata."""
        if self.current is not None:
            raise RuntimeError("Cannot change tracing during a forward")
        if enabled:
            if not label or not label.replace("-", "").isalnum() or self.enabled:
                raise ValueError("Require a fresh simple label and a disarmed trace")
            self.label, self.events, self.requests = label, [], {}
        self.enabled = bool(enabled)
        return self.report()

    def report(self) -> dict[str, Any]:
        """Return plain host metadata and explicitly delimit observer effects."""
        return {
            "enabled": self.enabled,
            "label": self.label,
            "events": list(self.events),
            "host_metadata_only": True,
            "GPU_tensors_copied": False,
            "model_or_scheduler_arithmetic_changed": False,
            "observer_changes_CPU_work": True,
        }


def forward_signature(event: dict[str, Any]) -> dict[str, Any]:
    """Normalize only transient request IDs/state slots, retaining true prompt row order."""
    prepared = event["prepared"]
    return {
        "total_scheduled_tokens": event["total_scheduled_tokens"],
        "num_reqs": prepared["num_reqs"],
        "num_tokens": prepared["num_tokens"],
        "num_tokens_after_padding": prepared["num_tokens_after_padding"],
        "has_prefill": prepared["has_prefill"],
        "prefill_lengths": prepared["prefill_lengths"],
        "descriptor": prepared["descriptor"],
        "row_order": [
            {key: row[key] for key in ("prompt_sha256", "prompt_tokens", "scheduled_tokens")}
            for row in prepared["row_order"]
        ],
    }
