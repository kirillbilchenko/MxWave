"""Opt-in deeper MoE diagnostics in disposable original-checkpoint workers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from nv_worker import NvWorker


class MoeWorker(NvWorker):
    """Expose internal snapshots and fixed-operand replay without replacing arithmetic."""

    def nv_moe_capture(self, enabled: bool, label: str = "") -> dict[str, Any]:
        """Arm or finish one bounded 256-token prefill capture."""
        from moe_capture import MoeCapture
        from vllm.distributed import get_tensor_model_parallel_rank

        if not hasattr(self, "_nv_moe"):
            self._nv_moe = MoeCapture(self.model_runner)
        result = (self._nv_moe.arm(label) if enabled else
                  self._nv_moe.finish(Path("/traces") / self._nv_moe.label))
        return {"rank": get_tensor_model_parallel_rank(), **result}

    def nv_moe_replay(self, repeats: int = 20) -> dict[str, Any]:
        """Replay captured runtime fixtures on both TP ranks in matching order."""
        from vllm.distributed import get_tensor_model_parallel_rank

        result = self._nv_moe.replay(Path("/traces") / self._nv_moe.label, repeats)
        return {"rank": get_tensor_model_parallel_rank(), **result}

    def nv_alignment_timing(self) -> dict[str, Any]:
        """Time eager fixed-routing layer calls with interleaved native/stable controls."""
        import statistics

        import stable_alignment
        import torch
        from vllm.distributed import get_tensor_model_parallel_rank

        if self._nv_moe.enabled or not stable_alignment.ENABLED:
            raise ValueError("Timing requires disarmed captures and the stable launch policy")
        rows = []
        try:
            with torch.inference_mode():
                for case in self._nv_moe.cases:
                    if case["kind"] != "fixed-routing":
                        continue
                    fn, args, kwargs = case["original"], case["args"], case["kwargs"]
                    for enabled in (False, True):
                        stable_alignment.ENABLED = enabled
                        for _ in range(5):
                            fn(*args, **kwargs)
                    samples = {key: [] for key in ("native_a", "stable", "native_b")}
                    for trial in range(16):
                        order = ("native_a", "stable", "native_b")
                        if trial % 2:
                            order = tuple(reversed(order))
                        for key in order:
                            stable_alignment.ENABLED = key == "stable"
                            start = torch.cuda.Event(enable_timing=True)
                            end = torch.cuda.Event(enable_timing=True)
                            start.record()
                            for _ in range(10):
                                fn(*args, **kwargs)
                            end.record()
                            end.synchronize()
                            samples[key].append(start.elapsed_time(end) / 10)
                    medians = {k: statistics.median(v) for k, v in samples.items()}
                    control = abs(medians["native_a"] / medians["native_b"] - 1)
                    rows.append({"name": case["name"], "milliseconds": samples,
                                 "median_ms": medians, "control_relative_difference": control,
                                 "control_pass_2pct": control <= .02,
                                 "stable_overhead_relative": medians["stable"] /
                                 ((medians["native_a"] + medians["native_b"]) / 2) - 1})
        finally:
            stable_alignment.ENABLED = True
        return {"rank": get_tensor_model_parallel_rank(), "cases": rows,
                "scope": "eager 256-row fixed-routing layer; no API or decode-speed claim"}
