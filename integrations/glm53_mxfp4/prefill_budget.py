"""Select a bounded scheduling budget between idle groups in disposable engines only."""

from __future__ import annotations

import functools
import hashlib
import inspect
import os
from pathlib import Path
from typing import Any

from prefill_admission import CORE_SHA, SCHEDULER_SHA

_INSTALLED = False
BUDGETS = (2048, 4096)


def configure(engine: Any, budget: int) -> dict[str, Any]:
    """Change only the per-forward scheduler cap within preallocated worker capacity."""
    if type(budget) is not int or budget not in BUDGETS:
        raise ValueError("Only the frozen 2,048/4,096 budgets are supported")
    if engine.has_work() or not engine.input_queue.empty() or engine.batch_queue:
        raise RuntimeError("Change budget only between completed idle request groups")
    config = engine.vllm_config.scheduler_config
    if (config.max_num_batched_tokens != 4096 or config.max_num_seqs != 16
            or engine.vllm_config.speculative_config is not None):
        raise ValueError("Require fixed 4,096-token capacity, 16 sequences and no speculation")
    engine.scheduler.max_num_scheduled_tokens = budget
    return report(engine)


def report(engine: Any) -> dict[str, Any]:
    """Attest the active scheduler cap separately from fixed worker allocation capacity."""
    return {"budget_tokens": engine.scheduler.max_num_scheduled_tokens,
            "allocation_capacity_tokens": engine.vllm_config.scheduler_config.max_num_batched_tokens,
            "engine_idle": not engine.has_work(), "only_idle_fresh_text_prefills": True,
            "scheduler_source_sha256": SCHEDULER_SHA, "engine_core_sha256": CORE_SHA}


def install() -> None:
    """Install only in an explicitly opted-in engine with the audited native sources."""
    global _INSTALLED
    if _INSTALLED or os.environ.get("MXWAVE_PREFILL_BUDGET") != "1":
        return
    from budget_dispatch import install as install_dispatch
    from prefill_admission import install as install_admission
    from vllm.v1.core.sched import scheduler
    from vllm.v1.engine import core

    for module, digest in ((core, CORE_SHA), (scheduler, SCHEDULER_SHA)):
        if hashlib.sha256(Path(inspect.getfile(module)).read_bytes()).hexdigest() != digest:
            raise ValueError("Budget experiment requires the pinned scheduler and engine")
    install_admission()
    install_dispatch()
    original = core.EngineCore.collective_rpc

    @functools.wraps(original)
    def rpc(engine: Any, method: Any, timeout: float | None = None, args: tuple = (),
            kwargs: dict[str, Any] | None = None) -> Any:
        if method not in ("mxwave_budget_configure", "mxwave_budget_report"):
            return original(engine, method, timeout, args, kwargs)
        if kwargs or (len(args) != (1 if method == "mxwave_budget_configure" else 0)):
            raise ValueError("Invalid budget control arguments")
        if method == "mxwave_budget_configure":
            configure(engine, args[0])
            admission = original(engine, "mxwave_admission_configure", timeout, (10,), None)[0]
        else:
            admission = original(engine, "mxwave_admission_report", timeout, (), None)[0]
        return [{**admission, **report(engine)}]

    core.EngineCore.collective_rpc = rpc
    _INSTALLED = True
