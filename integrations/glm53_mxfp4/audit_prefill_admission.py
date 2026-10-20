"""Exercise the actual pinned engine input-drain/handler with bounded CPU-only stand-ins."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import queue
import threading
import time
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any

from prefill_admission import CORE_SHA, SCHEDULER_SHA, install


def main() -> None:
    """Verify native queue integration without constructing a model, KV cache or CUDA context."""
    import torch
    from vllm.v1.core.sched.request_queue import FCFSRequestQueue
    from vllm.v1.engine import EngineCoreRequestType as Kind
    from vllm.v1.engine import core

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    assert not args.report.exists()
    torch.set_num_threads(1)
    install()
    cases = []

    class Input(queue.Queue):
        def __init__(self) -> None:
            super().__init__()
            self.collection_entered = threading.Event()

        def get(self, block: bool = True, timeout: float | None = None) -> Any:
            if timeout is not None:
                self.collection_entered.set()
            return super().get(block=block, timeout=timeout)

    def request(index: int, tokens: int = 256) -> Any:
        return SimpleNamespace(request_id=str(index), num_prompt_tokens=tokens,
            num_computed_tokens=0, num_output_tokens=0, pooling_params=None)

    def engine() -> Any:
        result = core.EngineCoreProc.__new__(core.EngineCoreProc)
        scheduler = SimpleNamespace(waiting=FCFSRequestQueue(), skipped_waiting=FCFSRequestQueue(),
            running=[], connector=None, ec_connector=None, num_waiting_for_streaming_input=0,
            max_num_running_reqs=8, max_num_scheduled_tokens=2048,
            pause_state=core.PauseState.UNPAUSED)
        scheduler.has_requests = lambda: bool(scheduler.waiting or scheduler.running)
        result.scheduler = scheduler
        result.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(
            data_parallel_size=1, pipeline_parallel_size=1))
        result.engines_running = False
        result.batch_queue = None
        result.process_input_queue_block = False
        result._idle_state_callbacks = []
        result.aborts_queue = queue.Queue()
        result.input_queue = Input()
        result.shutdown_state = core.EngineShutdownState.RUNNING
        result.handled = []

        def add(value: Any, req: Any, request_wave: int = 0) -> None:
            value.handled.append(req.request_id)
            value.scheduler.waiting.add_request(req)

        def abort(value: Any, request_ids: list[str]) -> None:
            for req in list(value.scheduler.waiting):
                if req.request_id in request_ids:
                    value.scheduler.waiting.remove_request(req)

        result.add_request = MethodType(add, result)
        result.abort_requests = MethodType(abort, result)
        result.model_executor = SimpleNamespace(collective_rpc=lambda *args: ["native_rpc"])
        return result

    for mode in ("zero", "collect", "decode", "full_tokens", "full_requests", "abort", "wakeup"):
        target = engine()
        target.collective_rpc("mxwave_admission_configure", args=(0. if mode == "zero" else 10.,))
        target.input_queue.put((Kind.ADD, (request(0, 2048 if mode == "full_tokens" else 256), 0)))
        if mode == "decode":
            target.scheduler.running.append(request(-1))
        if mode == "full_requests":
            for i in range(1, 8):
                target.input_queue.put((Kind.ADD, (request(i), 0)))

        def produce(target: Any = target, mode: str = mode) -> None:
            if not target.input_queue.collection_entered.wait(2):
                return
            if mode == "abort":
                target.input_queue.put((Kind.ABORT, ["0"]))
                target.input_queue.put((Kind.ADD, (request(1), 0)))
            elif mode == "wakeup":
                target.input_queue.put((Kind.WAKEUP, None))
                target.input_queue.put((Kind.ADD, (request(1), 0)))
            else:
                for i in (1, 2):
                    target.input_queue.put((Kind.ADD, (request(i), 0)))

        producer = None
        if mode in ("collect", "abort", "wakeup"):
            producer = threading.Thread(target=produce)
            producer.start()
        started = time.monotonic()
        target._process_input_queue()
        elapsed = time.monotonic() - started
        if producer is not None:
            producer.join(3)
            assert not producer.is_alive()
        value = target.collective_rpc("mxwave_admission_report")[0]
        waiting = [r.request_id for r in target.scheduler.waiting]
        if mode == "collect":
            assert waiting == ["0", "1", "2"] and value["events"][0]["additional_messages"] == 2
        elif mode in ("abort", "wakeup"):
            assert target.input_queue.qsize() == 1
            assert value["events"][0]["end_reason"] == "control_or_abort"
            assert waiting == ([] if mode == "abort" else ["0"])
        elif mode == "decode":
            assert value["events"] == [] and not target.input_queue.collection_entered.is_set()
        else:
            assert not target.input_queue.collection_entered.is_set()
            assert value["events"][0]["additional_messages"] == 0
        try:
            target.collective_rpc("mxwave_admission_configure", args=(3.,))
        except RuntimeError:
            pass
        else:
            raise AssertionError("Busy engine accepted an admission configuration change")
        assert target.collective_rpc("unmodified_worker_rpc") == ["native_rpc"]
        cases.append({"mode": mode, "status": "passed", "waiting_ids": waiting,
            "native_added_ids": target.handled, "elapsed_seconds": elapsed, **value})
    assert not torch.cuda.is_initialized()
    result = {"status": "passed", "cuda_initialized": False, "model_constructed": False,
        "native_input_drain_and_message_handler": True,
        "scheduler_and_add_request_replaced_by_bounded_standins": True,
        "engine_core_sha256": CORE_SHA, "scheduler_sha256": SCHEDULER_SHA,
        "native_module_path": inspect.getfile(core), "cases": cases,
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__), Path(__file__).with_name("prefill_admission.py"))},
        "limits": ["Native input drain/handler/control integration only; no real scheduling/GPU inference.",
                   "The actual scheduler object and model executor are bounded stand-ins.",
                   "No throughput, TTFT, quality or native model compatibility claim."]}
    args.report.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}))


if __name__ == "__main__":
    main()
