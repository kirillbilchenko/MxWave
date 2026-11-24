"""Exercise the actual probe worker on tiny CUDA graphs before a full-model outage."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import vllm.distributed
from decode_state_worker import DecodeStateWorker


def fixture(nondeterministic: bool, nondefault_stream: bool,
            worker_type: type[DecodeStateWorker] = DecodeStateWorker,
            completion_policy: str | None = None) -> dict[str, Any]:
    """Validate graph hook cleanup, exact tracked restoration and changing-workspace detection."""
    device = torch.device("cuda:0")
    caches = [torch.zeros(4, 8, device=device) for _ in range(68)]
    ids = torch.arange(4, dtype=torch.int32, device=device)
    inputs = SimpleNamespace(input_ids=ids, positions=ids.to(torch.int64) + 258)
    modules = {
        f"language_model.model.layers.{i}.self_attn": SimpleNamespace(kv_cache=caches[2*i:2*i+2])
        for i in range(34)
    }
    metadata = {name: SimpleNamespace(non_spec_state_indices_tensor=ids,
                                     num_decodes=4) for name in modules}
    batch = SimpleNamespace(num_tokens=4, num_reqs=4, has_prefill=False,
        idx_mapping_np=np.arange(4), prefill_len_np=np.full(4, 258),
        input_ids=inputs.input_ids, positions=inputs.positions,
        req_ids=[f"fixture{i}" for i in range(4)])
    output = torch.zeros(4, 8, device=device)
    counter = torch.zeros((), device=device)

    def operation() -> None:
        if nondeterministic:
            # Deliberately untracked workspace: the probe must report this drift.
            counter.add_(1)
        for cache in caches:
            cache.add_(inputs.input_ids[:, None])
        output.copy_(caches[0])
        if nondeterministic:
            output.add_(counter)

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(3):
            operation()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        operation()
    for cache in caches:
        cache.zero_()
    counter.zero_()
    torch.cuda.synchronize()

    class Manager:
        """Minimal instance with the same bound-method dispatch used by the pinned runner."""

        def __init__(self) -> None:
            self._graphs_captured = True
            self.graphs = {4: graph}

        def run_fullgraph(self, descriptor: Any) -> torch.Tensor:
            """Replay one actual captured CUDA graph and return its retained output."""
            graph.replay()
            return output

    class ModelState:
        """Supply actual CUDA metadata tensors through the worker's preparation wrapper."""

        def prepare_attn(self, supplied_batch: Any) -> dict[str, Any]:
            """Return the fixed batch metadata without modifying any numerical input."""
            assert supplied_batch is batch
            return metadata

    manager, model_state = Manager(), ModelState()
    runner = SimpleNamespace(cudagraph_manager=manager, model_state=model_state,
        speculator=None, compilation_config=SimpleNamespace(static_forward_context=modules),
        kv_caches=caches, input_buffers=inputs,
        req_states=SimpleNamespace(all_token_ids=SimpleNamespace(
            gpu=torch.arange(4 * 258, device=device).reshape(4, 258))))
    worker = worker_type()
    worker.model_runner = runner
    temporary = tempfile.TemporaryDirectory(prefix="mxwave-decode-fixture-values-")
    worker._mxwave_values_directory = Path(temporary.name)
    retained: list[dict[str, Any]] = []
    worker._mxwave_write_decode = lambda report: retained.append(json.loads(json.dumps(report)))
    original_graph, original_prepare = manager.run_fullgraph, model_state.prepare_attn
    armed = (worker.mxwave_decode_probe_arm("fixture", 0) if completion_policy is None
             else worker.mxwave_decode_probe_arm("fixture", 0, completion_policy))
    assert armed["status"] == "armed", armed
    chosen = stream if nondefault_stream else torch.cuda.current_stream()
    with torch.cuda.stream(chosen):
        model_state.prepare_attn(batch)
        result = manager.run_fullgraph(None)
    chosen.synchronize()
    report = worker.mxwave_decode_probe_finish()
    assert report["status"] == "complete" and len(retained) == 1
    assert manager.run_fullgraph == original_graph and model_state.prepare_attn == original_prepare
    assert len(report["replays"]) == 5 and report["served_iteration"] == 4
    assert all(r["initial_bytes_exact"] for r in report["replays"])
    assert len(report["active_kda_state_before"]) == 68
    assert report["all_hidden_outputs_exact"] is not nondeterministic
    assert report["all_resulting_storage_bytes_exact"]
    assert torch.equal(caches[0], ids[:, None].expand(4, 8).float())
    if completion_policy is not None:
        for index, replay in enumerate(report["replays"]):
            observed = replay["observation"]
            assert len(observed["hidden"]) == 1 and len(observed["active_kda_post"]) == 68
            assert all(row["metrics"]["bytes_exact"] for row in observed["active_kda_post"])
            assert observed["first_changed_active_cache"] is None
            metric = observed["hidden"][0]["metrics"]
            assert metric["max_absolute_error_finite"] == (index if nondeterministic else 0)
            assert metric["bytes_exact"] is (index == 0 or not nondeterministic)
        assert len(list(Path(temporary.name).glob("*.bin.gz"))) == 5
    if not nondeterministic:
        assert torch.equal(result, caches[0])
    assert worker.mxwave_decode_probe_finish()["status"] == "not_armed"
    temporary.cleanup()
    return {"nondeterministic_workspace": nondeterministic,
            "nondefault_stream": nondefault_stream, "status": "passed",
            "replays": len(report["replays"]),
            "all_hidden_outputs_exact": report["all_hidden_outputs_exact"],
            "tracked_cache_advanced_once": True,
            "snapshot_backing": report["snapshot_backing"],
            "retained_gpu_snapshot_bytes": report["retained_gpu_snapshot_bytes"],
            "snapshot_image_bytes": report["snapshot_image_bytes"]}


def main() -> None:
    """Run both stream modes and prove the observer detects deliberately changing scratch."""
    vllm.distributed.get_tensor_model_parallel_rank = lambda: 0
    cases = [fixture(drift, stream) for drift in (False, True) for stream in (False, True)]
    print(json.dumps({"status": "passed", "cases": cases,
                      "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated()}))
    print("MXWAVE_DECODE_FIXTURE_PASSED")


if __name__ == "__main__":
    main()
