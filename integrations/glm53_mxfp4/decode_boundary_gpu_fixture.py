"""Exercise actual graph copies and workspace restoration before loading GLM weights."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import vllm.distributed
from decode_boundary_capture import patch_manager
from decode_boundary_fixture_model import make_model
from decode_boundary_worker import DecodeBoundaryWorker


def fixture(drift: bool, nondefault: bool, policy: str) -> dict[str, Any]:
    """Detect semaphore-driven drift and remove it only with the explicit reset control."""
    device = torch.device("cuda")
    model, ops, caches, locks = make_model(device, drift)
    ids = torch.arange(4, device=device, dtype=torch.int32)
    inputs = SimpleNamespace(input_ids=ids, positions=ids.to(torch.int64) + 258)
    metadata = {f"language_model.model.layers.{i}.self_attn": SimpleNamespace(
        non_spec_state_indices_tensor=ids, num_decodes=4) for i in range(34)}
    modules = {name: SimpleNamespace(kv_cache=caches[2*i:2*i+2])
               for i, name in enumerate(metadata)}
    batch = SimpleNamespace(num_tokens=4, num_reqs=4, has_prefill=False,
        idx_mapping_np=np.arange(4), prefill_len_np=np.full(4, 258),
        input_ids=inputs.input_ids, positions=inputs.positions,
        req_ids=[f"fixture{i}" for i in range(4)])

    @dataclass(frozen=True)
    class Descriptor:
        """Keep the same retained-graph key property as the actual pinned manager."""

        num_tokens: int = 4

    class Manager:
        """Capture real operations through the same initial-capture adapter as GLM."""

        def __init__(self) -> None:
            self._graphs_captured = True
            self.graphs: dict[Descriptor, Any] = {}

        def capture(self, module: Any) -> None:
            """Warm up without observation, then capture one actual four-token graph."""
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                for _ in range(3):
                    module(inputs.input_ids, inputs.positions)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self.output = module(inputs.input_ids, inputs.positions)
            self.graphs[Descriptor()] = graph

        def run_fullgraph(self, descriptor: Any) -> torch.Tensor:
            """Replay the observed graph and return its retained hidden tensor."""
            self.graphs[Descriptor()].replay()
            return self.output

    class ModelState:
        """Use the real worker's metadata and batch wrappers."""

        def prepare_attn(self, supplied: Any) -> dict[str, Any]:
            """Return fixed active-state metadata."""
            assert supplied is batch
            return metadata

    patch_manager(Manager, ops)
    manager, model_state = Manager(), ModelState()
    manager.capture(model)
    for tensor in caches + locks:
        tensor.zero_()
    torch.cuda.synchronize()
    runner = SimpleNamespace(cudagraph_manager=manager, model_state=model_state,
        speculator=None, compilation_config=SimpleNamespace(static_forward_context=modules),
        kv_caches=caches, input_buffers=inputs,
        req_states=SimpleNamespace(all_token_ids=SimpleNamespace(
            gpu=torch.arange(4 * 258, device=device).reshape(4, 258))))
    worker = DecodeBoundaryWorker()
    worker.model_runner = runner
    retained = []
    worker._mxwave_write_decode = lambda report: retained.append(json.loads(json.dumps(report)))
    with tempfile.TemporaryDirectory(prefix="mxwave-boundary-fixture-") as directory:
        worker._mxwave_values_directory = Path(directory)
        assert worker.mxwave_boundary_ready()["python_wrappers_removed"]
        original_graph, original_prepare = manager.run_fullgraph, model_state.prepare_attn
        assert worker.mxwave_boundary_arm("fixture", 0, policy)["status"] == "armed"
        stream = torch.cuda.Stream() if nondefault else torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            model_state.prepare_attn(batch)
            manager.run_fullgraph(None)
        stream.synchronize()
        report = worker.mxwave_decode_probe_finish()
        assert report["status"] == "complete" and len(retained) == 1
        assert manager.run_fullgraph == original_graph and model_state.prepare_attn == original_prepare
        assert len(report["replays"]) == 5 and report["served_iteration"] == 4
        exact = not drift or policy == "stock_marlin_locks"
        assert report["all_hidden_outputs_exact"] == exact
        assert report["all_resulting_storage_bytes_exact"] == exact
        for index, row in enumerate(report["replays"]):
            obs = row["observation"]
            expected_first = None if exact or index == 0 else "layer3.moe_gemm#0/output"
            assert row["initial_bytes_exact"]
            assert obs["first_changed_nonworkspace_event"] == expected_first
            first_input = next(x for x in obs["boundaries"]
                               if x["name"] == "layer3.moe_gemm#0/input")
            assert all(x["metrics"]["bytes_exact"] for x in first_input["tensors"])
        assert worker.mxwave_decode_probe_finish()["status"] == "not_armed"
    return {"status": "passed", "drift": drift, "nondefault_stream": nondefault,
            "workspace_policy": policy, "graph_copies_checked": True,
            "first_changed_operation_checked": True, "workspace_control_checked": True,
            "python_wrappers_removed": True, "tracked_bytes_exact": True,
            "capture_bytes": manager._mxwave_boundary_trace.nbytes}


def main() -> None:
    """Check both streams, both policies and deterministic/changing exposed semaphore buffers."""
    vllm.distributed.get_tensor_model_parallel_rank = lambda: 0
    cases = [fixture(drift, stream, policy) for policy in ("tracked_only", "stock_marlin_locks")
             for drift in (False, True) for stream in (False, True)]
    print(json.dumps({"status": "passed", "cases": cases,
                      "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated()}))
    print("MXWAVE_DECODE_FIXTURE_PASSED")


if __name__ == "__main__":
    main()
