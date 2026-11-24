"""Localize reference graph drift with captured operands and a workspace-reset control."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from decode_boundary_capture import BoundaryCapture, boundary_metrics, install
from decode_state_disk import DiskByteSnapshots
from decode_state_values import save_tensor
from decode_state_values_worker import DecodeStateValuesWorker

install()


class DecodeBoundaryWorker(DecodeStateValuesWorker):
    """Preserve original source guards, actual graph execution and fixed replay selection."""

    def mxwave_boundary_ready(self) -> dict[str, Any]:
        """Attest that real retained graph copies were inserted and Python methods restored."""
        trace = self.model_runner.cudagraph_manager._mxwave_boundary_trace
        return {"status": "ready", **trace.metadata()}

    def mxwave_boundary_arm(self, label: str, graph_step: int,
                            workspace_policy: str) -> dict[str, Any]:
        """Select original tracked state or additionally reset the stock Marlin semaphore buffers."""
        if workspace_policy not in ("tracked_only", "stock_marlin_locks") or graph_step != 0:
            raise ValueError("Require a first-decode probe and one frozen workspace policy")
        trace: BoundaryCapture = self.model_runner.cudagraph_manager._mxwave_boundary_trace
        if not trace.events or trace.methods:
            raise ValueError("Require complete captured copies with Python wrappers removed")
        result = super().mxwave_decode_probe_arm(label, graph_step, "normal")
        if result["status"] == "armed":
            state = self._mxwave_decode_probe
            state["workspace_policy"] = workspace_policy
            state["report"].update(workspace_policy=workspace_policy,
                                  boundary_capture=trace.metadata())
            result.update(workspace_policy=workspace_policy)
        return result

    def _mxwave_decode_observer(
        self, state: dict[str, Any],
    ) -> Callable[[int, Any], dict[str, Any]]:
        """Retain every captured boundary tensor and independently measurable error."""
        trace: BoundaryCapture = self.model_runner.cudagraph_manager._mxwave_boundary_trace
        if state["workspace_policy"] == "stock_marlin_locks":
            extra = [t for t in trace.locks if (t.untyped_storage().data_ptr(),
                     t.untyped_storage().nbytes()) not in state["snapshots"].images]
            size = 0
            if extra:
                additional = DiskByteSnapshots(extra, 64 * 1024**2, Path("/tmp"))
                size = additional.payload_bytes
                state["snapshots"].absorb(additional)
            state["report"].update(extra_workspace_payload_bytes=size,
                                  snapshot_payload_bytes=state["snapshots"].payload_bytes,
                                  snapshot_image_bytes=2 * state["snapshots"].payload_bytes,
                                  restored_workspace_scope="captured stock Marlin layer-3/4 semaphore buffers")
        else:
            state["report"]["extra_workspace_payload_bytes"] = 0
        base = super()._mxwave_decode_observer(state)
        references: list[list[torch.Tensor]] = []
        directory = getattr(self, "_mxwave_values_directory", Path("/traces"))
        prefix = f"boundary-{state['label']}-r{state['rank']}"

        def observe(iteration: int, result: Any) -> dict[str, Any]:
            value = base(iteration, result)
            events = []
            for event_index, event in enumerate(trace.events):
                items = []
                if iteration == 0:
                    references.append([])
                for tensor_index, tensor in enumerate(event["tensors"]):
                    current = tensor["value"].detach().cpu().clone()
                    if iteration == 0:
                        references[event_index].append(current)
                    name = f"{prefix}-i{iteration}-e{event_index}-t{tensor_index}.bin.gz"
                    items.append({"path": tensor["path"],
                        "artifact": save_tensor(directory, name, current),
                        "metrics": boundary_metrics(
                            current, references[event_index][tensor_index])})
                events.append({"name": event["name"], "category": event["category"],
                               "parameters": event["parameters"], "tensors": items})
            value["boundaries"] = events
            value["first_changed_nonworkspace_event"] = next((e["name"] for e in events
                if e["category"] != "workspace"
                and any(not t["metrics"]["bytes_exact"] for t in e["tensors"])), None)
            return value

        return observe
