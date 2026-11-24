"""Replay real layer-three MLA operands with explicit shared-plan and private-workspace scope."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from decode_boundary_capture import boundary_metrics
from decode_state_disk import CHUNK_BYTES, DiskByteSnapshots
from decode_state_replay import tensor_digest
from decode_state_values import save_tensor
from decode_state_values_worker import DecodeStateValuesWorker
from mla_internal_capture import MLAInternalCapture, layout, patch_manager

_INSTALLED = False


def install() -> None:
    """Require the inspected native sources before inserting capture-only observers."""
    global _INSTALLED
    if _INSTALLED:
        return
    import flashinfer.mla._core as wrapper_module
    import vllm.model_executor.layers.attention.mla_attention as mla_module
    import vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90 as backend
    import vllm.v1.worker.gpu.cudagraph_utils as manager_module

    expected = {
        mla_module: "a309ca752b01e85360b268fba56245b816ff245340e9e361166ee96b99d93a94",
        backend: "4449ea25921dcae1ec7988581ff26a6ff5a9e5d6d4a0ac4a68a0017ea108f136",
        wrapper_module: "58d06e61aa753f938bf5682640767541dbd4cb6989340abcdc8f6f90792db6e7",
        manager_module: "c183937e6eb5b9c28c79d98fb4c64f562e7649d5f6d65743e6640b2f378ecf9f",
    }
    if any(hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != digest
           for module, digest in expected.items()):
        raise ValueError("Native MLA, backend, wrapper or graph manager source changed")
    patch_manager(manager_module.ModelCudaGraphManager, backend)
    _INSTALLED = True


def constants(trace: MLAInternalCapture) -> dict[str, Any]:
    """Fingerprint complete immutable BMM operands outside CUDA capture and retain their views."""
    return {name: {**tensor_digest(value), "layout": layout(value)}
            for name, value in trace.constants.items()}


def runtime_scales(trace: MLAInternalCapture) -> dict[str, Any]:
    """Keep current device/host scales distinct from scalar values baked into capture."""
    target = trace.target
    return {
        "device": {name: tensor_digest(getattr(target, name))
                   for name in ("_k_scale", "_q_scale", "_v_scale")
                   if isinstance(getattr(target, name, None), torch.Tensor)},
        "host": {name: getattr(target, name, None)
                 for name in ("_k_scale_float", "_q_scale_float", "_v_scale_float")},
    }


class MLAInternalWorker(DecodeStateValuesWorker):
    """Use the existing five-replay/fifth-result rule with a replacement internal trace."""

    def mxwave_mla_internal_ready(self) -> dict[str, Any]:
        """Attest to bounded real-operation copies and exact restoration of every Python hook."""
        trace = self.model_runner.cudagraph_manager._mxwave_mla_internal_trace
        return {"status": "ready", **trace.metadata(), "constants": constants(trace),
                "runtime_scales": runtime_scales(trace),
                "broad_boundary_trace_absent": not hasattr(
                    self.model_runner.cudagraph_manager, "_mxwave_boundary_trace")}

    def mxwave_mla_internal_arm(self, label: str, graph_step: int,
                               workspace_policy: str) -> dict[str, Any]:
        """Allocate both policies' SSD images; only the private-workspace restoration differs."""
        if graph_step != 0 or workspace_policy not in ("tracked_only", "restore_private"):
            raise ValueError("Require first-decode and one frozen private-workspace policy")
        trace: MLAInternalCapture = self.model_runner.cudagraph_manager._mxwave_mla_internal_trace
        if trace.methods or trace.nbytes + trace.staging_bytes + CHUNK_BYTES > 32 * 1024**2:
            raise ValueError("Require removed Python hooks and the complete32 MiB budget")
        result = super().mxwave_decode_probe_arm(label, graph_step, "normal")
        if result["status"] != "armed":
            return result
        state = self._mxwave_decode_probe
        private = None
        try:
            wrapper = trace.state.wrapper
            plan_buffers = [wrapper._qo_indptr_buf, wrapper._kv_indptr_buf,
                            wrapper._kv_indices_buf, wrapper._kv_len_arr_buf]
            extra = [tensor for tensor in plan_buffers if (
                tensor.untyped_storage().data_ptr(), tensor.untyped_storage().nbytes()
            ) not in state["snapshots"].images]
            plan_bytes = 0
            if extra:
                plan_images = DiskByteSnapshots(extra, 32 * 1024**2, Path("/tmp"))
                plan_bytes = plan_images.payload_bytes
                state["snapshots"].absorb(plan_images)
            workspace = [wrapper._float_workspace_buffer, wrapper._int_workspace_buffer]
            private = DiskByteSnapshots(workspace, 160 * 1024**2, Path("/tmp"))
            state.update(workspace_policy=workspace_policy, private_workspace_images=private,
                         private_before_each_call=[], constant_reference=constants(trace))
            state["report"].update(workspace_policy=workspace_policy,
                mla_internal_capture=trace.metadata(), constants_before=state["constant_reference"],
                runtime_scales_at_admission=runtime_scales(trace),
                shared_plan_buffers_always_tracked_bytes=plan_bytes,
                private_workspace_payload_bytes=private.payload_bytes,
                private_workspace_SSD_image_bytes=2 * private.payload_bytes,
                private_images_allocated_in_both_policies=True,
                maximum_transfer_chunk_bytes=CHUNK_BYTES, retained_gpu_snapshot_bytes=0,
                captured_plan_info=trace.captured_plan,
                snapshot_payload_bytes=state["snapshots"].payload_bytes,
                snapshot_image_bytes=2 * state["snapshots"].payload_bytes,
                diagnostic_limits=[
                    "Intrusive full-model operation observations, not serving KL or timing.",
                    "The broad multi-layer boundary capture is replaced, not added.",
                    "Both policies reset original cache/input/metadata plus shared device plan buffers.",
                    "Only restore_private additionally resets the real float/int workspace before each call.",
                    "Capture copies plus conservative staging and8 MiB transfer allowance stay within32 MiB.",
                    "Non-reproduction after instrumentation does not establish a causal correction.",
                ])
            return {**result, "workspace_policy": workspace_policy,
                    "private_workspace_payload_bytes": private.payload_bytes}
        except Exception as exc:  # noqa: BLE001 -- fail admission without killing reference workers.
            if private is not None:
                private.close()
            super().mxwave_decode_probe_finish()
            return {"status": "failed", "error": repr(exc), "rank": result["rank"]}

    def _mxwave_decode_call(self, state: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        """Restore actual private bytes only for the declared control, never choose a replay."""
        private: DiskByteSnapshots = state["private_workspace_images"]
        if state["workspace_policy"] == "restore_private":
            private.restore_before()
        exact = private.before_exact()
        state["private_before_each_call"].append(exact)
        if state["workspace_policy"] == "restore_private" and not exact:
            raise RuntimeError("Private-workspace byte restoration failed")
        return super()._mxwave_decode_call(state, *args, **kwargs)

    def _mxwave_decode_observer(
        self, state: dict[str, Any],
    ) -> Callable[[int, Any], dict[str, Any]]:
        """Retain every internal operand/result, exact weights and private-buffer observations."""
        trace: MLAInternalCapture = self.model_runner.cudagraph_manager._mxwave_mla_internal_trace
        private: DiskByteSnapshots = state["private_workspace_images"]
        private.capture_before()  # Actual request plan is built now, not at admission.
        state["report"]["runtime_plan_info"] = list(trace.state.wrapper._plan_info)
        base = super()._mxwave_decode_observer(state)
        references: list[list[torch.Tensor]] = []
        directory = getattr(self, "_mxwave_values_directory", Path("/traces"))
        prefix = f"mla-internal-{state['label']}-r{state['rank']}"

        def observe(iteration: int, result: Any) -> dict[str, Any]:
            value = base(iteration, result)
            events = []
            for event_index, event in enumerate(trace.events):
                if iteration == 0:
                    references.append([])
                tensors = []
                for tensor_index, tensor in enumerate(event["tensors"]):
                    current = tensor["value"].detach().cpu().clone()
                    if iteration == 0:
                        references[event_index].append(current)
                    tensors.append({"path": tensor["path"],
                        "artifact": save_tensor(directory,
                            f"{prefix}-i{iteration}-e{event_index}-t{tensor_index}.bin.gz", current),
                        "metrics": boundary_metrics(current, references[event_index][tensor_index])})
                events.append({"name": event["name"], "category": event["category"],
                               "parameters": event["parameters"], "tensors": tensors})
            constant_values = constants(trace)
            if constant_values != state["constant_reference"]:
                raise RuntimeError("Immutable MLA BMM operand bytes or layout changed")
            if iteration == 0:
                private.capture_reference_after()
            value.update(mla_internal_events=events, constants_after=constant_values,
                runtime_scales=runtime_scales(trace),
                private_before_original_exact=state["private_before_each_call"][iteration],
                private_changed_storages_vs_first=private.changed_storages(),
                runtime_plan_info=list(trace.state.wrapper._plan_info),
                first_changed_internal_event=next((event["name"] for event in events
                    if any(not tensor["metrics"]["bytes_exact"] for tensor in event["tensors"])), None))
            return value
        return observe

    def mxwave_mla_internal_finish(self) -> dict[str, Any]:
        """Disarm base replay wrappers and remove only this probe's owned private SSD images."""
        state = getattr(self, "_mxwave_decode_probe", None)
        private = state.get("private_workspace_images") if state else None
        try:
            return super().mxwave_decode_probe_finish()
        finally:
            if private is not None:
                private.close()
