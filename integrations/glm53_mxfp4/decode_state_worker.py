"""Intrusive decode-graph replay with fixed KV/input bytes in disposable reference engines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from decode_state_disk import DiskByteSnapshots
from decode_state_replay import replay_identical_state, tensor_digest, tensor_values
from graph_quality_worker import GraphQualityWorker

MAX_CACHE_BYTES = 4608 * 1024**2
MAX_METADATA_BYTES = 64 * 1024**2


def metadata_tensors(value: Any, prefix: str = "", depth: int = 0) -> list[tuple[str, torch.Tensor]]:
    """Inspect bounded metadata containers; never traverse modules or model weights."""
    if isinstance(value, torch.Tensor):
        return [(prefix, value)] if value.device.type == "cuda" else []
    if depth >= 4 or isinstance(value, torch.nn.Module):
        return []
    if isinstance(value, dict):
        entries = value.items()
    elif isinstance(value, (tuple, list)):
        entries = enumerate(value)
    elif hasattr(value, "__dict__"):
        entries = vars(value).items()
    else:
        return []
    return [item for key, child in entries
            for item in metadata_tensors(child, prefix + "/" + str(key), depth + 1)]


class DecodeStateWorker(GraphQualityWorker):
    """Retain all source proofs; replay only the explicitly armed first M4 graph."""

    def mxwave_decode_probe_arm(self, label: str, graph_step: int) -> dict[str, Any]:
        """Reserve both cache images before admission, failing closed without killing a worker."""
        from vllm.distributed import get_tensor_model_parallel_rank

        if not label.isalnum() or graph_step not in (0, 2):
            raise ValueError("Require an owned label and one of the two frozen target steps")
        if getattr(self, "_mxwave_decode_probe", None) is not None:
            raise ValueError("Finish the preceding probe before arming another")
        runner = self.model_runner
        manager = runner.cudagraph_manager
        if not manager._graphs_captured or not manager.graphs:
            raise ValueError("Require the original retained CUDA graphs")
        if runner.speculator is not None:
            raise ValueError("The reference diagnostic forbids speculative decoding")
        bindings = []
        for name, module in runner.compilation_config.static_forward_context.items():
            for index, tensor in enumerate(tensor_values(getattr(module, "kv_cache", None))):
                bindings.append((name + "/cache" + str(index), tensor))
        snapshots = None
        state: dict[str, Any] | None = None
        try:
            caches = tensor_values(runner.kv_caches)
            inputs = [t for t in vars(runner.input_buffers).values() if isinstance(t, torch.Tensor)]
            snapshots = DiskByteSnapshots(caches + inputs, MAX_CACHE_BYTES, Path("/tmp"))
            free, _ = torch.cuda.mem_get_info()
            if free < 128 * 1024**2:
                raise MemoryError("Require 128 MiB of headroom for bounded transfer buffers")
            if not bindings or not caches:
                raise ValueError("Require named cache views and original owned KV allocations")
            for _, value in bindings:
                storage = value.untyped_storage()
                if (storage.data_ptr(), storage.nbytes()) not in snapshots.images:
                    raise ValueError("A named cache is outside the owned snapshot allocations")
            state = {"label": label, "rank": get_tensor_model_parallel_rank(),
                "status": "armed", "target_graph_step": graph_step, "graph_calls_seen": 0,
                "snapshots": snapshots, "bindings": bindings,
                "original_graph": manager.run_fullgraph,
                "original_prepare_attn": runner.model_state.prepare_attn,
                "report": {"label": label, "rank": get_tensor_model_parallel_rank(),
                    "status": "armed", "intrusive_not_timing_or_accuracy": True,
                    "iterations": 5, "target_graph_step": graph_step,
                    "temporary_snapshot_payload_bytes": snapshots.payload_bytes,
                    "temporary_image_bytes": 2 * snapshots.payload_bytes,
                    "snapshot_backing": "ssd_mmap", "maximum_transfer_chunk_bytes": 8 * 1024**2,
                    "retained_gpu_snapshot_bytes": 0,
                    "cache_views": [{"name": n, "shape": list(v.shape),
                                     "dtype": str(v.dtype)} for n, v in bindings]}}

            def prepare_attn(*args: Any, **kwargs: Any) -> Any:
                result = state["original_prepare_attn"](*args, **kwargs)
                state["batch"], state["metadata"] = args[0], result
                return result

            def graph(*args: Any, **kwargs: Any) -> Any:
                index = state["graph_calls_seen"]
                state["graph_calls_seen"] += 1
                if state["status"] != "armed" or index != graph_step:
                    return state["original_graph"](*args, **kwargs)
                batch = state["batch"]
                if batch.num_tokens != 4 or batch.num_reqs != 4 or batch.has_prefill:
                    raise ValueError("The targeted graph must be a pure four-request decode")
                meta = metadata_tensors(state["metadata"])
                extra = []
                excluded = []
                for name, value in meta:
                    storage = value.untyped_storage()
                    if (storage.data_ptr(), storage.nbytes()) in snapshots.images:
                        continue
                    if storage.nbytes() <= 16 * 1024**2:
                        extra.append(value)
                    else:
                        excluded.append({"name": name, "shape": list(value.shape),
                                         "storage_bytes": storage.nbytes()})
                if extra:
                    additional = DiskByteSnapshots(extra, MAX_METADATA_BYTES, Path("/tmp"))
                    snapshots.absorb(additional)
                report = state["report"]
                report.update(status="replaying", prompt_hashes=[
                    tensor_digest(runner.req_states.all_token_ids.gpu[int(idx), :int(length)])
                    ["sha256"] for idx, length in zip(batch.idx_mapping_np,
                                                      batch.prefill_len_np, strict=True)],
                    input_ids=batch.input_ids[:4].cpu().tolist(),
                    positions=batch.positions[:4].cpu().tolist(),
                    req_ids=list(batch.req_ids), request_state_indices=batch.idx_mapping_np.tolist(),
                    metadata_excluded_large_allocations=excluded,
                    snapshot_payload_bytes=snapshots.payload_bytes,
                    snapshot_image_bytes=2 * snapshots.payload_bytes,
                    metadata=[{"name": n, **tensor_digest(v)} for n, v in meta
                              if v.numel() * v.element_size() <= 1024**2],
                    active_kda_state_before=self._mxwave_active_kda(state))
                try:
                    result, rows = replay_identical_state(
                        lambda: self._mxwave_decode_call(state, *args, **kwargs), snapshots, 5,
                        bindings, self._mxwave_decode_observer(state))
                    report.update(status="complete", replays=rows,
                        all_hidden_outputs_exact=all(r["outputs"] == rows[0]["outputs"]
                                                     for r in rows),
                        all_resulting_storage_bytes_exact=all(
                            r["changed_storage_count"] == 0 for r in rows),
                        served_iteration=4, final_state_not_rewound=True)
                    state["status"] = "complete"
                    self._mxwave_write_decode(report)
                    return result
                except BaseException as exc:
                    state["status"] = "failed"
                    report.update(status="failed", error=repr(exc))
                    self._mxwave_write_decode(report)
                    raise

            self._mxwave_decode_probe = state
            runner.model_state.prepare_attn = prepare_attn
            manager.run_fullgraph = graph
            return {k: v for k, v in state["report"].items() if k != "cache_views"}
        except Exception as exc:  # noqa: BLE001 -- allocation failure must not kill RPC workers.
            if state is not None:
                manager.run_fullgraph = state["original_graph"]
                runner.model_state.prepare_attn = state["original_prepare_attn"]
                state.clear()
            self._mxwave_decode_probe = None
            if snapshots is not None:
                snapshots.close()
            del snapshots
            return {"status": "failed", "rank": get_tensor_model_parallel_rank(),
                    "error": repr(exc)}

    def _mxwave_decode_call(self, state: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        """Invoke the original graph with its normal stream-completion behavior."""
        return state["original_graph"](*args, **kwargs)

    def _mxwave_decode_observer(self, state: dict[str, Any]) -> Any:
        """Leave value recording disabled in the original fingerprint-only diagnostic."""
        return None

    def _mxwave_active_kda(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        """Hash only selected logical KDA rows, preserving actual batch/request ordering."""
        rows = []
        for name, module in self.model_runner.compilation_config.static_forward_context.items():
            metadata = state["metadata"].get(name)
            if not hasattr(metadata, "non_spec_state_indices_tensor"):
                continue
            caches = tensor_values(getattr(module, "kv_cache", None))
            indices = metadata.non_spec_state_indices_tensor
            if len(caches) != 2 or indices is None or metadata.num_decodes != 4:
                raise ValueError("Expected exactly two KDA caches and four state indices")
            ids = indices[:4].to(dtype=torch.int64)
            if ids.ndim == 2 and ids.shape[1] == 1:
                ids = ids[:, 0]
            if ids.ndim != 1 or ids.numel() != 4:
                raise ValueError("Unsupported KDA state-index layout")
            for index, cache in enumerate(caches):
                selected = cache.index_select(0, ids)
                rows.append({"name": name + "/cache" + str(index),
                             "state_indices": ids.cpu().tolist(), **tensor_digest(selected)})
        if len(rows) != 68:
            raise ValueError("Expected both states from all 34 KDA layers")
        return rows

    def _mxwave_write_decode(self, report: dict[str, Any]) -> None:
        """Persist a plain, bounded report on this rank's owned trace mount."""
        path = Path("/traces") / f"decode-state-{report['label']}-r{report['rank']}.json"
        if path.exists():
            raise FileExistsError("Preserve the first diagnostic report")
        path.write_text(json.dumps(report, indent=2) + "\n")

    def mxwave_decode_probe_finish(self) -> dict[str, Any]:
        """Disarm wrappers and release temporary images after the single request is complete."""
        from vllm.distributed import get_tensor_model_parallel_rank

        state = getattr(self, "_mxwave_decode_probe", None)
        if state is None:
            return {"status": "not_armed", "rank": get_tensor_model_parallel_rank()}
        self.model_runner.cudagraph_manager.run_fullgraph = state["original_graph"]
        self.model_runner.model_state.prepare_attn = state["original_prepare_attn"]
        report = state["report"]
        self._mxwave_decode_probe = None
        snapshots = state.pop("snapshots", None)
        if snapshots is not None:
            snapshots.close()
        state.pop("bindings", None)
        # The wrappers close over state, so sever those bound closures before cache cleanup.
        state.clear()
        torch.cuda.empty_cache()
        return report
