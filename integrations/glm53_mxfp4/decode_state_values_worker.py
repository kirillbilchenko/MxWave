"""Record active post-state and raw hidden changes for a bounded reference-only probe."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from decode_state_replay import tensor_digest, tensor_values
from decode_state_values import save_tensor, value_metrics
from decode_state_worker import DecodeStateWorker


class DecodeStateValuesWorker(DecodeStateWorker):
    """Add bounded evidence to the exact-state replay without changing model arithmetic."""

    def mxwave_decode_probe_arm(self, label: str, graph_step: int,
                               completion_policy: str = "normal") -> dict[str, Any]:
        """Choose normal completion or explicit device synchronization before and after replay."""
        if completion_policy not in ("normal", "device_sync"):
            raise ValueError("Unknown frozen graph-completion policy")
        result = super().mxwave_decode_probe_arm(label, graph_step)
        if result["status"] == "armed":
            state = self._mxwave_decode_probe
            state["completion_policy"] = completion_policy
            state["report"]["completion_policy"] = completion_policy
            state["report"]["value_recording"] = "hidden_all_and_first_changed_active_cache"
            result.update(completion_policy=completion_policy)
        return result

    def _mxwave_decode_call(self, state: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        """Apply only the explicitly selected ordering control around the original graph."""
        if state["completion_policy"] == "device_sync":
            torch.cuda.synchronize()
        result = state["original_graph"](*args, **kwargs)
        if state["completion_policy"] == "device_sync":
            torch.cuda.synchronize()
        return result

    def _mxwave_decode_observer(
        self, state: dict[str, Any],
    ) -> Callable[[int, Any], dict[str, Any]]:
        """Compare all active KDA rows; retain raw hidden outputs and the earliest changed view."""
        directory = getattr(self, "_mxwave_values_directory", Path("/traces"))
        prefix = f"decode-values-{state['label']}-r{state['rank']}"
        selected = []
        for name, module in self.model_runner.compilation_config.static_forward_context.items():
            metadata = state["metadata"].get(name)
            if not hasattr(metadata, "non_spec_state_indices_tensor"):
                continue
            indices = metadata.non_spec_state_indices_tensor[:4].to(dtype=torch.int64).reshape(-1)
            if indices.numel() != 4 or metadata.num_decodes != 4:
                raise ValueError("Require exactly four active KDA state rows")
            match = re.search(r"\.layers\.(\d+)\.", name)
            if match is None:
                raise ValueError("Require the numeric model layer for active-state ordering")
            for index, cache in enumerate(tensor_values(getattr(module, "kv_cache", None))):
                selected.append((int(match[1]), index, name + "/cache" + str(index),
                                 cache, indices, indices.cpu()))
        selected.sort(key=lambda entry: entry[:2])
        if len(selected) != 68:
            raise ValueError("Require both active states from all 34 KDA layers")
        hidden_reference: list[torch.Tensor] = []

        def observe(iteration: int, result: Any) -> dict[str, Any]:
            hidden = []
            for output_index, tensor in enumerate(tensor_values(result)):
                current = tensor.detach().cpu().clone()
                if iteration == 0:
                    hidden_reference.append(current)
                hidden.append({"artifact": save_tensor(directory,
                    f"{prefix}-i{iteration}-hidden{output_index}.bin.gz", current),
                    "metrics": value_metrics(current, hidden_reference[output_index])})
            active = []
            first_changed = None
            for layer, cache_index, name, cache, ids, cpu_ids in selected:
                current = cache.index_select(0, ids).cpu()
                reference = state["snapshots"].reference_view(cache).index_select(0, cpu_ids)
                metrics = value_metrics(current, reference)
                active.append({"name": name, "layer": layer, "cache_index": cache_index,
                    "state_indices": cpu_ids.tolist(), "current": tensor_digest(current),
                    "reference": tensor_digest(reference), "metrics": metrics})
                if first_changed is None and not metrics["bytes_exact"]:
                    stem = f"{prefix}-i{iteration}-layer{layer}-cache{cache_index}"
                    first_changed = {"name": name, "layer": layer, "cache_index": cache_index,
                        "reference": save_tensor(directory, stem + "-reference.bin.gz", reference),
                        "current": save_tensor(directory, stem + "-current.bin.gz", current)}
                del current, reference
            state["snapshots"]._release_pages()
            return {"hidden": hidden, "active_kda_post": active,
                    "first_changed_active_cache": first_changed,
                    "raw_active_evidence_rule": "first differing numeric layer/cache per replay"}

        return observe
