"""Bounded, read-only activation evidence for a private stock-vLLM diagnostic."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any


def tensor_record(value: Any, *, full: bool, tail_path: Path | None = None) -> dict[str, Any]:
    """Hash exact tensor bytes in bounded CPU chunks and retain the last row separately."""
    import torch

    tensor = value.detach()
    digest = hashlib.sha256()
    if full:
        if tensor.ndim == 0:
            chunks = [tensor.reshape(1)]
        else:
            chunks = tensor.split(64, dim=0)
        for chunk in chunks:
            raw = chunk.contiguous().view(torch.uint8).cpu().numpy().tobytes()
            digest.update(raw)
    tail = tensor[-1:] if tensor.ndim else tensor.reshape(1)
    tail_bytes = tail.contiguous().view(torch.uint8).cpu().numpy().tobytes()
    result = {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
              "full_sha256": digest.hexdigest() if full else None,
              "tail_sha256": hashlib.sha256(tail_bytes).hexdigest()}
    if tail_path is not None:
        tail_path.write_bytes(tail_bytes)
        result.update(tail_file=tail_path.name, tail_shape=list(tail.shape),
                      tail_bytes=len(tail_bytes))
    return result


def tensors(value: Any, path: str = "") -> list[tuple[str, Any]]:
    """Enumerate tensor arguments without interpreting or modifying model state."""
    import torch

    if isinstance(value, torch.Tensor):
        return [(path, value)]
    if isinstance(value, (tuple, list)):
        return [item for i, child in enumerate(value) for item in tensors(child, f"{path}/{i}")]
    if isinstance(value, dict):
        return [item for key, child in value.items() for item in tensors(child, f"{path}/{key}")]
    return []


class NumericalWorker:
    """Install temporary observation hooks only after model initialization and graph capture."""

    def mxwave_numerical_trace(self, label: str | None = None) -> dict[str, Any]:
        """Arm one request, or disarm and persist its exact activation fingerprints."""
        import torch
        from vllm.distributed import get_tensor_model_parallel_rank

        from mxwave_glm53.worker import snapshot

        rank = get_tensor_model_parallel_rank()
        if label is None:
            for handle in getattr(self, "_numerical_handles", []):
                handle.remove()
            self._numerical_handles = []
            result = {"rank": rank, "label": getattr(self, "_numerical_label", None),
                      "events": getattr(self, "_numerical_events", []),
                      "memory": snapshot(self)}
            if result["label"] is not None:
                target = Path(os.environ["MXWAVE_GLM53_REPORTS"]) / (
                    f"numerical-{result['label']}-rank{rank}.json")
                target.write_text(json.dumps(result, indent=2) + "\n")
            return result
        if not re.fullmatch(r"[a-z0-9-]+", label) or getattr(self, "_numerical_handles", []):
            raise ValueError("Require a fresh simple label and disarmed hooks")
        if self.vllm_config.speculative_config is not None:
            raise ValueError("This fixed-input diagnostic requires speculation off")
        self._numerical_label = label
        self._numerical_events = []
        self._numerical_handles = []
        deep = os.environ.get("MXWAVE_DIAGNOSTIC_DEEP_MLA") == "1"
        reports = Path(os.environ["MXWAVE_GLM53_REPORTS"])
        active_rows = 0
        index_buffers = {
            module.topk_indices_buffer.data_ptr()
            for name, module in self.model_runner.model.named_modules()
            if name.startswith("language_model.model.layers.3.self_attn")
            and isinstance(getattr(module, "topk_indices_buffer", None), torch.Tensor)
        }

        def record(name: str, phase: str, value: Any, full: bool) -> None:
            nonlocal active_rows
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Numerical observation must not run inside graph capture")
            if len(self._numerical_events) >= 2000:
                raise RuntimeError("Bounded numerical trace exhausted")
            leaves = tensors(value)
            if name == "language_model.model.layers.3.self_attn" and phase == "input":
                positions = [t for _, t in leaves if t.ndim == 1 and not t.is_floating_point()]
                if len(positions) != 1:
                    raise RuntimeError("Cannot identify layer-three positions for active-row masking")
                active_rows = positions[0].numel()
            fields = {}
            for ordinal, (key, tensor) in enumerate(leaves):
                original_shape = list(tensor.shape)
                if tensor.data_ptr() in index_buffers:
                    if not 0 < active_rows <= tensor.shape[0]:
                        raise RuntimeError("Unknown active index-buffer rows")
                    tensor = tensor[:active_rows]
                tail_path = None
                if deep and name.startswith("language_model.model.layers.3.self_attn"):
                    tail_path = reports / (
                        f"tail-{label}-rank{rank}-{len(self._numerical_events)}-{ordinal}.bin")
                fields[key] = tensor_record(tensor, full=full, tail_path=tail_path)
                if original_shape != list(tensor.shape):
                    fields[key]["original_shape"] = original_shape
                    fields[key]["only_active_index_rows"] = True
            self._numerical_events.append({
                "module": name, "phase": phase, "tensors": fields,
            })

        selected = []
        for name, module in self.model_runner.model.named_modules():
            match = re.fullmatch(r"language_model\.model\.layers\.(\d+)(?:\.(.*))?", name)
            if match is None:
                continue
            index, suffix = int(match[1]), match[2]
            deep_attention = deep and index == 3 and suffix is not None and (
                suffix == "self_attn" or suffix.startswith("self_attn."))
            if not deep_attention and suffix is not None and (index >= 6 or suffix not in {
                "self_attn", "mlp", "input_layernorm", "post_attention_layernorm",
                "mhc_pre_op", "mhc_post_op", "mhc_fused_post_pre_op",
            }):
                continue
            full = index < 6
            selected.append({"name": name, "class": type(module).__module__ + "."
                             + type(module).__name__,
                             "impl": type(module.impl).__module__ + "."
                             + type(module.impl).__name__ if hasattr(module, "impl") else None})

            def before(mod: Any, args: Any, kwargs: Any, *, n: str = name,
                       f: bool = full) -> None:
                record(n, "input", (args, kwargs), f)

            def after(mod: Any, args: Any, kwargs: Any, output: Any, *, n: str = name,
                      f: bool = full) -> None:
                record(n, "output", output, f)

            self._numerical_handles += [
                module.register_forward_pre_hook(before, with_kwargs=True),
                module.register_forward_hook(after, with_kwargs=True),
            ]
        if not selected:
            raise RuntimeError("No audited GLM decoder modules found")
        return {"rank": rank, "label": label, "selected_modules": selected,
                "full_hash_layers": list(range(6)), "later_layers": "last-row hashes only",
                "speculation": False, "performance_measurement": False}
