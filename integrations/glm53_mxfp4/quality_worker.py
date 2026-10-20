"""Source proofs and bounded, eager-only fingerprints for quality diagnostics."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any

import model_plugin
import torch
from palette_model_worker import PaletteModelWorker


def install() -> None:
    """Install the existing adapters and source-hashing hook only in disposable engines."""
    if os.environ.get("MXWAVE_MTP_PILOT") == "1":
        from mtp_pilot_support import prepare

        prepare()
    model_plugin.install()
    if os.environ.get("MXWAVE_STABLE_MOE_DISPATCH") == "1":
        from stable_dispatch import install as stable_install

        stable_install()


def runtime_metadata(graphs_captured: bool) -> dict[str, Any]:
    """Return plain RPC primitives; TorchVersion is a str subclass rejected by msgspec."""
    return {
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda) if torch.version.cuda is not None else None,
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "graphs_captured": bool(graphs_captured),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


class QualityWorker(PaletteModelWorker):
    """Preserve source checks and expose explicitly intrusive repeat diagnostics."""

    def mxwave_quality_source_report(self) -> dict[str, Any]:
        """Hash all 84 actual scale groups on either the stock or compressed arm."""
        from vllm.distributed import get_tensor_model_parallel_rank

        candidate = os.environ.get("MXWAVE_PALETTE_MODEL") == "1"
        if candidate:
            result = self.mxwave_palette_report()
            native = model_plugin.native(1)
            path = Path(os.environ["MXWAVE_MOE_ALIGNED_LIBRARY"])
            if not native.shared_stage or native.encoding != "sparse_palette":
                raise ValueError("Quality evaluation requires the best U8 shared-stage consumer")
            result["native_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        else:
            groups, pointers, layers = [], set(), set()
            for name, module in self.model_runner.model.named_modules():
                match = re.search(
                    r"layers\.(\d+)\.mlp\.experts$", getattr(module, "layer_name", name)
                )
                if match is None or not hasattr(module, "w13_weight"):
                    continue
                index = int(match[1])
                if not 3 <= index <= 44 or module.w13_weight.data_ptr() in pointers:
                    continue
                pointers.add(module.w13_weight.data_ptr())
                layers.add(index)
                for family in ("w13_weight_scale", "w2_weight_scale"):
                    value = getattr(module, family)
                    groups.append(
                        {
                            "shape": list(value.shape),
                            "sha256": hashlib.sha256(
                                value.detach().contiguous().view(torch.uint8).cpu().numpy()
                            ).hexdigest(),
                        }
                    )
            if layers != set(range(3, 45)) or len(groups) != 84:
                raise RuntimeError("Stock scale source coverage is incomplete")
            result = {
                "status": "passed",
                "candidate": False,
                "tp_rank": get_tensor_model_parallel_rank(),
                "expert_layers": 42,
                "scale_groups": 84,
                "source_scale_hashes": groups,
                "allocation": self.mxwave_scale_report(),
            }
        result["runtime"] = runtime_metadata(self.model_runner.cudagraph_manager._graphs_captured)
        return result

    def mxwave_quality_trace(self, enabled: bool) -> dict[str, Any]:
        """Fingerprint layer/attention/MLP outputs with graphs disabled; never time these calls."""
        from vllm.distributed import get_tensor_model_parallel_rank

        handles = getattr(self, "_mxwave_quality_handles", [])
        for handle in handles:
            handle.remove()
        self._mxwave_quality_handles = []
        if not enabled:
            return {
                "status": "complete",
                "rank": get_tensor_model_parallel_rank(),
                "events": getattr(self, "_mxwave_quality_events", []),
                "intrusive_cpu_sync": True,
            }
        if self.model_runner.cudagraph_manager._graphs_captured:
            raise ValueError("Fingerprints require eager dispatch")
        self._mxwave_quality_events: list[dict[str, Any]] = []

        def tensors(value: Any) -> list[torch.Tensor]:
            if isinstance(value, torch.Tensor):
                return [value]
            if isinstance(value, (list, tuple)):
                return [tensor for child in value for tensor in tensors(child)]
            return []

        def hook(name: str) -> Any:
            def record(module: Any, inputs: Any, output: Any) -> None:
                if len(self._mxwave_quality_events) >= 1200:
                    raise RuntimeError("Fingerprint event bound exceeded")
                values = [
                    {
                        "shape": list(t.shape),
                        "dtype": str(t.dtype),
                        "sha256": hashlib.sha256(
                            t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy()
                        ).hexdigest(),
                    }
                    for t in tensors(output)
                ]
                self._mxwave_quality_events.append({"module": name, "outputs": values})

            return record

        for name, module in self.model_runner.model.named_modules():
            if re.search(r"\.layers\.\d+(?:\.(?:mlp|self_attn|linear_attn))?$", name):
                self._mxwave_quality_handles.append(module.register_forward_hook(hook(name)))
        if not self._mxwave_quality_handles:
            raise RuntimeError("No decoder modules available for fingerprints")
        return {
            "status": "armed",
            "hooks": len(self._mxwave_quality_handles),
            "rank": get_tensor_model_parallel_rank(),
            "intrusive_cpu_sync": True,
        }
