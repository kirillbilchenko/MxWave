"""Preserve every existing source comparison when all four storage adapters are active."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from diagnostic_worker import DiagnosticWorker
from kda_worker import DecodedView, KdaWorker


class ScaleSlice:
    """Decode one expert comparison at a time without retaining expanded layer scales."""

    def __init__(self, state: dict[str, Any]) -> None:
        self.state = state

    def __getitem__(self, expert: int) -> torch.Tensor:
        import model_plugin

        decoded = model_plugin.native(1).decode(self.state["packed"])
        return decoded[expert].to("cpu").view(torch.float8_e8m0fnu)


class VocabSlice:
    """Expose only the source boundary rows requested by the existing head/embed checker."""

    def __init__(self, layer: torch.nn.Module) -> None:
        self.layer = layer
        self.shape = (layer.num_embeddings_per_partition, layer.embedding_dim)

    def __getitem__(self, rows: slice) -> torch.Tensor:
        import vocab_runtime
        from mxwave_zipserv import runtime

        if self.layer.mxwave_vocab_kind == "head":
            decoded = runtime.decompress(vocab_runtime.packed_head(self.layer))
            return decoded[rows].to("cpu")
        from mxwave.lossless_bf16_rows_gpu import gather_rows

        ids = torch.arange(rows.start, rows.stop, device="cuda", dtype=torch.int64)
        return gather_rows(vocab_runtime.packed_embedding(self.layer), ids).cpu()


class SourceView:
    """Lazy compressed-value view so the original checker still verifies all 658 records."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    def named_modules(self) -> Any:
        """Retain references to packed storage while decoding each compared expert on demand."""
        import model_plugin

        for name, layer in self.model.named_modules():
            if not hasattr(layer, "w13_weight"):
                yield name, layer
                continue
            fields = {key: getattr(layer, key) for key in
                      ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale")}
            changed = False
            for key in ("w13_weight_scale", "w2_weight_scale"):
                state = model_plugin._STORAGE.get(id(fields[key]))
                if state and state["kind"] == 1:
                    fields[key] = ScaleSlice(state)
                    changed = True
            if changed:
                yield name, SimpleNamespace(layer_name=getattr(layer, "layer_name", name), **fields)
            else:
                yield name, layer

    def named_parameters(self) -> Any:
        """Keep original norm and mHC checks and restore the four vocabulary boundary checks."""
        yield from self.model.named_parameters()
        for name, layer in self.model.named_modules():
            if getattr(layer, "mxwave_vocab_metadata", None) is not None:
                yield name + ".weight", VocabSlice(layer)


class CombinedWorker(KdaWorker):
    """Use inherited graph controls and qualify source integrity for the combined stack."""

    def mxwave_checkpoint_check(self) -> dict[str, Any]:
        """Run unchanged base and attention checkers through bounded decoded views."""
        from pathlib import Path

        from attention_diagnostic import check_attention_loaded
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        model = self.model_runner.model
        proxy = SimpleNamespace(model_runner=SimpleNamespace(model=SourceView(model)))
        result = DiagnosticWorker.mxwave_checkpoint_check(proxy)
        try:
            attention = check_attention_loaded(DecodedView(model), Path("/model"),
                get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size())
        except Exception as exc:  # noqa: BLE001 -- report the gate failure without killing workers.
            attention = {"status": "failed", "error": repr(exc)}
        result.update(attention=attention, variant="combined")
        if attention["status"] != "passed" or len(result["records"]) != 658:
            result["status"] = "failed"
        self._mxwave_checkpoint_report = result
        return result

    def mxwave_scale_report(self) -> dict[str, Any]:
        """Include vocabulary ownership alongside expert scales, KDA scratch and allocation."""
        import vocab_runtime

        result = super().mxwave_scale_report()
        result["vocabulary"] = vocab_runtime.report()
        return result
