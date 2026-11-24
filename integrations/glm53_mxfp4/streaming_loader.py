"""Opt-in vLLM loader retaining the stock model's TP slicing and weight preparation."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from checkpoint_stream import checked_shards, iter_checkpoint
from download_full_model import REPOSITORY, REVISION
from vllm.config import ModelConfig, VllmConfig
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader


@register_model_loader("mxwave-stream")
class MxWaveStreamLoader(DefaultModelLoader):
    """Stream the exact audited HF tensors through the unchanged model weight loader."""

    def download_model(self, model_config: ModelConfig) -> None:
        """Require pre-staged metadata and a hash-bound inventory, without downloading again."""
        self._inventory(model_config)

    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str = "",
    ) -> torch.nn.Module:
        """Save actual retained weight allocation before KV-cache and graph allocations."""
        model = super().load_model(vllm_config, model_config, prefix)
        import model_plugin

        state = model_plugin.report()
        state.update(phase="weights_loaded", mode=os.environ.get("MXWAVE_MOE_MODE", "stock"))
        target = Path(os.environ["MXWAVE_LOAD_REPORT"]).with_name("weights-memory.json")
        target.write_text(json.dumps(state, indent=2) + "\n")
        return model

    def _inventory(self, model_config: ModelConfig) -> dict[str, Any]:
        inventory = json.loads(Path(os.environ["MXWAVE_CHECKPOINT_INVENTORY"]).read_bytes())
        if inventory["repository"] != REPOSITORY or inventory["revision"] != REVISION:
            raise ValueError("Streaming loader requires the audited source revision")
        checked_shards(inventory)
        directory = Path(model_config.model)
        for name, expected in {
            "config.json": "194d290ebb0f19aedc60d24ea9e236e33c86616e460f4a108a842f6fd5b8b198",
            "model.safetensors.index.json":
                "6d1364a27eef78675390422b1b5230601817fb7460ce816951b2cd515934c85d",
        }.items():
            if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Streaming metadata differs from source: {name}")
        return inventory

    def get_all_weights(
        self, model_config: ModelConfig, model: torch.nn.Module,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Replace checkpoint I/O only; model.load_weights still owns naming and TP slicing."""
        if getattr(model, "secondary_weights", ()):
            raise ValueError("This prepared loader does not support secondary checkpoints")
        inventory = self._inventory(model_config)
        self.counter_before_loading_weights = time.perf_counter()
        events: list[dict[str, Any]] = []
        path = Path(os.environ["MXWAVE_LOAD_REPORT"])

        def record(entry: dict[str, Any]) -> None:
            events.append(entry)
            temporary = path.with_suffix(".incomplete")
            temporary.write_text(json.dumps({"status": "loading", "shards": events}, indent=2)
                                 + "\n")
            temporary.replace(path)
            print(f"MxWave source verified and loaded: {len(events)}/{len(inventory['shards'])}",
                  flush=True)

        yield from iter_checkpoint(
            inventory, Path(model_config.model), Path(os.environ["MXWAVE_LOAD_SCRATCH"]),
            remote_base=os.environ.get("MXWAVE_CHECKPOINT_HTTP"),
            after_shard=lambda: torch.cuda.current_stream().synchronize(), progress=record,
        )
        path.write_text(json.dumps({"status": "complete", "shards": events,
                                    "tensors": sum(x["tensors"] for x in events)}, indent=2) + "\n")
