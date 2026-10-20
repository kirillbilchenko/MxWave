"""Explicit opt-in selective BF16 KDA adapter for the pinned mixed compressed-tensors model."""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any

import torch
from kda_policy import Target, load_policy
from vllm.model_executor.layers import linear
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)

_INSTALLED = False
_TARGETS: dict[str, Target] = {}


class KdaLinearMethod(linear.UnquantizedLinearMethod):
    """Keep the stock TP shard loader and replace only qualified post-load BF16 storage."""

    def __init__(self, target: Target, max_words: int) -> None:
        self.target, self.max_words = target, max_words

    def create_weights(self, layer: torch.nn.Module, input_size_per_partition: int,
                       output_partition_sizes: list[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype, **attrs: Any) -> None:
        """Reject casts and incomplete fused-source loading while delegating TP slicing."""
        if (params_dtype != torch.bfloat16 or layer.has_bias
                or (sum(output_partition_sizes), input_size_per_partition)
                != (self.target.rows, self.target.width)):
            raise ValueError("Selective KDA requires the audited bias-free BF16 TP2 shape")
        loader = attrs.pop("weight_loader")
        layer.mxwave_kda_loaded_shards = set()

        def load(param: torch.nn.Parameter, loaded: torch.Tensor,
                 *args: Any, **kwargs: Any) -> None:
            if loaded.dtype != torch.bfloat16 or loaded.ndim != 2:
                raise ValueError("KDA source weights must be two-dimensional BF16 without casts")
            shard = args[0] if args else kwargs.get("loaded_shard_id")
            if shard not in self.target.expected_shards or shard in layer.mxwave_kda_loaded_shards:
                raise ValueError("Unknown or duplicate KDA source shard")
            loader(param, loaded, *args, **kwargs)
            layer.mxwave_kda_loaded_shards.add(shard)

        super().create_weights(layer, input_size_per_partition, output_partition_sizes,
                               input_size, output_size, params_dtype, weight_loader=load, **attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Verify lossless GPU reconstruction before removing the dense parameter."""
        import kda_runtime

        kda_runtime.prepare(layer, self.target, self.max_words)

    def apply(self, layer: torch.nn.Module, x: torch.Tensor,
              bias: torch.Tensor | None = None) -> torch.Tensor:
        """Preserve vLLM's tensor/tuple and TP behavior outside the quantization method."""
        import kda_runtime

        if bias is not None:
            raise ValueError("The qualified KDA projections do not have bias")
        return kda_runtime.apply(layer, x)


def install(source: Path | None = None) -> None:
    """Wrap exactly 68 configuration-derived BF16 projections, leaving all others delegated."""
    global _INSTALLED, _TARGETS
    if _INSTALLED:
        return
    if os.environ.get("MXWAVE_GLM53_ATTENTION_MAPPING") != "1":
        raise ValueError("Selective KDA storage requires the audited fused-attention mapping")
    _TARGETS = load_policy(source or Path(os.environ.get("MXWAVE_GLM53_KDA_SOURCE", "/model")))
    max_words = max(t.native_rows * t.width for t in _TARGETS.values())
    original = CompressedTensorsConfig.get_quant_method

    @functools.wraps(original)
    def select(config: Any, layer: torch.nn.Module, prefix: str) -> Any:
        method = original(config, layer, prefix)
        target = _TARGETS.get(prefix)
        if target is None:
            return method
        if type(method) is not linear.UnquantizedLinearMethod:
            raise ValueError("Selected KDA path is quantized or transformed; refuse replacement")
        return KdaLinearMethod(target, max_words)

    if KdaLinearMethod.__name__ not in linear.WEIGHT_LOADER_V2_SUPPORTED:
        linear.register_weight_loader_v2_supported_method(KdaLinearMethod)
    CompressedTensorsConfig.get_quant_method = select
    _INSTALLED = True
