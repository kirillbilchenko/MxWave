"""Selective TP2 vocabulary methods composed with the existing KDA and expert adapters."""

from __future__ import annotations

import functools
import inspect
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)
from vocab_policy import VocabTarget, load_targets, validate_layer

_INSTALLED = False


class VocabMethod(UnquantizedEmbeddingMethod):
    """Retain stock row loading and the caller's TP mask/reduce/gather behavior."""

    requires_device_loading = False

    def __init__(self, target: VocabTarget) -> None:
        self.target = target

    def create_weights(self, layer: torch.nn.Module, input_size_per_partition: int,
                       output_partition_sizes: list[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype, **attrs: Any) -> None:
        """Create a CPU-only parameter; reject casts, duplicate loads and changed dimensions."""
        if (params_dtype != torch.bfloat16
                or (input_size_per_partition, input_size, output_size)
                != (self.target.width, self.target.width, self.target.vocabulary)
                or output_partition_sizes != [self.target.rows]):
            raise ValueError("Vocabulary adapter requires the audited BF16 TP2 dimensions")
        validate_layer(layer, self.target)
        loader = attrs.pop("weight_loader")
        layer.mxwave_vocab_source_loaded = False

        def load(param: torch.nn.Parameter, source: torch.Tensor) -> None:
            if (source.dtype != torch.bfloat16
                    or tuple(source.shape) != (self.target.vocabulary, self.target.width)
                    or layer.mxwave_vocab_source_loaded):
                raise ValueError("Vocabulary source must be one complete original BF16 matrix")
            loader(param, source)
            layer.mxwave_vocab_source_loaded = True

        with torch.device("cpu"):
            super().create_weights(layer, input_size_per_partition, output_partition_sizes,
                                   input_size, output_size, params_dtype,
                                   weight_loader=load, **attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Replace verified loaded CPU storage with its hash-bound GPU representation."""
        import vocab_runtime

        vocab_runtime.prepare(layer, self.target)

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        """Decode local rows; stock forward handles the TP2 mask and collective reduction."""
        import vocab_runtime

        if self.target.kind != "embedding":
            raise ValueError("The untied head cannot be reused as an embedding")
        buffers = [getattr(layer, "mxwave_vocab_" + n) for n in vocab_runtime.ROW_BUFFERS]
        return vocab_runtime.embedding_rows(input_, buffers, *layer.mxwave_vocab_metadata)

    def apply(self, layer: torch.nn.Module, x: torch.Tensor,
              bias: torch.Tensor | None = None) -> torch.Tensor:
        """Project the local vocabulary shard; stock logits processing gathers it."""
        import vocab_runtime

        if self.target.kind != "head" or bias is not None:
            raise ValueError("Only the audited bias-free untied head supports linear execution")
        return vocab_runtime.apply_head(layer, x)

    def tie_weights(self, layer: torch.nn.Module, embed_tokens: torch.nn.Module) -> torch.nn.Module:
        """Reject unsupported tying instead of retaining a duplicate dense parameter."""
        raise ValueError("The GLM vocabulary adapter requires untied weights")


def install() -> None:
    """Intercept exactly two BF16 prefixes, preserving other selectors and loading contexts."""
    global _INSTALLED
    if _INSTALLED:
        return
    if os.environ.get("MXWAVE_GLM53_ATTENTION_MAPPING") != "1":
        raise ValueError("The combined vocabulary adapter requires corrected attention mapping")
    targets = load_targets(Path(os.environ.get("MXWAVE_GLM53_KDA_SOURCE", "/model")))
    original = CompressedTensorsConfig.get_quant_method

    @functools.wraps(original)
    def select(config: Any, layer: torch.nn.Module, prefix: str) -> Any:
        method = original(config, layer, prefix)
        target = targets.get(prefix)
        if target is None:
            return method
        expected = ParallelLMHead if target.kind == "head" else VocabParallelEmbedding
        if (type(layer) is not expected
                or method is not None and type(method) is not UnquantizedEmbeddingMethod):
            raise ValueError("Vocabulary selection requires an ordinary BF16 GLM vocab method")
        return VocabMethod(target)

    # This pinned development loader predates requires_device_loading. Keep
    # its context unchanged for every other module, including parents.
    from vllm.model_executor.model_loader import utils

    original_context = utils.device_loading_context

    @contextmanager
    def loading_context(layer: torch.nn.Module, device: torch.device) -> Any:
        if isinstance(getattr(layer, "quant_method", None), VocabMethod):
            yield layer
        else:
            with original_context(layer, device) as value:
                yield value

    utils.device_loading_context = loading_context
    CompressedTensorsConfig.get_quant_method = select

    # The pinned GLM constructor omits quant_config for its text embedding.
    # Supply the existing engine config only at this exact audited prefix;
    # other embeddings and the explicit-config head retain their stock call.
    original_constructor = VocabParallelEmbedding.__init__
    constructor_signature = inspect.signature(original_constructor)

    @functools.wraps(original_constructor)
    def construct(layer: torch.nn.Module, *args: Any, **kwargs: Any) -> None:
        call = constructor_signature.bind(layer, *args, **kwargs)
        call.apply_defaults()
        target = targets.get(call.arguments["prefix"])
        if (type(layer) is VocabParallelEmbedding and target is not None
                and target.kind == "embedding" and call.arguments["quant_config"] is None):
            from vllm.config import get_current_vllm_config

            quant = get_current_vllm_config().quant_config
            if not isinstance(quant, CompressedTensorsConfig):
                raise ValueError("The audited GLM embedding requires its engine CT config")
            call.arguments["quant_config"] = quant
            original_constructor(*call.args, **call.kwargs)
        else:
            original_constructor(layer, *args, **kwargs)

    VocabParallelEmbedding.__init__ = construct
    _INSTALLED = True
