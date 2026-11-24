"""Exercise official TP embedding dispatch with no dense parameter retained."""

from __future__ import annotations

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch


def check(device: str = "cpu") -> dict:
    """Check both shard boundaries, multidimensional IDs and the native mask path."""
    import torch
    from vllm.model_executor.layers import vocab_parallel_embedding as upstream

    from mxwave_glm53._runtime import vocab_runtime
    from mxwave_glm53._runtime.vocab_adapter import VocabMethod
    from mxwave_glm53._runtime.vocab_policy import VocabTarget

    values = (torch.arange(32 * 64).reshape(32, 64).float() / 128).bfloat16()
    ids = torch.tensor([[0, 15, 16, 31], [31, 16, 15, 0]], device=device)
    outputs = []
    with ExitStack() as scope:
        # Exercise the CUDA branch selection on CPU without invoking a CUDA op.
        scope.enter_context(patch.object(upstream, "current_platform",
                                         SimpleNamespace(is_cuda=lambda: True)))
        if device == "cpu":
            mask = upstream.get_masked_input_and_mask
            scope.enter_context(patch.object(upstream, "get_masked_input_and_mask",
                                             mask._torchdynamo_orig_callable))
        for rank in (0, 1):
            group = SimpleNamespace(world_size=2, rank_in_group=rank)
            target = VocabTarget("embedding", "fixture", "fixture.weight",
                                 rows=16, width=64, vocabulary=32)
            layer = upstream.VocabParallelEmbedding(
                32, 64, params_dtype=torch.bfloat16, padding_size=1,
                quant_method=VocabMethod(target), parallel_group=group, reduce_results=False)
            assert not layer.use_fused_embedding
            layer.weight.weight_loader(layer.weight, values)
            local = layer.weight.detach().clone()
            layer.register_parameter("weight", None)
            if device == "cuda":
                from mxwave.lossless_bf16_rows import pack_rows
                from mxwave.lossless_bf16_rows_gpu import upload_rows

                packed = upload_rows(pack_rows(local.view(torch.uint16).numpy(), block_size=32),
                                     device=torch.device(device))
                for name, buffer in zip(vocab_runtime.ROW_BUFFERS, packed.tensors(), strict=True):
                    layer.register_buffer("mxwave_vocab_" + name, buffer)
                layer.mxwave_vocab_metadata = (16, 64, 32)
                actual = layer(ids)
            else:
                for name in vocab_runtime.ROW_BUFFERS:
                    layer.register_buffer("mxwave_vocab_" + name, torch.empty(0))
                layer.mxwave_vocab_metadata = (16, 64, 32)
                with patch.object(vocab_runtime, "embedding_rows", side_effect=(
                    lambda indices, buffers, *metadata, table=local:
                    torch.nn.functional.embedding(indices, table)
                )) as decoder:
                    actual = layer(ids)
                    assert decoder.call_count == 1
            expected = values.to(device)[ids].clone()
            expected[(ids < rank * 16) | (ids >= (rank + 1) * 16)] = 0
            assert torch.equal(actual, expected)
            assert layer.weight is None
            outputs.append(actual)
        assert torch.equal(outputs[0] + outputs[1], values.to(device)[ids])
        dense = upstream.VocabParallelEmbedding(
            32, 64, params_dtype=torch.bfloat16, padding_size=1,
            parallel_group=group, reduce_results=False)
        assert dense.use_fused_embedding, "Unowned dense embeddings keep the native fast path"
    return {"status": "passed", "device": device, "tp_shards": 2,
            "dense_parameter_absent": True, "shard_mask_bits_exact": True,
            "dense_dispatch_unchanged": True}
