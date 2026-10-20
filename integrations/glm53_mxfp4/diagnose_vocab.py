"""Qualify full GLM TP2 vocabulary shards through the installed vLLM classes."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import inspect
import json
import os
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
from diagnose_bf16_attention import write
from diagnose_bf16_kda import compare
from diagnose_bf16_overlap import capture
from safetensors import safe_open
from vocab_policy import load_targets


def gpu_digest(tensor: torch.Tensor) -> str:
    """Hash an entire reconstructed shard using bounded CPU staging."""
    digest = hashlib.sha256()
    for start in range(0, tensor.shape[0], 2048):
        digest.update(tensor[start:start + 2048].cpu().contiguous().view(torch.uint8).numpy())
    return digest.hexdigest()


def main() -> None:
    """Test exact bytes, real loading/masking, eager/graph/stream arithmetic and ownership."""
    from benchmark_bf16_head import _paired
    from mxwave_zipserv import runtime
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor import parameter
    from vllm.model_executor.layers import vocab_parallel_embedding as vocab
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )
    from vllm.model_executor.model_loader import utils
    from vllm.models.glm5next.nvidia.model import Glm5NextForConditionalGeneration

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--artifacts", required=True, type=Path)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("Preserve vocabulary qualification attempts")
    os.environ.update(MXWAVE_GLM53_ATTENTION_MAPPING="1", MXWAVE_GLM53_KDA_MODE="packed",
        MXWAVE_GLM53_KDA_SOURCE=str(args.source), MXWAVE_GLM53_KDA_FUSED_MAX_TOKENS="8",
        MXWAVE_GLM53_VOCAB_MODE="packed", MXWAVE_MOE_MODE="aligned",
        MXWAVE_GLM53_VOCAB_ARTIFACTS=str(args.artifacts),
        MXWAVE_GLM53_VOCAB_MANIFEST_SHA256=hashlib.sha256(
            (args.artifacts / "prepared.json").read_bytes()).hexdigest())
    import model_plugin

    model_plugin.install()
    import vocab_adapter
    import vocab_runtime

    import mxwave.lossless_bf16_rows as rows_codec
    import mxwave.lossless_bf16_rows_gpu as rows_gpu

    torch.set_num_threads(2)
    torch.manual_seed(757)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(
        1.5 * 1024**3 / torch.cuda.get_device_properties(device).total_memory, device)
    targets = load_targets(args.source)
    config = json.loads((args.source / "config.json").read_bytes())
    index = json.loads((args.source / "model.safetensors.index.json").read_bytes())["weight_map"]
    quant = CompressedTensorsConfig.from_config(copy.deepcopy(config["quantization_config"]))
    utils.configure_quant_config(quant, Glm5NextForConditionalGeneration)
    from kda_adapter import KdaLinearMethod
    from kda_policy import load_policy
    from vllm.model_executor.layers.linear import RowParallelLinear

    dummy = object.__new__(RowParallelLinear)
    torch.nn.Module.__init__(dummy)
    selected_kda = [prefix for prefix in load_policy(args.source)
                    if isinstance(quant.get_quant_method(dummy, prefix), KdaLinearMethod)]
    if len(selected_kda) != 68:
        raise RuntimeError("The combined vocabulary selector changed KDA coverage")
    report: dict[str, Any] = {"status": "running", "cases": [], "rejections": [],
        "manifest_sha256": os.environ["MXWAVE_GLM53_VOCAB_MANIFEST_SHA256"],
        "combined_kda_selected_prefixes": selected_kda,
        "gates": {"full_gpu_weight_sha256_matches_source": True,
                  "embedding_outputs_bits_exact": True, "head_maximum_relative_l2": .001,
                  "original_dense_parameters_released": True},
        "code_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("vocab_policy.py", "vocab_adapter.py", "vocab_runtime.py",
                         "model_plugin.py", "kda_policy.py", "kda_adapter.py", "kda_runtime.py",
                         "moe_runtime.py", "diagnose_vocab.py")},
        "core_sha256": {Path(module.__file__).name: hashlib.sha256(
            Path(module.__file__).read_bytes()).hexdigest() for module in (rows_codec, rows_gpu)},
        "vocab_source_sha256": hashlib.sha256(Path(inspect.getfile(vocab)).read_bytes()).hexdigest(),
        "limits": ["Both TP2 shards tested sequentially on Spark2; collectives are mocked here.",
                   "The actual stock forward, vocabulary mask and shard loader are exercised.",
                   "Random head activations; full-model decode quality remains a separate gate.",
                   "Simultaneous KDA/expert selection installed; no full model in this fixture."]}
    write(args.report, report)
    started = time.monotonic()
    stream = torch.cuda.Stream()
    try:
        for rank in (0, 1):
            for target in targets.values():
                with ExitStack() as stack:
                    stack.enter_context(set_current_vllm_config(VllmConfig(quant_config=quant)))
                    for module in (vocab, parameter):
                        stack.enter_context(patch.object(module, "get_tensor_model_parallel_rank",
                                                         return_value=rank))
                        stack.enter_context(patch.object(module,
                            "get_tensor_model_parallel_world_size", return_value=2))
                    stack.enter_context(patch.object(vocab, "tensor_model_parallel_all_reduce",
                                                     side_effect=lambda x: x))
                    cls = vocab.ParallelLMHead if target.kind == "head" else vocab.VocabParallelEmbedding
                    # Match the real GLM call: its embedding omits quant_config.
                    constructor_options = {"quant_config": quant} if target.kind == "head" else {}
                    with torch.device(device):
                        layer = cls(target.vocabulary, target.width, params_dtype=torch.bfloat16,
                                    prefix=target.prefix, **constructor_options)
                    if (not isinstance(layer.quant_method, vocab_adapter.VocabMethod)
                            or layer.weight.device.type != "cpu"):
                        raise TypeError("The actual constructor missed CPU vocabulary dispatch")
                    try:
                        layer.quant_method.process_weights_after_loading(layer)
                    except ValueError:
                        report["rejections"].append(f"{target.kind}-r{rank}-unloaded")
                    else:
                        raise RuntimeError("Unloaded vocabulary parameter was accepted")
                    with safe_open(args.source / index[target.source_key], framework="pt",
                                   device="cpu") as handle:
                        source = handle.get_tensor(target.source_key)
                        parameter_weight = layer.weight
                        if rank == 0:
                            try:
                                parameter_weight.weight_loader(parameter_weight, source[:1].float())
                            except ValueError:
                                report["rejections"].append(target.kind + "-source-cast")
                            else:
                                raise RuntimeError("Vocabulary loader accepted a cast source")
                        parameter_weight.weight_loader(parameter_weight, source)
                        try:
                            parameter_weight.weight_loader(parameter_weight, source)
                        except ValueError:
                            report["rejections"].append(f"{target.kind}-r{rank}-duplicate")
                        else:
                            raise RuntimeError("Vocabulary loader accepted duplicate input")
                        # Exercise the same pinned loading context as the whole model.
                        with utils.device_loading_context(layer, device):
                            layer.quant_method.process_weights_after_loading(layer)
                        del parameter_weight
                        gc.collect()
                        if (layer.weight is not None
                                or layer.mxwave_vocab_original_parameter() is not None):
                            raise RuntimeError("The original dense vocabulary parameter survived")
                        case: dict[str, Any] = {"rank": rank, "kind": target.kind,
                            "qualification": layer.mxwave_vocab_qualification, "measurements": [],
                            "constructor_quant_config_explicit": target.kind == "head"}
                        control = None
                        if target.kind == "head":
                            decoded = runtime.decompress(vocab_runtime.packed_head(layer))
                            digest = gpu_digest(decoded)
                            del decoded
                        else:
                            digest_builder = hashlib.sha256()
                            packed = vocab_runtime.packed_embedding(layer)
                            from mxwave.lossless_bf16_rows_gpu import gather_rows

                            for start in range(0, target.rows, 2048):
                                ids = torch.arange(start, min(start + 2048, target.rows), device=device)
                                out = gather_rows(packed, ids)
                                digest_builder.update(out.cpu().contiguous().view(torch.uint8).numpy())
                            digest = digest_builder.hexdigest()
                            del out, ids, packed
                        if digest != layer.mxwave_vocab_qualification["source_sha256"]:
                            raise RuntimeError("Complete vocabulary GPU reconstruction differs from source")
                        case["full_gpu_sha256"] = digest
                        dense = source[rank * target.rows:(rank + 1) * target.rows].to(device)
                        if target.kind == "head":
                            for tokens in (0, 1, 2, 3, 4, 7, 8, 16, 128):
                                x = torch.randn((tokens, target.width), device=device,
                                                dtype=torch.bfloat16)
                                control = lambda x=x, dense=dense: torch.nn.functional.linear(x, dense)
                                candidate = lambda x=x, layer=layer: layer.quant_method.apply(layer, x)
                                check = compare(candidate(), control(), exact=tokens == 0)
                                if tokens in (1, 4, 128):
                                    check["paired"] = _paired(control, candidate, trials=5, iterations=3)
                                if tokens in (1, 4):
                                    graph, output = capture(candidate, stream)
                                    graph.replay()
                                    torch.cuda.synchronize()
                                    check["graph"] = compare(output, control(), exact=False)
                                    del graph, output
                                case["measurements"].append({"tokens": tokens, **check})
                        else:
                            local_inputs = torch.tensor([0, 1, target.rows - 1, target.rows,
                                target.vocabulary - 1, -1, target.vocabulary], device=device)
                            for shape in ((7,), (1, 7), (7, 1)):
                                inputs = local_inputs.reshape(shape)
                                masked, invalid = vocab.get_masked_input_and_mask(inputs,
                                    rank * target.rows, (rank + 1) * target.rows, 0,
                                    target.vocabulary, target.vocabulary)
                                reference = torch.nn.functional.embedding(masked.long(), dense)
                                reference.masked_fill_(invalid.unsqueeze(-1), 0)
                                candidate = lambda inputs=inputs, layer=layer: layer(inputs)
                                check = compare(candidate(), reference, exact=True)
                                graph, output = capture(candidate, stream)
                                graph.replay()
                                torch.cuda.synchronize()
                                check["graph"] = compare(output, reference, exact=True)
                                with torch.cuda.stream(stream):
                                    alternate = candidate()
                                stream.synchronize()
                                check["non_default_stream"] = compare(alternate, reference, exact=True)
                                del graph, output, alternate
                                case["measurements"].append({"shape": list(shape), **check})
                            empty = layer(torch.empty(0, device=device, dtype=torch.int64))
                            if empty.shape != (0, target.width):
                                raise RuntimeError("Empty embedding changed its shape")
                        case["dense_parameter_removed"] = layer.weight is None
                        report["cases"].append(case)
                        write(args.report, report)
                        print(json.dumps({"rank": rank, "kind": target.kind,
                                          "status": "passed", "full_gpu_sha256": digest}), flush=True)
                        del dense, source, layer, control, candidate
                gc.collect()
                torch.cuda.empty_cache()
        report.update(status="passed", seconds=time.monotonic() - started,
                      cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      code_scope="All four full source vocabulary shards")
        write(args.report, report)
    except BaseException as exc:
        report.update(status="failed", error=repr(exc), seconds=time.monotonic() - started)
        write(args.report, report)
        raise


if __name__ == "__main__":
    main()
