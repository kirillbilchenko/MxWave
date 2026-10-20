"""Audit native MTP compressed-tensors selectors without a GPU or a model allocation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import struct
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def main() -> None:
    """Compare draft policy with actual source-policy resolution before and after metadata repair."""
    import torch
    from attention_mapping import install as install_base
    from mtp_mapping import register_mtp_mapping
    from vllm import platforms
    from vllm.platforms.cpu import CpuPlatform

    if platforms.current_platform.is_unspecified():
        platforms.current_platform = CpuPlatform()
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )
    from vllm.model_executor.model_loader.utils import configure_quant_config
    from vllm.models.glm5next.nvidia.model import Glm5NextForConditionalGeneration as Base
    from vllm.models.glm5next.nvidia.mtp import Glm5NextMTP as Draft

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--all-experts", action="store_true")
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("Preserve the first native MTP selector audit")
    config = json.loads((args.source / "config.json").read_bytes())
    text = config["text_config"]
    if text["num_hidden_layers"] != 45 or text["num_nextn_predict_layers"] != 1:
        raise ValueError("Require the audited single-MTP GLM checkpoint")
    install_base()
    base_mapping = copy.deepcopy(Base.packed_modules_mapping)
    base_mapper = Base.hf_to_vllm_mapper

    def quant(model: type[Any]) -> Any:
        value = CompressedTensorsConfig.from_config(copy.deepcopy(config["quantization_config"]))
        configure_quant_config(value, model)
        return value

    dummy = torch.nn.Linear(2, 2, bias=False, device="meta")

    def scheme(value: Any, name: str) -> str | None:
        selected = value.get_scheme_dict(dummy, name)
        return str(selected["format"]) if selected else None

    source_quant, before = quant(Base), quant(Draft)
    suffixes = ["self_attn.fused_qkv_a_proj", "self_attn.q_b_proj", "self_attn.kv_b_proj",
        "self_attn.o_proj", "self_attn.indexer.wk_weights_proj", "mlp.gate",
        "mlp.shared_experts.gate_up_proj", "mlp.shared_experts.down_proj"]
    experts = range(288) if args.all_experts else (0, 287)
    suffixes.extend(f"mlp.experts.{expert}.{projection}" for expert in experts
                    for projection in ("gate_proj", "up_proj", "down_proj"))
    records = []
    for suffix in suffixes:
        source_name = f"model.language_model.layers.45.{suffix}"
        base_name = Base.hf_to_vllm_mapper._map_name(source_name)
        draft_name = "model.layers.45." + suffix
        records.append({"source_name": source_name, "draft_name": draft_name,
            "expected_from_source_policy": scheme(source_quant, base_name),
            "before": scheme(before, draft_name)})
    register_mtp_mapping(Base, Draft, layer_index=text["num_hidden_layers"])
    after = quant(Draft)
    unchanged = quant(Base)
    for row in records:
        row["after"] = scheme(after, row["draft_name"])
        assert scheme(unchanged, Base.hf_to_vllm_mapper._map_name(row["source_name"])) == row[
            "expected_from_source_policy"]
    assert Base.packed_modules_mapping == base_mapping and Base.hf_to_vllm_mapper is base_mapper
    args.report.with_suffix(".incomplete.json").write_text(json.dumps({
        "phase": "selectors_complete_before_vocabulary_and_headers", "records": records,
        "cuda_initialized": torch.cuda.is_initialized(), "base_policy_unchanged": True,
    }, indent=2, allow_nan=False) + "\n")
    vocab_checks = []
    sharing = None
    if args.all_experts:
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            ParallelLMHead,
            VocabParallelEmbedding,
        )
        from vllm.v1.spec_decode.eagle import EagleProposer

        for cls, prefix in ((ParallelLMHead, "model.layers.45.head"),
                            (VocabParallelEmbedding, "model.embed_tokens")):
            layer = object.__new__(cls)
            torch.nn.Module.__init__(layer)
            selected = after.get_quant_method(layer, prefix)
            if selected is not None:
                raise ValueError("Native MTP vocabulary must remain unquantized before sharing")
            vocab_checks.append({"prefix": prefix, "method": None})
        target = torch.nn.Module()
        target.model = torch.nn.Module()
        target.model.embed_tokens = torch.nn.Module()
        target.lm_head = torch.nn.Module()
        draft = torch.nn.Module()
        draft.model = torch.nn.Module()
        draft.model.embed_tokens = torch.nn.Module()
        layer = torch.nn.Module()
        layer.shared_head = torch.nn.Module()
        layer.shared_head.head = torch.nn.Module()
        draft.model.layers = torch.nn.ModuleDict({"45": layer})
        proposer = object.__new__(EagleProposer)
        proposer.model = draft
        proposer.vllm_config = SimpleNamespace(speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=SimpleNamespace(
                index_share_for_mtp_iteration=True))))
        proposer.use_local_argmax_reduction = False
        native_module = inspect.getmodule(EagleProposer._maybe_share_embeddings)
        assert native_module is not None
        original_pp = native_module.get_pp_group
        try:
            native_module.get_pp_group = lambda: SimpleNamespace(world_size=1)
            proposer._maybe_share_embeddings(target)
            proposer._maybe_share_lm_head(target)
        finally:
            native_module.get_pp_group = original_pp
        assert proposer.model.model.embed_tokens is target.model.embed_tokens
        assert proposer.model.lm_head is target.lm_head
        assert layer.shared_head.head is target.lm_head
        assert proposer._share_mtp_indices
        assert not hasattr(target.lm_head, "weight") and not hasattr(target.model.embed_tokens, "weight")
        sharing = {"native_whole_module_sharing_passed": True,
            "target_embedding_has_no_dense_weight": True, "target_head_has_no_dense_weight": True,
            "shared_mtp_head_is_target_head": True, "native_mtp_index_reuse_selected": True,
            "local_argmax_execution_tested": False}
    assert not torch.cuda.is_initialized()
    passed = all(row["after"] == row["expected_from_source_policy"] for row in records)
    header_proofs = []
    if args.all_experts:
        prefix = "model.language_model.layers.45."
        expected = {prefix + f"mlp.experts.{expert}.{projection}" for expert in range(288)
                    for projection in ("gate_proj", "up_proj", "down_proj")}
        targets = {name for group in config["quantization_config"]["config_groups"].values()
                   if group["weights"]["num_bits"] == 4 for name in group["targets"]
                   if name.startswith(prefix)}
        if targets != expected:
            raise ValueError("MTP MXFP4 target coverage differs from 288 experts/all projections")
        weight_map = json.loads((args.source / "model.safetensors.index.json").read_bytes())["weight_map"]
        quantized_names = {name + suffix for name in expected
                           for suffix in (".weight_packed", ".weight_scale")}
        shared = {prefix + f"mlp.shared_experts.{projection}" for projection in
                  ("gate_proj", "up_proj", "down_proj")}
        quantized_names.update(name + suffix for name in shared
                               for suffix in (".weight", ".weight_scale"))
        names = {name for name in weight_map if name.startswith(prefix)}
        if len(names) != 1756 or not quantized_names.issubset(names):
            raise ValueError("Native MTP source tensor coverage differs from the audited checkpoint")
        headers = {}
        for shard in sorted({weight_map[name] for name in names}):
            with (args.source / shard).open("rb") as handle:
                length = struct.unpack("<Q", handle.read(8))[0]
                if not 0 < length <= 64 * 1024**2:
                    raise ValueError("Unbounded MTP safetensors header")
                raw = handle.read(length)
            headers.update({name: value for name, value in json.loads(raw).items() if name in names})
            header_proofs.append({"shard": shard, "sha256": hashlib.sha256(raw).hexdigest()})
        if set(headers) != names:
            raise ValueError("MTP packed weights or scales are missing from indexed headers")
        for name in quantized_names:
            spec = headers[name]
            if name.endswith(".weight"):
                assert spec["dtype"] == "F8_E4M3"
            else:
                assert spec["dtype"] == "U8"
        assert all(headers[name]["dtype"] in ("BF16", "F32")
                   for name in names - quantized_names)
        for name in expected:
            packed, scale = headers[name + ".weight_packed"], headers[name + ".weight_scale"]
            rows, packed_width = packed["shape"]
            assert scale["shape"] == [rows, packed_width // 16]
        for name in shared:
            rows, width = headers[name + ".weight"]["shape"]
            assert headers[name + ".weight_scale"]["shape"] == [rows, width // 32]
    files = {"audit": Path(__file__), "mapping": Path(__file__).with_name("mtp_mapping.py"),
        "base_aliases": Path(__file__).with_name("attention_mapping.py"),
        "base_model": Path(inspect.getfile(Base)), "draft_model": Path(inspect.getfile(Draft)),
        "selector": Path(inspect.getfile(CompressedTensorsConfig)),
        "configure": Path(inspect.getfile(configure_quant_config))}
    result = {"status": "passed" if passed else "failed", "passed": passed,
        "cuda_initialized": False, "model_allocated": False, "base_policy_unchanged": True,
        "source_sha256": {name: hashlib.sha256((args.source / name).read_bytes()).hexdigest()
            for name in ("config.json", "model.safetensors.index.json")},
        "code_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                        for name, path in files.items()},
        "mismatches_before": sum(row["before"] != row["expected_from_source_policy"] for row in records),
        "mismatches_after": sum(row["after"] != row["expected_from_source_policy"] for row in records),
        "all_expert_selectors": args.all_experts,
        "expert_projections_checked": len(expected) if args.all_experts else 6,
        "source_tensor_headers_checked": len(names) if args.all_experts else 0,
        "source_header_proofs": header_proofs,
        "vocabulary_checks": vocab_checks, "vocabulary_sharing": sharing,
        "records": records,
        "limits": ["Native selector metadata only; no MTP weights loaded, no inference or speed result.",
                   "Draft experts must stay on stock methods for the isolated MTP pilot."]}
    args.report.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}))
    if not passed:
        raise RuntimeError("Native MTP selector policy differs from checkpoint source policy")


if __name__ == "__main__":
    main()
