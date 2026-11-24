"""Reproduce new-runner MTP config reuse and test isolation through its actual sharing helper."""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def main() -> None:
    """Run native loading orchestration with a meta-only model factory, without weights or CUDA."""
    import torch
    from mtp_draft_loader import install, report
    from mtp_mapping import install as install_mapping
    from vllm import platforms
    from vllm.platforms.cpu import CpuPlatform

    platforms.current_platform = CpuPlatform()
    import vllm.models.glm5next.nvidia.mtp as native_mtp
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )
    from vllm.model_executor.model_loader.utils import configure_quant_config
    from vllm.models.glm5next.nvidia.model import Glm5NextForConditionalGeneration as Base
    from vllm.models.glm5next.nvidia.mtp import Glm5NextMTP as Draft
    from vllm.v1.worker.gpu.spec_decode.eagle import utils

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("Preserve first actual new-runner MTP loader evidence")
    source = json.loads((args.source / "config.json").read_bytes())
    install_mapping()
    quant = CompressedTensorsConfig.from_config(copy.deepcopy(source["quantization_config"]))
    configure_quant_config(quant, Base)
    dummy = torch.nn.Linear(2, 2, bias=False, device="meta")
    suffixes = ["self_attn.fused_qkv_a_proj", "self_attn.q_b_proj", "self_attn.kv_b_proj",
        "self_attn.o_proj", "self_attn.indexer.wk_weights_proj", "mlp.gate",
        "mlp.shared_experts.gate_up_proj", "mlp.shared_experts.down_proj"]
    suffixes.extend(f"mlp.experts.{expert}.{projection}" for expert in range(288)
                    for projection in ("gate_proj", "up_proj", "down_proj"))

    def scheme(config: Any, name: str) -> str | None:
        value = config.get_scheme_dict(dummy, name)
        return str(value["format"]) if value else None

    expected = [scheme(quant, Base.hf_to_vllm_mapper._map_name(
        "model.language_model.layers.45." + suffix)) for suffix in suffixes]
    observed: dict[str, list[str | None]] = {}
    current = "inherited_target_config"
    created = []

    def create(*, vllm_config: Any, model_config: Any, **kwargs: Any) -> Any:
        configure_quant_config(vllm_config.quant_config, Draft)
        observed[current] = [scheme(vllm_config.quant_config, "model.layers.45." + suffix)
                             for suffix in suffixes]
        draft = torch.nn.Module()
        draft.model = torch.nn.Module()
        draft.model.embed_tokens = torch.nn.Module()
        draft.model.topk_indices_buffer = object()
        layer = torch.nn.Module()
        layer.shared_head = torch.nn.Module()
        layer.shared_head.head = torch.nn.Module()
        layer.attention = torch.nn.Module()
        layer.attention.topk_indices_buffer = object()
        draft.model.layers = torch.nn.ModuleDict({"45": layer})
        created.append(draft)
        return draft

    target = torch.nn.Module()
    target.language_model = torch.nn.Module()
    target.language_model.model = torch.nn.Module()
    target.language_model.model.embed_tokens = torch.nn.Module()
    target.language_model.model.topk_indices_buffer = object()
    target.language_model.lm_head = torch.nn.Module()
    target.get_language_model = lambda: target.language_model
    draft_config = SimpleNamespace(hf_config=SimpleNamespace(architectures=["Glm5NextMTPModel"]))
    spec = SimpleNamespace(method="mtp", num_speculative_tokens=2,
                           draft_model_config=draft_config, kv_cache_dtype=None)
    original_factory, original_pp = utils.get_model, utils.get_pp_group
    original_weights = Draft.load_weights
    original_expert_mapping = native_mtp.fused_moe_make_expert_params_mapping
    from safetensors import safe_open

    prefix = "model.language_model.layers.45.self_attn."
    index = json.loads((args.source / "model.safetensors.index.json").read_bytes())["weight_map"]
    key = prefix + "kv_a_proj_with_mqa.weight"
    with safe_open(args.source / index[key], framework="pt", device="cpu") as handle:
        kv_source = handle.get_tensor(key)
    fake = object.__new__(Draft)
    torch.nn.Module.__init__(fake)
    fake.config = SimpleNamespace(num_hidden_layers=45, num_nextn_predict_layers=1,
        mla_nope=True, qk_rope_head_dim=64, kv_lora_rank=512, q_lora_rank=1536,
        hidden_size=4096, n_routed_experts=288)
    fake.model = torch.nn.Module()
    fake.model.mtp_start_layer_idx = 45
    fake.model.num_mtp_layers = 1
    layer = torch.nn.Module()
    layer.mtp_block = torch.nn.Module()
    layer.mtp_block.self_attn = torch.nn.Module()
    fused = torch.nn.Module()
    fused.register_parameter("weight", torch.nn.Parameter(
        torch.zeros((2112, 4096), dtype=torch.bfloat16), requires_grad=False))

    def projection_load(param: Any, value: Any, shard: int) -> None:
        if shard != 1 or list(value.shape) != [576, 4096]:
            raise ValueError("BF16 NoPE KV source did not satisfy native fused projection shape")
        param.data[1536:].copy_(value)

    fused.weight.weight_loader = projection_load
    layer.mtp_block.self_attn.fused_qkv_a_proj = fused
    fake.model.layers = torch.nn.ModuleDict({"45": layer})
    inherited = SimpleNamespace(quant_config=copy.deepcopy(quant), speculative_config=spec)
    native_before = hashlib.sha256(pickle.dumps(inherited.quant_config, protocol=5)).hexdigest()
    target_before = hashlib.sha256(pickle.dumps(quant, protocol=5)).hexdigest()
    try:
        utils.get_model = create
        utils.get_pp_group = lambda: SimpleNamespace(world_size=1)
        native_mtp.fused_moe_make_expert_params_mapping = lambda *args, **kwargs: []
        utils.load_eagle_model(target, inherited)
        inherited_changed = hashlib.sha256(pickle.dumps(
            inherited.quant_config, protocol=5)).hexdigest() != native_before
        try:
            original_weights(fake, iter([(key, kv_source)]))
        except ValueError as exc:
            assert "NoPE KV source" in str(exc)
        else:
            raise AssertionError("Missing native BF16 NoPE padding was not reproduced")
        current = "isolated_source_config"
        install(args.source / "config.json")
        fixed = SimpleNamespace(quant_config=quant, speculative_config=spec)
        result = utils.load_eagle_model(target, fixed)
        assert fixed.quant_config is quant
        assert hashlib.sha256(pickle.dumps(quant, protocol=5)).hexdigest() == target_before
        assert result is created[-1]
        assert result.model.embed_tokens is target.language_model.model.embed_tokens
        assert result.lm_head is target.language_model.lm_head
        assert result.model.layers["45"].shared_head.head is target.language_model.lm_head
        assert result.model.topk_indices_buffer is target.language_model.model.topk_indices_buffer
        assert (result.model.layers["45"].attention.topk_indices_buffer
                is target.language_model.model.topk_indices_buffer)
        assert not hasattr(result.lm_head, "weight") and not hasattr(result.model.embed_tokens, "weight")
        loaded = Draft.load_weights(fake, iter([(key, kv_source)]))
        assert loaded == {"model.layers.45.mtp_block.self_attn.fused_qkv_a_proj.weight"}
        assert torch.equal(fused.weight[1536:2048].view(torch.int16), kv_source.view(torch.int16))
        assert torch.count_nonzero(fused.weight[2048:].view(torch.int16)).item() == 0
        loader_report = report(2)
    finally:
        utils.get_model = original_factory
        utils.get_pp_group = original_pp
        native_mtp.fused_moe_make_expert_params_mapping = original_expert_mapping
        Draft.load_weights = original_weights
    assert len(suffixes) == 872
    mismatch_before = sum(a != b for a, b in zip(
        expected, observed["inherited_target_config"], strict=True))
    mismatch_after = sum(a != b for a, b in zip(
        expected, observed["isolated_source_config"], strict=True))
    assert mismatch_before == 870 and mismatch_after == 0
    assert not torch.cuda.is_initialized()
    records = [{"draft_name": "model.layers.45." + suffix, "expected": want,
                "inherited": before, "isolated": after}
               for suffix, want, before, after in zip(suffixes, expected,
                    observed["inherited_target_config"], observed["isolated_source_config"], strict=True)]
    output = {"status": "passed", "cuda_initialized": False, "full_model_loaded": False,
        "real_native_helper": "vllm.v1.worker.gpu.spec_decode.eagle.utils.load_eagle_model",
        "native_model_factory_replaced_by_meta_fixture": True,
        "selectors": len(records), "mismatches_inherited": mismatch_before,
        "mismatches_isolated": mismatch_after, "native_reuse_mutates_target_policy": inherited_changed,
        "fixed_target_policy_byte_pickle_unchanged": True,
        "native_whole_module_vocabulary_sharing_without_dense_weights": True,
        "native_inner_and_per_layer_index_buffer_sharing": True, "loader": loader_report,
        "real_cpu_bf16_projection_fixture": {"source_shape": list(kv_source.shape),
            "padded_shape": [576, 4096], "native_unpadded_loader_shape_failure_reproduced": True,
            "fixed_native_load_weights_binding_passed": True,
            "source_values_bits_exact": True, "rope_rows_zero": True,
            "expert_mapping_stubbed_for_this_single_projection_fixture": True},
        "source_config_sha256": hashlib.sha256((args.source / "config.json").read_bytes()).hexdigest(),
        "code_sha256": {"audit": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "loader": hashlib.sha256(Path(__file__).with_name("mtp_draft_loader.py").read_bytes()).hexdigest(),
            "native_helper": hashlib.sha256(Path(inspect.getfile(utils)).read_bytes()).hexdigest()},
        "records": records,
        "limits": ["CPU tests actual native orchestration/config/sharing with a substituted factory.",
                   "One source BF16 projection loads on CPU into bounded stand-in parameter storage.",
                   "No full draft construction, CUDA, argmax or numerical inference."]}
    args.report.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k != "records"}))


if __name__ == "__main__":
    main()
