"""Read-only CPU audit of native MTP selectors in the installed stock image."""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import struct
from pathlib import Path
from typing import Any


def main() -> None:
    """Check checkpoint coverage and native draft policy without constructing a model."""
    import torch
    from vllm import platforms
    from vllm.platforms.cpu import CpuPlatform

    platforms.current_platform = CpuPlatform()
    from vllm.config.speculative import SpeculativeConfig
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )
    from vllm.model_executor.model_loader.utils import configure_quant_config
    from vllm.models.glm5next.common.model import Glm5NextForConditionalGeneration as Base
    from vllm.models.glm5next.common.mtp import Glm5NextMTP as Draft
    from vllm.v1.worker.gpu.spec_decode.eagle import utils

    from mxwave_glm53.plugin import check_upstream

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("Retain the original audit receipt")
    check_upstream()
    source = json.loads((args.source / "config.json").read_bytes())
    config = source["text_config"]
    assert config["num_hidden_layers"] == 45 and config["num_nextn_predict_layers"] == 1
    # These are the independently maintained aliases used by our installed plugin.
    Base.packed_modules_mapping = dict(Base.packed_modules_mapping) | {
        "in_proj_qkvbfg_a": ["q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj", "g_a_proj"],
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
        "wk_weights_proj": ["wk", "weights_proj"],
    }

    def quant(model: type[Any]) -> Any:
        value = CompressedTensorsConfig.from_config(copy.deepcopy(source["quantization_config"]))
        configure_quant_config(value, model)
        return value

    target_quant, draft_quant = quant(Base), quant(Draft)
    dummy = torch.nn.Linear(2, 2, bias=False, device="meta")

    def policy(value: Any, name: str) -> str | None:
        selected = value.get_scheme_dict(dummy, name)
        return str(selected["format"]) if selected else None

    suffixes = ["self_attn.fused_qkv_a_proj", "self_attn.q_b_proj", "self_attn.kv_b_proj",
                "self_attn.o_proj", "self_attn.indexer.wk_weights_proj", "mlp.gate",
                "mlp.shared_experts.gate_up_proj", "mlp.shared_experts.down_proj"]
    suffixes += [f"mlp.experts.{i}.{p}" for i in range(288)
                 for p in ("gate_proj", "up_proj", "down_proj")]
    records = []
    for suffix in suffixes:
        name = "model.language_model.layers.45." + suffix
        records.append({
            "source_name": name, "draft_name": "model.layers.45." + suffix,
            "expected": policy(target_quant, Base.hf_to_vllm_mapper._map_name(name)),
            "native_fresh": policy(draft_quant, "model.layers.45." + suffix),
            "target_inherited": policy(target_quant, "model.layers.45." + suffix),
        })
    index = json.loads((args.source / "model.safetensors.index.json").read_bytes())["weight_map"]
    names = {n for n in index if n.startswith("model.language_model.layers.45.")}
    headers, proofs = {}, []
    for shard in sorted({index[n] for n in names}):
        with (args.source / shard).open("rb") as handle:
            size = struct.unpack("<Q", handle.read(8))[0]
            assert 0 < size <= 64 * 1024**2
            raw = handle.read(size)
            assert len(raw) == size
        headers.update({n: v for n, v in json.loads(raw).items() if n in names})
        proofs.append({"shard": shard, "header_sha256": hashlib.sha256(raw).hexdigest()})
    assert len(names) == 1756 and set(headers) == names
    dtypes: dict[str, int] = {}
    for spec in headers.values():
        dtypes[spec["dtype"]] = dtypes.get(spec["dtype"], 0) + 1
    assert not torch.cuda.is_initialized()
    audited = {
        "target_class": Base, "draft_class": Draft, "quant_selector": CompressedTensorsConfig,
        "draft_loader": utils.load_eagle_model,
        "draft_overrides": SpeculativeConfig.apply_draft_overrides,
    }
    code = {n: inspect.getsource(value) for n, value in audited.items()}
    for name in ("draft_loader", "draft_overrides"):
        args.report.with_name(name + ".txt").write_text(code[name])
    result = {
        "status": "audit_completed", "gpu_used": False, "model_constructed": False,
        "inference_tested": False, "selectors": records,
        "native_fresh_mismatches": sum(r["expected"] != r["native_fresh"] for r in records),
        "target_inherited_mismatches": sum(
            r["expected"] != r["target_inherited"] for r in records),
        "draft_has_mapper": hasattr(Draft, "hf_to_vllm_mapper"),
        "draft_packed_aliases": getattr(Draft, "packed_modules_mapping", {}),
        "source_tensors": len(names), "source_dtype_counts": dtypes,
        "source_header_proofs": proofs,
        "geometry": {k: config[k] for k in (
            "hidden_size", "q_lora_rank", "kv_lora_rank", "qk_rope_head_dim")},
        "source_sha256": {name: hashlib.sha256((args.source / name).read_bytes()).hexdigest()
                          for name in ("config.json", "model.safetensors.index.json")},
        "code_sha256": {n: hashlib.sha256(s.encode()).hexdigest() for n, s in code.items()},
    }
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("selectors", "source_header_proofs", "code_sha256")}))


if __name__ == "__main__":
    main()
