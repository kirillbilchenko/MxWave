"""Check the actual checkpoint's native MTP BF16 projection without constructing a model."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def main() -> None:
    """Load both source projections through native CPU weight binding at declared dimensions."""
    import torch
    from mtp_draft_loader import CONFIG_SHA, bf16_projection_slices, install
    from vllm import platforms
    from vllm.platforms.cpu import CpuPlatform

    platforms.current_platform = CpuPlatform()
    import mtp_draft_loader as bridge
    import vllm.models.glm5next.nvidia.mtp as native
    from safetensors import safe_open
    from vllm.v1.worker.gpu.spec_decode.eagle import utils

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("Preserve the first native projection geometry proof")
    raw = (args.source / "config.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != CONFIG_SHA:
        raise ValueError("MTP projection fixture requires the audited source config")
    declared = json.loads(raw)["text_config"]
    geometry = {"q_rank": int(declared["q_lora_rank"]),
        "kv_rank": int(declared["kv_lora_rank"]),
        "rope_rows": int(declared["qk_rope_head_dim"]),
        "hidden_size": int(declared["hidden_size"]),
        "nope": int(declared["qk_rope_head_dim"]) == 0}
    if geometry != {"q_rank": 1536, "kv_rank": 512, "rope_rows": 0,
                    "hidden_size": 4096, "nope": True}:
        raise ValueError("Actual source MTP geometry differs from the zero-rope retry scope")
    index = json.loads((args.source / "model.safetensors.index.json").read_bytes())["weight_map"]
    prefix = "model.language_model.layers.45.self_attn."
    sources = {}
    for suffix in ("q_a_proj.weight", "kv_a_proj_with_mqa.weight"):
        key = prefix + suffix
        with safe_open(args.source / index[key], framework="pt", device="cpu") as handle:
            sources[key] = handle.get_tensor(key)
    q, kv = sources.values()

    def save_values(name: str, value: Any) -> dict[str, Any]:
        """Retain bounded raw BF16 evidence for an independent byte comparison."""
        path = args.report.parent / (args.report.stem + "-" + name + ".bin.gz")
        if path.exists():
            raise FileExistsError("Preserve the first native projection artifact")
        raw = value.detach().contiguous().view(torch.uint8).numpy().tobytes()
        with path.open("xb") as handle, gzip.GzipFile(fileobj=handle, mode="wb", mtime=0) as packed:
            packed.write(raw)
        return {"path": path.name, "shape": list(value.shape), "dtype": str(value.dtype),
            "raw_bytes": len(raw), "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    source_artifacts = {name: save_values("source-" + str(i), value)
                        for i, (name, value) in enumerate(sources.items())}
    rows = geometry["q_rank"] + geometry["kv_rank"] + geometry["rope_rows"]
    fake = object.__new__(native.Glm5NextMTP)
    torch.nn.Module.__init__(fake)
    fake.config = SimpleNamespace(**(declared | {"mla_nope": geometry["nope"]}))
    fake.model = torch.nn.Module()
    fake.model.mtp_start_layer_idx = 45
    fake.model.num_mtp_layers = 1
    layer = torch.nn.Module()
    layer.mtp_block = torch.nn.Module()
    layer.mtp_block.self_attn = torch.nn.Module()
    fused = torch.nn.Module()
    fused.register_parameter("weight", torch.nn.Parameter(
        torch.empty((rows, geometry["hidden_size"]), dtype=torch.bfloat16), requires_grad=False))
    calls: list[dict[str, Any]] = []

    def projection_load(param: Any, value: Any, shard: int) -> None:
        """Stand in for destination storage only; leave native naming and shard selection intact."""
        start, count = {0: (0, geometry["q_rank"]),
                        1: (geometry["q_rank"], geometry["kv_rank"])}[shard]
        if tuple(value.shape) != (count, geometry["hidden_size"]):
            raise ValueError("Native weight binding supplied unexpected projection rows")
        param.data[start:start + count].copy_(value)
        calls.append({"shard": shard, "shape": list(value.shape)})

    fused.weight.weight_loader = projection_load
    layer.mtp_block.self_attn.fused_qkv_a_proj = fused
    fake.model.layers = torch.nn.ModuleDict({"45": layer})
    original_weights = native.Glm5NextMTP.load_weights
    original_mapping = native.fused_moe_make_expert_params_mapping
    original_factory = utils.get_model
    results = []
    try:
        native.fused_moe_make_expert_params_mapping = lambda *args, **kwargs: []
        for label in ("native", "isolated_loader_bridge"):
            calls.clear()
            fused.weight.data.fill_(123)
            if label == "isolated_loader_bridge":
                install(args.source / "config.json")
            loaded = native.Glm5NextMTP.load_weights(fake, iter(sources.items()))
            if loaded != {"model.layers.45.mtp_block.self_attn.fused_qkv_a_proj.weight"}:
                raise ValueError("Native source-to-fused parameter binding changed")
            checks = bf16_projection_slices(fused.weight, q, kv, **geometry)
            exact = [torch.equal(a.view(torch.int16), b.view(torch.int16)) for _, a, b in checks]
            if len(checks) != 2 or not all(exact) or sorted(c["shard"] for c in calls) != [0, 1]:
                raise ValueError("Source BF16 projection bytes did not survive native binding")
            results.append({"path": label, "shape": list(fused.weight.shape),
                "artifact": save_values(label, fused.weight),
                "calls": list(calls), "checks": [{"name": name, "bits_exact": match}
                    for (name, _, _), match in zip(checks, exact, strict=True)]})
        if bridge._PADDING or len(bridge._WEIGHT_LOADS) != 1:
            raise RuntimeError("Zero-rope source unexpectedly padded or bypassed bridge loading")
    finally:
        native.Glm5NextMTP.load_weights = original_weights
        native.fused_moe_make_expert_params_mapping = original_mapping
        utils.get_model = original_factory
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU projection audit initialized CUDA")
    report = {"status": "passed", "cuda_initialized": False, "full_model_loaded": False,
        "source_config_sha256": CONFIG_SHA, "projection_geometry": geometry,
        "nope_derived_from_source_zero_rope_width": True,
        "fused_projection_shape": [rows, geometry["hidden_size"]],
        "required_projection_byte_checks": 2, "prospective_selected_draft_checks_per_rank": 18,
        "prospective_other_checks_not_run_here": 16, "padding_required": False,
        "padding_events": list(bridge._PADDING), "results": results,
        "source_tensors": [{"name": name, "shape": list(value.shape),
            "artifact": source_artifacts[name],
            "bf16_sha256": hashlib.sha256(value.contiguous().view(torch.uint8).numpy()).hexdigest()}
            for name, value in sources.items()],
        "code_sha256": {"audit": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "bridge": hashlib.sha256(Path(inspect.getfile(bridge)).read_bytes()).hexdigest(),
            "native_mtp": hashlib.sha256(Path(inspect.getfile(native)).read_bytes()).hexdigest()},
        "limits": ["Real native load_weights naming/shard orchestration; bounded destination storage stand-in.",
            "Expert mapping is stubbed; no expert/shared weight check runs in this fixture.",
            "No model constructor, draft inference, acceptance, quality or speed measurement."]}
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("source_tensors", "results")}))


if __name__ == "__main__":
    main()
