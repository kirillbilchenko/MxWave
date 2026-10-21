"""An inert-by-default vLLM plugin; activation requires a verified asset bundle."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path

from .bundle import PINNED_ASSETS, VLLM_VERSION, digest, verify_bundle

_ACTIVE: tuple[str, ...] | None = None


def check_upstream() -> dict[str, str]:
    """Check the exact upstream API implementation before wrapping internal methods."""
    version = importlib.metadata.version("vllm")
    if version != VLLM_VERSION:
        raise RuntimeError(f"mxwave-glm53 requires vllm=={VLLM_VERSION}; found {version}")
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        raise RuntimeError("Cannot locate installed vLLM")
    root = Path(spec.origin).parent.parent
    contract = json.loads(Path(__file__).with_name("upstream-contract.json").read_bytes())
    actual = {name: digest(root / name) for name in contract["files"]}
    if actual != contract["files"]:
        changed = [name for name in actual if actual[name] != contract["files"][name]]
        raise RuntimeError(f"Installed vLLM internals differ from reviewed 0.32.0: {changed}")
    return actual


def register() -> None:
    """Install only the selected CT consumers in each opted-in vLLM process."""
    global _ACTIVE
    if os.environ.get("MXWAVE_GLM53_ENABLE") != "1":
        return
    identity = tuple(os.environ[key] for key in (
        "MXWAVE_GLM53_BUNDLE", "MXWAVE_GLM53_MODEL", "MXWAVE_GLM53_REPORTS"))
    identity += (os.environ.get("MXWAVE_GLM53_VOCAB_MODE", "all"),)
    if _ACTIVE is not None:
        if _ACTIVE != identity:
            raise RuntimeError("Cannot change an active runtime bundle within one process")
        return
    check_upstream()
    bundle = Path(identity[0])
    verify_bundle(bundle)
    os.environ.update({
        "MXWAVE_MOE_MODE": "aligned", "MXWAVE_MOE_DISPATCH": "dispatcher",
        "MXWAVE_MOE_RAW_LIBRARY": str(bundle / "native/libmxwave_moe_raw.so"),
        "MXWAVE_MOE_ALIGNED_LIBRARY": str(bundle / "native/libmxwave_moe_sparse.so"),
        # Sparse-palette encoding does not use the historical external record codec.
        "MXWAVE_MOE_RECORD_CODEC": str(bundle / "bundle.json"),
        "MXWAVE_ZIPSERV_LIBRARY": str(bundle / "native/libmxwave_kda_zipserv.so"),
        "MXWAVE_GLM53_VOCAB_ARTIFACTS": str(bundle / "vocab"),
        "MXWAVE_GLM53_VOCAB_MANIFEST_SHA256": PINNED_ASSETS["vocab/prepared.json"],
        "MXWAVE_GLM53_ATTENTION_MAPPING": "1",
        "MXWAVE_GLM53_KDA_SOURCE": identity[1],
    })
    from vllm.models.glm5next.common.model import Glm5NextForConditionalGeneration

    aliases = {
        "in_proj_qkvbfg_a": ["q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj", "g_a_proj"],
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
        "wk_weights_proj": ["wk", "weights_proj"],
    }
    mapping = dict(Glm5NextForConditionalGeneration.packed_modules_mapping)
    for key, parts in aliases.items():
        if key in mapping and mapping[key] != parts:
            raise RuntimeError(f"Upstream fused-module mapping changed: {key}")
        mapping[key] = parts
    Glm5NextForConditionalGeneration.packed_modules_mapping = mapping
    from ._runtime import model_plugin, vocab_adapter
    from .worker import install_worker

    vocab_adapter.install()
    model_plugin.install()
    install_worker()
    _ACTIVE = identity
