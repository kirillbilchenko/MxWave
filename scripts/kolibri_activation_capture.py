"""Capture actual routed RMS inputs through the installed vLLM FP8 kernel interfaces."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

from mxwave.calibration import save_calibration_data
from mxwave.routed_calibration import RoutedLayerMoments, dequantize_fp8_activation_blocks


def install_capture(model) -> dict:
    """Install process-local observers without changing the teacher's inputs or outputs."""
    from vllm.model_executor.layers.fused_moe.experts import triton_moe

    if hasattr(model, "_mxwave_routed_capture"):
        raise RuntimeError("A routed activation capture is already installed")
    config = model.config
    state = {
        "layers": {}, "active_layer": None, "hidden": None, "routes": None,
        "down_calls": 0, "handles": [], "tokens": {},
    }
    old_apply = triton_moe.TritonExperts.apply
    signature = inspect.signature(old_apply)
    old_quant = triton_moe.ops.silu_and_mul_per_block_quant

    def begin(_module, inputs, *, layer):
        hidden = inputs[0]
        if hidden.ndim != 2 or hidden.shape[1] != config.hidden_size:
            raise ValueError("Unexpected Kolibri MLP input layout")
        state["active_layer"], state["hidden"] = layer, hidden
        state["tokens"][layer] = state["tokens"].get(layer, 0) + hidden.shape[0]
        if layer not in state["layers"]:
            state["layers"][layer] = RoutedLayerMoments.create(
                config.num_experts, config.hidden_size, config.moe_intermediate_size, hidden.device,
            )

    def capture_quant(*args, **kwargs):
        result = old_quant(*args, **kwargs)
        if state["routes"] is not None:
            ids, weights = state["routes"]
            quantized, scales = result
            values = dequantize_fp8_activation_blocks(quantized, scales)
            state["layers"][state["active_layer"]].update_down(values, ids, weights)
            state["down_calls"] += 1
        return result

    def capture_apply(kernel, *args, **kwargs):
        arguments = signature.bind(kernel, *args, **kwargs).arguments
        if (arguments["expert_map"] is not None or arguments["apply_router_weight_on_input"]
                or arguments["global_num_experts"] != config.num_experts):
            raise ValueError("Capture only supports unsharded, output-weighted Kolibri routing")
        layer = state["active_layer"]
        if layer is None or state["hidden"] is None or state["routes"] is not None:
            raise ValueError("Routed kernel is outside its expected MLP observer context")
        ids, weights = arguments["topk_ids"], arguments["topk_weights"]
        state["layers"][layer].update_inputs(state["hidden"], ids, weights)
        before = state["down_calls"]
        state["routes"] = ids, weights
        try:
            result = old_apply(kernel, *args, **kwargs)
        finally:
            state["routes"] = None
        if state["down_calls"] != before + 1:
            raise ValueError("Did not capture exactly one actual down input for this routed call")
        return result

    from functools import partial

    for index, layer in enumerate(model.model.layers):
        state["handles"].append(layer.mlp.register_forward_pre_hook(partial(begin, layer=index)))
    triton_moe.TritonExperts.apply = capture_apply
    triton_moe.ops.silu_and_mul_per_block_quant = capture_quant
    state["restore"] = lambda: (
        setattr(triton_moe.TritonExperts, "apply", old_apply),
        setattr(triton_moe.ops, "silu_and_mul_per_block_quant", old_quant),
    )
    model._mxwave_routed_capture = state
    return {"status": "installed", "layers": len(state["handles"]), "experts": config.num_experts}


def finish_capture(model, *, path: str, metadata: dict[str, str]) -> dict:
    """Write observed RMS targets, with explicit retained targets and routing coverage."""
    state = model._mxwave_routed_capture
    if len(state["layers"]) != model.config.num_hidden_layers:
        raise ValueError("Calibration did not execute every decoder layer")
    statistics, layers = {}, []
    for index, moments in sorted(state["layers"].items()):
        values, report = moments.finalize(index)
        statistics.update(values)
        report["input_tokens"] = state["tokens"][index]
        layers.append(report)
    metadata = {
        **metadata, "policy": "kolibri1-routed-experts", "statistics_teacher": "official FP8",
        "routing_weighting": "actual selected expert IDs and squared output router weights",
        "down_inputs": "actual FP8 down-GEMM inputs decoded with their float32 block scales",
        "gamma_normalization": "per-target maximum; objective-preserving scalar normalization",
        "unobserved_targets": "retain the previous unweighted quantized payloads",
    }
    save_calibration_data(path, {"rms": statistics}, metadata)
    expected = model.config.num_hidden_layers * model.config.num_experts * 3
    report = {
        "status": "passed", "calibrated_targets": len(statistics), "expected_targets": expected,
        "retained_unweighted_targets": expected - len(statistics), "layers": layers,
        "file_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "capture_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "metadata": metadata,
    }
    Path(path).with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    for handle in state["handles"]:
        handle.remove()
    state["restore"]()
    del model._mxwave_routed_capture
    return {key: report[key] for key in
            ("status", "calibrated_targets", "expected_targets", "retained_unweighted_targets", "file_sha256")}
