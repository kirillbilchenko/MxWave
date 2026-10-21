"""Opt-in, in-memory KDA MXFP8 conversion for the isolated serving experiment."""

from __future__ import annotations

import gc
import hashlib
import inspect
import json
import os
import time
import weakref
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .kda_mxfp8_roundtrip import numpy_reference, roundtrip

MARLIN_SHA256 = "a5cc60882dc32851b2cb7d573e2b28980319a99ef816fd6ed9925b6fabfffd9a"


def digest(value: torch.Tensor) -> str:
    """Hash contiguous tensor bytes without changing their numerical representation."""
    return hashlib.sha256(value.detach().contiguous().view(torch.uint8).cpu().numpy()).hexdigest()


@torch.inference_mode()
def encode_weight(
    weight: torch.Tensor, chunk_rows: int = 512,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Encode one BF16 matrix using the previously measured rounding recipe."""
    if (
        weight.dtype != torch.bfloat16 or weight.ndim != 2
        or weight.shape[1] % 32 or weight.numel() == 0 or chunk_rows < 1
    ):
        raise ValueError("Require nonempty BF16 rows, 32-aligned columns and positive chunk size")
    n, k = weight.shape
    codes = torch.empty((n, k), dtype=torch.uint8, device=weight.device)
    scales = torch.empty((n, k // 32), dtype=torch.uint8, device=weight.device)
    decoded = torch.empty_like(weight)
    source_hash = hashlib.sha256()
    rounded_hash = hashlib.sha256()
    for start in range(0, n, chunk_rows):
        source = weight[start:start + chunk_rows]
        values, code, scale = roundtrip(source)
        rounded = values.bfloat16()
        if not torch.equal(rounded.float(), values):
            raise ValueError("The declared decoded weights must be exactly BF16-representable")
        codes[start:start + chunk_rows].copy_(code)
        scales[start:start + chunk_rows].copy_(scale)
        decoded[start:start + chunk_rows].copy_(rounded)
        source_hash.update(source.contiguous().view(torch.uint8).cpu().numpy().tobytes())
        rounded_hash.update(rounded.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return codes, scales, decoded, {
        "shape": [n, k], "source_sha256": source_hash.hexdigest(),
        "rounded_sha256": rounded_hash.hexdigest(),
        "encoded_bytes": codes.numel() + scales.numel(),
    }


def method_type() -> type:
    """Build an adapter which delegates only to the installed MXFP8 Marlin consumer."""
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
        apply_mxfp8_marlin_linear,
    )

    class KdaMxfp8MarlinMethod(UnquantizedLinearMethod):
        """Preserve vLLM's linear interface while consuming repacked MXFP8 weights."""

        def apply(
            self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None,
        ) -> torch.Tensor:
            """Run the same native W8A16 helper used by the installed MXFP8 route."""
            return apply_mxfp8_marlin_linear(
                input=x, weight=layer.weight, weight_scale=layer.weight_scale,
                workspace=layer.workspace, size_n=layer.output_size_per_partition,
                size_k=layer.input_size_per_partition, bias=bias,
            )

        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            """Reject accidentally processing an already converted layer a second time."""
            raise RuntimeError("KDA MXFP8 conversion is performed exactly once before graph capture")

    return KdaMxfp8MarlinMethod


@torch.inference_mode()
def install(model: torch.nn.Module, reference_path: Path, output: Path) -> dict[str, Any]:
    """Convert all 68 audited KDA matrices and verify native consumers before capture."""
    from vllm.distributed import get_tensor_model_parallel_rank
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.layers.quantization.utils import marlin_utils_fp8

    from .kda_policy import load_policy

    if output.exists():
        raise FileExistsError("Preserve the first conversion attempt")
    if torch.get_default_dtype() != torch.bfloat16:
        raise ValueError("Match the installed BF16 activation/scaling policy")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Convert weights before capturing graphs")
    source = Path(inspect.getfile(marlin_utils_fp8))
    if hashlib.sha256(source.read_bytes()).hexdigest() != MARLIN_SHA256:
        raise ValueError("Installed MXFP8 helper changed from the reviewed source")
    rank = get_tensor_model_parallel_rank()
    reference_bytes = reference_path.read_bytes()
    reference = json.loads(reference_bytes)
    expected = {row["name"]: row for row in reference["ranks"][str(rank)]}
    targets = load_policy(Path(os.environ["MXWAVE_GLM53_MODEL"]))
    modules = dict(model.named_modules())
    if set(expected) != set(targets) or len(targets) != 68:
        raise ValueError("Require exact coverage of the already measured KDA rounding")
    report: dict[str, Any] = {
        "status": "running", "rank": rank, "started_epoch": time.time(),
        "marlin_source_sha256": MARLIN_SHA256,
        "rounding_reference_sha256": hashlib.sha256(reference_bytes).hexdigest(),
        "layers": [], "cuda_allocated_before": torch.cuda.memory_allocated(),
        "actual_MXFP8_W8A16": True, "checkpoint_files_changed": False,
        "full_model_KL_qualified": False,
    }
    original_refs = []
    cls = method_type()
    generator = torch.Generator(device="cuda").manual_seed(20261020)
    checked_kinds: set[str] = set()

    def save() -> None:
        temporary = output.with_suffix(".incomplete")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(output)

    save()
    try:
        for name, target in targets.items():
            layer = modules[name]
            if type(layer.quant_method) is not UnquantizedLinearMethod:
                raise ValueError(f"Expected the original BF16 KDA method: {name}")
            weight = layer.weight
            if tuple(weight.shape) != (target.rows, target.width):
                raise ValueError(f"Unexpected TP2 matrix geometry: {name}")
            if layer.bias is not None or hasattr(layer, "weight_scale"):
                raise ValueError(f"Unexpected original bias/scales: {name}")
            original_refs.append(weakref.ref(weight))
            codes, scales, decoded, record = encode_weight(weight)
            if (
                record["source_sha256"] != expected[name]["source_sha256"]
                or record["rounded_sha256"] != expected[name]["result_sha256"]
            ):
                raise ValueError(f"Rounding differs from the measured quality experiment: {name}")
            sample = torch.cat((weight[:2], weight[-2:])).float().cpu().numpy()
            wanted = numpy_reference(sample)
            for actual, correct in zip(
                (torch.cat((decoded[:2], decoded[-2:])).float().cpu().numpy(),
                 torch.cat((codes[:2], codes[-2:])).cpu().numpy(),
                 torch.cat((scales[:2], scales[-2:])).cpu().numpy()),
                wanted, strict=True,
            ):
                np.testing.assert_array_equal(actual, correct)
            record.update(name=name, kind=target.kind, layer=target.layer,
                          independent_sample_elements=int(sample.size), checks=[])
            report["layers"].append(record)
            layer.weight = torch.nn.Parameter(codes.view(torch.float8_e4m3fn), requires_grad=False)
            layer.register_parameter("weight_scale", torch.nn.Parameter(scales, requires_grad=False))
            marlin_utils_fp8.prepare_mxfp8_layer_for_marlin(layer)
            layer.quant_method = cls()
            columns = [0, 31, target.width // 2, target.width - 1]
            basis = torch.zeros((4, target.width), dtype=torch.bfloat16, device=weight.device)
            basis[torch.arange(4, device=weight.device), columns] = 1
            actual = layer.quant_method.apply(layer, basis)
            if not torch.equal(actual, decoded[:, columns].T):
                raise ValueError(f"Native MXFP8 basis decoding differs: {name}")
            record["basis_bits_exact"] = True
            rows = [8]
            if target.kind not in checked_kinds:
                rows = [1, 8, 12, 16, 24, 32, 64, 128, 256, 1024, 2048]
                checked_kinds.add(target.kind)
            # Independent FP32 accumulation over decoded weights tests the actual native layout.
            decoded_fp32 = decoded.float()
            for count in rows:
                x = torch.randn(
                    (count, target.width), generator=generator, device=weight.device,
                    dtype=torch.float32,
                ).bfloat16() * 0.125
                expected_output = (x.float() @ decoded_fp32.T).bfloat16()
                actual = layer.quant_method.apply(layer, x)
                if actual.shape != expected_output.shape or not bool(torch.isfinite(actual).all()):
                    raise ValueError(f"Invalid native output shape or values: {name}, M={count}")
                delta = actual.float() - expected_output.float()
                relative = float(delta.norm() / expected_output.float().norm().clamp_min(1e-30))
                record["checks"].append({
                    "rows": count, "relative_L2": relative,
                    "maximum_absolute_error": float(delta.abs().max()),
                    "reference": "FP32 matmul of the declared rounded weights, cast to BF16",
                    "existing_component_relative_L2_limit": 0.01,
                })
                if relative > 0.01:
                    raise ValueError(f"Native MXFP8 component reference failed: {name}, M={count}")
            record["resident"] = {
                "weight_shape": list(layer.weight.shape), "weight_dtype": str(layer.weight.dtype),
                "scale_shape": list(layer.weight_scale.shape),
                "scale_dtype": str(layer.weight_scale.dtype),
                "weight_bytes": layer.weight.numel() * layer.weight.element_size(),
                "scale_bytes": layer.weight_scale.numel() * layer.weight_scale.element_size(),
                "workspace_bytes": layer.workspace.numel() * layer.workspace.element_size(),
            }
            del weight, codes, scales, decoded, decoded_fp32, actual, expected_output, delta, x, basis
            save()
        gc.collect()
        torch.cuda.synchronize()
        if any(ref() is not None for ref in original_refs):
            raise RuntimeError("An original BF16 KDA parameter is still retained")
        report.update(
            status="complete", finished_epoch=time.time(), original_BF16_parameters_alive=0,
            cuda_allocated_after=torch.cuda.memory_allocated(),
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            all_68_rounding_hashes_match_quality_run=True,
        )
        save()
    except BaseException as exc:
        report.update(status="failed", error=repr(exc), finished_epoch=time.time())
        save()
        raise
    return report
