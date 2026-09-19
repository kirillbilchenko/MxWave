"""Tests for bounded packed-prefix counteraction replay."""

from __future__ import annotations

import math
from pathlib import Path
from typing import ClassVar

import pytest
import torch
from safetensors.torch import save_file

from mxwave.calibration import CalibrationData
from mxwave.core import quantize_mxfp4
from mxwave.counteraction_probe import probe_mlp_counteraction
from mxwave.mxfp4_candidates import Mxfp4CandidateSpec
from mxwave.runtime_ir import (
    ResponsePoint,
    RuntimeGraph,
    RuntimeLinearGroup,
    RuntimeOperation,
    RuntimeOutputPath,
    RuntimeWeight,
)


class _TinyConfig:
    model_type = "qwen3_5_text"
    layer_types: ClassVar[list[str]] = [
        "full_attention",
        "full_attention",
        "full_attention",
    ]


class _TinyRotary(torch.nn.Module):
    def __init__(self, _config: object) -> None:
        super().__init__()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (1, position_ids.shape[-1], 1)
        return hidden_states.new_ones(shape), hidden_states.new_zeros(shape)


class _TinyMlp(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = torch.nn.Linear(32, 32, bias=False)
        self.up_proj = torch.nn.Linear(32, 32, bias=False)
        self.down_proj = torch.nn.Linear(32, 32, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        gated = torch.nn.functional.silu(self.gate_proj(inputs)) * self.up_proj(inputs)
        return self.down_proj(gated)


class _TinyLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = _TinyMlp()

    def forward(self, hidden_states: torch.Tensor, **_kwargs: object) -> torch.Tensor:
        return hidden_states + self.mlp(hidden_states)


class _TinyBase(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _TinyConfig()
        self.embed_tokens = torch.nn.Embedding(64, 32)
        self.layers = torch.nn.ModuleList([_TinyLayer(), _TinyLayer(), _TinyLayer()])
        self.rotary_emb = _TinyRotary(self.config)
        self.norm = torch.nn.LayerNorm(32)


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = _TinyBase()
        self.lm_head = torch.nn.Linear(32, 48, bias=False)


def _graph() -> RuntimeGraph:
    prefix = "language_model.layers.1.mlp"
    operation = RuntimeOperation(
        name=prefix,
        layer_index=1,
        kind="gated-mlp",
        linear_groups=(
            RuntimeLinearGroup(
                f"{prefix}.gate_up_proj",
                (
                    RuntimeWeight(f"{prefix}.gate_proj.weight", "gate"),
                    RuntimeWeight(f"{prefix}.up_proj.weight", "up"),
                ),
            ),
            RuntimeLinearGroup(
                f"{prefix}.down_proj",
                (RuntimeWeight(f"{prefix}.down_proj.weight", "down"),),
            ),
        ),
        response_points=(ResponsePoint("output"),),
    )
    return RuntimeGraph(
        "test",
        "1",
        (operation,),
        output_path=RuntimeOutputPath("language_model.norm", "lm_head"),
    )


def _calibration() -> CalibrationData:
    identity = torch.eye(32).unsqueeze(0)
    prefix = "language_model.layers.1.mlp"
    return CalibrationData(
        objective="block-hessian",
        tensors={
            f"{prefix}.{projection}.weight": identity.clone()
            for projection in ("gate_proj", "up_proj", "down_proj")
        },
        metadata={},
        file_sha256="test",
    )


def _packed_baseline(
    source: dict[str, torch.Tensor],
    calibration: CalibrationData,
) -> dict[str, torch.Tensor]:
    baseline: dict[str, torch.Tensor] = {}
    for name, value in source.items():
        if ".mlp." not in name or not name.endswith(".weight"):
            baseline[name] = value.contiguous()
            continue
        packed, scales = quantize_mxfp4(
            value,
            hessian=calibration.tensors.get(name),
            scale_percentile=99.5,
            mse_clip_depth=2,
        )
        module = name.removesuffix(".weight")
        baseline[f"{module}.weight_packed"] = packed
        baseline[f"{module}.weight_scale"] = scales
    return baseline


def test_probe_uses_real_inherited_error_and_reproduces_baseline(tmp_path: Path) -> None:
    torch.manual_seed(47)
    model = _TinyModel().eval()
    original = {
        name: value.detach().clone().contiguous()
        for name, value in model.state_dict().items()
    }
    source_checkpoint = tmp_path / "source.safetensors"
    save_file(original, source_checkpoint)
    source_files = {name: source_checkpoint for name in original}

    calibration = _calibration()
    baseline_tensors = _packed_baseline(original, calibration)
    baseline_checkpoint = tmp_path / "baseline.safetensors"
    save_file(baseline_tensors, baseline_checkpoint)
    baseline_files = {name: baseline_checkpoint for name in baseline_tensors}

    result = probe_mlp_counteraction(
        model,
        _graph(),
        calibration,
        1,
        (
            Mxfp4CandidateSpec("rtn", method="rtn"),
            Mxfp4CandidateSpec(
                "block-hessian",
                weighting="block-hessian",
                mse_clip_depth=2,
            ),
        ),
        ([1, 2, 3, 4], [5, 6, 7, 8]),
        source_files,
        baseline_files,
        device=torch.device("cpu"),
        dtype=torch.float32,
        row_chunk_size=5,
        logit_positions_per_sequence=2,
        suffix_jvp=True,
    )

    candidates = {candidate.candidate: candidate for candidate in result.candidates}
    candidate = candidates["block-hessian"]
    assert candidate.baseline_weight_nmse == 0.0
    assert candidate.mean_operator_nmse == 0.0
    assert candidate.sample_teacher_kl == result.baseline.sample_teacher_kl
    assert candidate.mean_suffix_jvp_teacher_kl == pytest.approx(
        result.baseline.mean_teacher_kl,
        abs=1e-8,
    )
    assert candidate.mean_inherited_hidden_nmse > 0.0
    assert candidate.max_recurrence_relative_residual < 1e-12
    assert len(candidate.sample_counteraction) == 2
    assert result.suffix_sensitivity == "forward-ad"
    rtn_suffix_kl = candidates["rtn"].mean_suffix_jvp_teacher_kl
    assert rtn_suffix_kl is not None
    assert math.isfinite(rtn_suffix_kl)
    assert set(result.as_dict()["suffix_jvp_ranking"]) == {"rtn", "block-hessian"}
