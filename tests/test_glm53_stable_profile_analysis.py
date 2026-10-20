"""Guard full-forward coverage and time unions used in the saved profile review."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))
from analyze_stable_profiles import complete_forwards, cpu_forwards, group


def test_nested_substream_does_not_duplicate_forward_or_invent_idle_time() -> None:
    ranges = [
        {"name": "execute_4_context_0_generation_4", "ts": 0., "dur": 100., "tid": 1},
        {"name": "execute_4_context_0_generation_4", "ts": 5., "dur": 90., "tid": 2},
    ]
    kernels = [
        {"name": "marlin_moe_wna16::Marlin", "ts": 10., "dur": 20., "tid": 1}
        for _ in range(84)
    ] + [
        {"name": "marlin::Marlin", "ts": 30., "dur": 60., "tid": 2},
        {"name": "_canonicalize", "ts": 20., "dur": 5., "tid": 1},
    ]
    rows = complete_forwards(ranges, kernels)
    assert len(rows) == 1
    assert rows[0]["kernel_union_all_streams_us"] == 80.
    assert rows[0]["uncovered_all_streams_us"] == 20.
    assert rows[0]["kernel_groups"]["stable_expert_order"]["calls"] == 1


def test_partial_expert_capture_cannot_be_called_a_full_model_forward() -> None:
    ranges = [{"name": "execute_1_context_0_generation_1", "ts": 0., "dur": 10., "tid": 1}]
    kernels = [{"name": "marlin_moe_wna16::Marlin", "ts": 0., "dur": 1., "tid": 1}]
    assert complete_forwards(ranges, kernels) == []
    assert group("cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm>") == (
        "bf16_dense_linear"
    )


def test_cpu_projection_coverage_excludes_nested_other_thread_and_partial_ranges() -> None:
    ranges = [{"name": "execute_4_context_0_generation_4", "ts": 0., "dur": 1000., "tid": 1},
              {"name": "execute_4_context_0_generation_4", "ts": 0., "dur": 50., "tid": 1}]
    ops = [{"name": "mxwave_glm53::kda_linear", "ts": float(i * 10), "dur": 5., "tid": 1}
           for i in range(68)]
    ops += [{"name": "mxwave_zipserv::linear", "ts": float(i * 10 + 1), "dur": 2., "tid": 1}
            for i in range(68)]
    ops.append({"name": "mxwave_glm53::kda_linear", "ts": 1., "dur": 5., "tid": 2})
    result = cpu_forwards(ranges, ops)
    assert len(result) == 1
    assert result[0]["operations"]["mxwave_glm53::kda_linear"] == {"calls": 68, "sum_us": 340.}
    assert result[0]["operations"]["mxwave_zipserv::linear"] == {"calls": 68, "sum_us": 136.}
    assert result[0]["cpu_forward_span_us"] == 1000.
