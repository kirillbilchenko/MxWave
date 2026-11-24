"""Checks for trace streaming and exact full-vocabulary comparison evidence."""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import io
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


@pytest.fixture
def analysis() -> ModuleType:
    """Load the optional offline report script without any CUDA runtime imports."""
    path = Path(__file__).parents[1] / "integrations/glm53_mxfp4/compare_full_model.py"
    spec = importlib.util.spec_from_file_location("glm53_analysis_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stream_trace_handles_chunk_boundary_unicode_and_nested_arrays(analysis: ModuleType) -> None:
    events = [{"name": "Дані" * 20000, "args": {"shape": [[1, 2], [3, 4]]}},
              {"name": "kernel", "dur": 123}]
    stream = io.StringIO(json.dumps({"schemaVersion": 1, "traceEvents": events,
                                    "after": "root metadata"}, ensure_ascii=False))
    assert list(analysis.trace_events(stream)) == events


def test_truncated_trace_fails_without_partial_success(analysis: ModuleType) -> None:
    with pytest.raises(ValueError, match="Truncated"):
        list(analysis.trace_events(io.StringIO('{"traceEvents":[{"name":"partial')))


def test_profile_does_not_sum_overlapping_kernels_as_wall_time(
    analysis: ModuleType, tmp_path: Path,
) -> None:
    path = tmp_path / "trace.json"
    path.write_text(json.dumps({"traceEvents": [
        {"name": "marlin_kernel", "cat": "kernel", "ph": "X", "ts": 100, "dur": 20,
         "args": {"device": 0}},
        {"name": "nccl_kernel", "cat": "kernel", "ph": "X", "ts": 110, "dur": 30,
         "args": {"device": 0}}]}))
    result = analysis.profile(path)
    assert result["kernel_sum_duration_us"] == 50
    assert result["device_spans"][0]["span_first_last_kernel_us"] == 40
    assert result["device_spans"][0]["kernel_union_us"] == 40


def test_quality_normalizes_probabilities_and_rejects_tampered_evidence(
    analysis: ModuleType, tmp_path: Path,
) -> None:
    def arm(label: str, values: list[float]) -> dict[str, object]:
        path = tmp_path / (label + ".f32")
        path.write_bytes(np.array(values, dtype="<f4").tobytes())
        return {"vocabulary": 3, "mean_nll": 1.25, "contexts": [
            {"index": 0, "next_logprobs_file": path.name,
             "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
             "next_token_ids": [0], "nll": 1.25}]}

    left = arm("left", [-1, -2, -3])
    right = arm("right", [-1, -2, -3])
    result = analysis.quality(left, right, tmp_path)
    assert result["all_logprobs_bits_exact"] and result["mean_kl"] == 0
    shifted = arm("shifted", [0, -1, -2])
    assert analysis.quality(left, shifted, tmp_path)["mean_kl"] < 1e-14
    changed = arm("changed", [-3, -2, -1])
    assert analysis.quality(left, changed, tmp_path)["mean_kl"] > 1
    (tmp_path / "right.f32").write_bytes(b"invalid")
    with pytest.raises(ValueError, match="changed"):
        analysis.quality(left, right, tmp_path)


def test_unstable_control_never_widens_candidate_quality_tolerance(analysis: ModuleType) -> None:
    repeat = {"mean_kl": .0575, "mean_nll_difference": -.002, "all_greedy_tokens_exact": False}
    candidate = {"mean_kl": .0575, "mean_nll_difference": .0005,
                 "all_greedy_tokens_exact": False}
    gate = analysis.quality_gate(repeat, candidate)
    assert not gate["passed"] and not gate["raw_repeat_stable"]
    assert gate["status"] == "unqualified_unstable_control"
    assert gate["mean_kl_limit"] is None


def test_stable_quality_control_keeps_tight_candidate_gate(analysis: ModuleType) -> None:
    repeat = {"mean_kl": 1e-9, "mean_nll_difference": 1e-7, "all_greedy_tokens_exact": True}
    candidate = {"mean_kl": 1e-8, "mean_nll_difference": 1e-6, "all_greedy_tokens_exact": True}
    gate = analysis.quality_gate(repeat, candidate)
    assert gate["passed"] and gate["mean_kl_limit"] == 1e-7
    candidate["mean_kl"] = 1e-3
    assert not analysis.quality_gate(repeat, candidate)["passed"]


@pytest.fixture
def profile_analysis(analysis: ModuleType, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load offline attribution with its trace reader, without importing vLLM."""
    monkeypatch.setitem(sys.modules, "compare_full_model", analysis)
    path = Path(__file__).parents[1] / "integrations/glm53_mxfp4/analyze_full_profiles.py"
    spec = importlib.util.spec_from_file_location("glm53_profile_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gpu_attribution_requires_same_stream_and_complete_enclosure(
    profile_analysis: ModuleType, tmp_path: Path,
) -> None:
    name = "execute_1_context_0_generation_1(sq1sk257)"
    events = [
        {"cat": "gpu_user_annotation", "name": name, "tid": stream, "ts": 100, "dur": 100}
        for stream in (17, 21)
    ] + [
        {"cat": "kernel", "name": name, "ts": start, "dur": duration,
         "tid": stream, "args": {"stream": stream}}
        for name, stream, start, duration in [
            ("mxwave_grouped_aligned::Marlin", 17, 110, 10),
            ("marlin::Marlin", 21, 115, 12),
            ("nccl_allreduce", 22, 120, 11),
            ("late_kernel", 17, 195, 15),
        ]
    ]
    path = tmp_path / "trace.json.gz"
    with gzip.open(path, "wt") as stream:
        json.dump({"traceEvents": events}, stream)
    result = profile_analysis.gpu_steps(path)
    assert result["unmatched_kernels"] == 2
    rows = {row["stream"]: row for row in result["steps"]}
    assert rows[17]["kernel_us"] == {"routed_expert_gemm": 10}
    assert rows[21]["kernel_us"] == {"generic_marlin_linear": 12}
    assert sum(result["kernel_groups_us"].values()) == 48


def test_jit_attribution_uses_embedded_compile_time_not_buffered_docker_time(
    profile_analysis: ModuleType,
) -> None:
    report = {"cases": [{"batch": 1, "prompt_tokens": 256, "trials": [
        {"wall_seconds": 8, "aggregate_tokens_per_second": 8,
         "requests": [{"ttft_seconds": 1}]}]}]}
    logs = {"completion_access_timestamps": [f"2026-10-15T00:00:0{i}Z" for i in range(3)],
            "tilelang_events": [
                "2026-10-15T00:20:00Z 2026-10-15 00:00:04 TileLang begins kernel `mhc_test`",
                "2026-10-15T00:20:00Z 2026-10-15 00:00:09 TileLang completes kernel `mhc_test`",
            ]}
    row = profile_analysis.trial_compiles(report, logs)[0]
    assert len(row["tilelang_compiles"]) == 1
    assert row["tilelang_compiles"][0]["seconds_at_log_precision"] == 5
