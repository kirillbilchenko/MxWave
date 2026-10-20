"""Compact N8 staging must preserve consumed fragment addresses and fail closed on source drift."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import torch


@pytest.fixture(autouse=True)
def source_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Import integration helpers without starting CUDA."""
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))


def test_n8_consumed_addresses_are_unchanged_and_fit_compact_double_buffer() -> None:
    def address(lane: int, k: int, compact: bool) -> int:
        row = lane % 8
        if lane > 15 and not compact:
            row += 8
        column = k + (8 if lane % 16 >= 8 else 0)
        return (row * 64 * 2 + column * 2) ^ ((lane % 8) << 4)

    for k in (0, 16, 32, 48):
        for lane in range(32):
            new = address(lane, k, True)
            assert new % 16 == 0 and new + 16 <= 8 * 64 * 2
            if lane < 16:
                assert new == address(lane, k, False)
    assert 2 * 8 * 64 * 2 == 2048
    old = 4048 * 2 + 2464 * 4 + 16 * 64 * 2 * 2 + 64 * 8 * 6
    compact = old - 2048
    assert old == 25120 and compact == 23072
    allocated = ((compact + 1024 + 127) // 128) * 128
    assert allocated == 24192 and allocated <= 25600
    assert 102400 // allocated == 4


def test_api_patch_cannot_touch_n16_or_accept_ambiguous_n8() -> None:
    patcher = importlib.import_module("build_bf16_n8")
    original = (b"prefix\n        case 8:\n"
                b"TilingConfigBF16TripleBitmap<4, 1, 1, 1>::TILE_N * TILE_K\n"
                b"        case 16:\nunchanged\n")
    updated = patcher.patch_api(original)
    assert updated.startswith(b"prefix\n") and updated.endswith(b"        case 16:\nunchanged\n")
    assert b"::TILE_N2 * TILE_K" in updated
    with pytest.raises(ValueError, match="Ambiguous"):
        patcher.patch_api(original.replace(b"::TILE_N * TILE_K", b"different"))
    with pytest.raises(ValueError, match="Ambiguous"):
        patcher.replace_once(b"same same", b"same", b"modified")


def test_candidate_gate_requires_exact_bits_even_for_small_rounding_changes() -> None:
    benchmark = importlib.import_module("benchmark_bf16_n8")
    expected = torch.ones((4, 4096), dtype=torch.bfloat16)
    actual = expected.clone()
    assert benchmark.exact(actual, expected)
    actual[0, 0] = 1.0078125
    with pytest.raises(RuntimeError, match="changed output bits"):
        benchmark.exact(actual, expected)
    assert benchmark.exact(torch.empty((0, 4096)), torch.empty((0, 4096)))


def test_confirmation_uses_separate_owned_container_without_a_profiler(tmp_path: Path) -> None:
    runner = importlib.import_module("run_bf16_kernels")
    argv = runner.command("native", tmp_path, profile=False, n8=True, confirmation=True)
    assert "mxwave-glm53-full-native-aligned-r1" in argv
    assert "/work/benchmark_bf16_n8.py" in argv and "--confirmation" in argv
    assert "/result/n8-confirm.json" in argv
    assert "/ncu/ncu" not in argv and "SYS_ADMIN" not in argv
    with pytest.raises(ValueError, match="unprofiled"):
        runner.command("native", tmp_path, profile=True, n8=True)


def test_tail_first_is_a_bijection_and_other_geometries_are_unchanged() -> None:
    patcher = importlib.import_module("build_bf16_n8")
    for tokens in (1, 4, 8):
        rows = [patcher.input_row_tile(y, 12608, 4096, tokens, 1) for y in range(197)]
        assert rows[:5] == list(range(192, 197))
        assert sorted(rows) == list(range(197))
    for geometry in ((12608, 4096, 16, 1), (12608, 4096, 4, 2),
                     (4096, 12608, 4, 1), (12672, 4096, 4, 1)):
        assert [patcher.input_row_tile(y, *geometry) for y in range(197)] == list(range(197))
    before = (b"__global__ void BF16TripleBitmap_MM_Kernel(\r\n"
              b"    const int y           = blockIdx.y % (M_Global / TilingConfig::TILE_M);\r\n"
              b"    independent K loop and stores\r\n"
              b"__device__ __forceinline__ void VectorizedWriteToGlobalMemory remainder")
    updated = patcher.patch_schedule(before)
    assert b"&& Split_K == 1)" in updated and b"TilingConfig::N8" in updated
    assert updated.endswith(b"    independent K loop and stores\r\n"
                            b"__device__ __forceinline__ void VectorizedWriteToGlobalMemory remainder")
    with pytest.raises(ValueError, match="Ambiguous"):
        patcher.patch_schedule(before.replace(b"const int y", b"changed int y"))


def test_variant_session_rejects_unknown_or_non_n8_candidates(tmp_path: Path) -> None:
    runner = importlib.import_module("run_bf16_kernels")
    variants = ("narrow", "compact_x4", "tailfirst", "compact_tailfirst")
    argv = runner.command("variants", tmp_path, profile=False, n8=True, candidates=variants)
    assert "/work/benchmark_bf16_n8_matrix.py" in argv
    assert argv[argv.index("--candidates") + 1:] == list(variants)
    for invalid in (("control",), ("tailfirst", "tailfirst"), ("unknown",)):
        with pytest.raises(ValueError, match="recognized"):
            runner.command("variants", tmp_path, profile=False, n8=True, candidates=invalid)
    with pytest.raises(ValueError, match="recognized"):
        runner.command("variants", tmp_path, profile=False, candidates=variants)


def test_offline_review_uses_raw_samples_and_rejects_invalid_timings() -> None:
    analyzer = importlib.import_module("analyze_bf16_n8")
    records = [{"reference": ref, "block": block, "mode": mode,
                "order": [ref, "compact"] if block % 2 == 0 else ["compact", ref],
                ref + "_samples_us": [100.] * 12, "compact_samples_us": [110.] * 12,
                "ratio": .5, ref + "_median_us": 200., "compact_median_us": 100.}
               for ref in ("original", "control") for block in range(8)
               for mode in ("warm", "evicted")]
    result = analyzer.timing_summaries(records)
    assert len(result) == 4
    assert all(s["median_paired_ratio"] == 1.1 for s in result)
    records[0]["compact_samples_us"][0] = float("nan")
    with pytest.raises(ValueError, match="finite positive"):
        analyzer.timing_summaries(records)


def test_sparse_session_has_a_separate_confirmation_and_no_profiler(tmp_path: Path) -> None:
    runner = importlib.import_module("run_bf16_kernels")
    argv = runner.command("sparse", tmp_path, profile=False, sparse=True, confirmation=True)
    assert "/work/benchmark_sparse_scales.py" in argv and "--confirmation" in argv
    assert "/result/sparse-confirm.json" in argv and "--fixture" in argv
    assert "/ncu/ncu" not in argv and "SYS_ADMIN" not in argv
    for flags in ({"profile": True}, {"profile": False, "n8": True},
                  {"profile": False, "cache": True}):
        with pytest.raises(ValueError):
            runner.command("sparse", tmp_path, sparse=True, **flags)


def test_sparse_launch_diagnostics_cannot_masquerade_as_confirmation(tmp_path: Path) -> None:
    runner = importlib.import_module("run_bf16_kernels")
    argv = runner.command("debug", tmp_path, profile=False, sparse=True, sparse_diagnostic=True)
    assert "CUDA_LAUNCH_BLOCKING=1" in argv and "--diagnostic" in argv
    assert "--confirmation" not in argv and "/ncu/ncu" not in argv
    with pytest.raises(ValueError, match="single-process"):
        runner.command("debug", tmp_path, profile=False, sparse_diagnostic=True)
    with pytest.raises(ValueError, match="single-process"):
        runner.command("debug", tmp_path, profile=False, sparse=True,
                       sparse_diagnostic=True, confirmation=True)


def test_sparse_review_recomputes_both_orders_from_finite_samples() -> None:
    analyzer = importlib.import_module("analyze_sparse_scales")
    records = [{"block": block, "mode": mode,
                "order": ["control", "candidate"] if block % 2 == 0 else ["candidate", "control"],
                "control_samples_us": [100.] * 12, "candidate_samples_us": [103.] * 12,
                "ratio": .5, "control_median_us": 200., "candidate_median_us": 100.}
               for block in range(8) for mode in ("warm", "evicted")]
    result = analyzer.latency(records)
    assert len(result) == 2 and all(r["median_paired_ratio"] == 1.03 for r in result)
    assert all(set(r["order_ratios"].values()) == {1.03} for r in result)
    records[0]["candidate_samples_us"][0] = float("nan")
    with pytest.raises(ValueError, match="finite positive"):
        analyzer.latency(records)


def test_sparse_counters_use_ncu_and_keep_arm_exports_distinct(tmp_path: Path) -> None:
    runner = importlib.import_module("run_bf16_kernels")
    control = runner.command("counters", tmp_path, profile=False, sparse=True, sparse_ncu=True)
    candidate = runner.command("counters", tmp_path, profile=False, sparse=True,
                               sparse_ncu=True, confirmation=True)
    assert "/work/profile_sparse_scales.py" in control and "/ncu/ncu" in control
    assert "SYS_ADMIN" in control and "--profile-from-start" in control
    assert "/result/sparse" in control and "/result/sparse-confirm" in candidate
    assert "/result/sparse-ncu.csv" in control and "/result/sparse-confirm-ncu.csv" in candidate
    with pytest.raises(ValueError, match="separate counter"):
        runner.command("counters", tmp_path, profile=False, sparse_ncu=True)


def test_affine_native_policy_rejects_nonconsecutive_palettes_before_cuda() -> None:
    runtime = importlib.import_module("moe_runtime")
    native = runtime.NativeMoe.__new__(runtime.NativeMoe)
    native.compressed, native.encoding, native.affine_only = True, "sparse_palette", True
    empty = torch.empty(0)
    packed = runtime.PackedScales((1, 2, 64), empty, empty, empty, "sparse_palette", False)
    with pytest.raises(ValueError, match="verified consecutive"):
        native.bind(torch.empty((1, 64)), empty, packed, None, (), empty, 1, 8, False, (64, 128, 3))
