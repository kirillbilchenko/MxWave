"""FP32 partial precision cannot bypass historical gates or workspace ownership checks."""

from __future__ import annotations

import copy
import importlib
import struct
from pathlib import Path

import pytest
import torch


@pytest.fixture(autouse=True)
def source_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Import integration contracts without initializing CUDA."""
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))


def test_bf16_partial_rounding_loses_information_before_fp32_reduction() -> None:
    partial = torch.tensor([257., -256.], dtype=torch.float32)
    assert partial.sum().item() == 1.
    assert partial.bfloat16().float().sum().item() == 0.
    probe = importlib.import_module("fp32_split_probe")
    one, zero = struct.pack("<H", 0x3F80), struct.pack("<H", 0)
    assert probe.bf16_metrics(one, one, exact=True)["passed"]
    assert not probe.bf16_metrics(zero, one, exact=False)["passed"]


def test_raw_metrics_preserve_strict_word_gate_and_reject_nonfinite() -> None:
    probe = importlib.import_module("fp32_split_probe")
    expected = struct.pack("<1000H", *([0x3F80] * 1000))
    actual = struct.pack("<1000H", *([0x3F80] * 999 + [0x3F81]))
    assert probe.bf16_metrics(actual, expected, exact=False)["passed"]
    assert not probe.bf16_metrics(actual, expected, exact=True)["passed"]
    assert not probe.bf16_metrics(struct.pack("<H", 0x7FC0), b"\0\0", exact=False)["passed"]
    for bad in (b"", b"x", b"xxxx"):
        with pytest.raises(ValueError):
            probe.bf16_metrics(bad, b"\0\0", exact=False)


def test_fp32_workspace_includes_all_splits_and_padded_rows() -> None:
    probe = importlib.import_module("fp32_split_probe")
    assert probe.workspace_bytes(12608, 16, 8) == 6455296
    assert probe.workspace_bytes(4096, 8, 4) == 524288
    for shape in ((12576, 16, 8), (4096, 4, 4), (4096, 8, 1), (4096, 8, 16)):
        with pytest.raises(ValueError):
            probe.workspace_bytes(*shape)


def complete_report() -> dict:
    """Build a complete declared report to exercise omissions and changed decision gates."""
    probe = importlib.import_module("fp32_split_probe")
    p = probe.protocol()
    cases = []
    for layer in p["layers"]:
        for kind in p["kinds"]:
            checks = []
            for n in p["tokens"]:
                for activation in p["activation_cases"]:
                    for split in p["splits"]:
                        checks.append({"tokens": n, "activation": activation, "split": split,
                            "stock_comparison": {"finite": True, "bits_exact": True,
                                "relative_l2": 0., "different_words": 0, "passed": True,
                                "requires_bits_exact": n in (4, 8) or activation == "basis"},
                            "rebuilt_matches_original": True, "within_arm_repeats_exact": True,
                            "graph_matches_eager": True, "guards_unchanged": True,
                            "invalid_workspace_rejected": True, "graph_retained_bytes": 0})
            cases.append({"name": f"kda-{kind}-l{layer}-r0", "source_bits_exact": True,
                "decoder_bits_exact": True, "packed_buffers_unchanged": True, "checks": checks})
    return {"status": "complete", "rank": 0, "protocol": p, "cases": cases}


def test_assessor_rejects_missing_coverage_and_preserves_numeric_failure() -> None:
    probe = importlib.import_module("fp32_split_probe")
    report = complete_report()
    assert probe.assess(report)["splits"]["4"]["checks"] == 96
    missing = copy.deepcopy(report)
    missing["cases"][0]["checks"].pop()
    with pytest.raises(ValueError, match="Missing precision"):
        probe.assess(missing)
    row = next(x for x in report["cases"][0]["checks"] if x["tokens"] == 4)
    row["stock_comparison"].update(bits_exact=False, different_words=1,
                                   relative_l2=1e-6, passed=False)
    assert not probe.assess(report)["splits"][str(row["split"])]["passed"]
    row["stock_comparison"]["passed"] = True
    with pytest.raises(ValueError, match="Numerical gate altered"):
        probe.assess(report)


def test_patch_changes_only_store_precision_and_fails_closed_on_drift() -> None:
    builder = importlib.import_module("build_bf16_fp32_split")
    kernel = (b"prefix\n__global__ void BF16TripleBitmap_MM_Kernel(\n"
              b"__nv_bfloat16* Output,\nunchanged MMA loop\n"
              b"__nv_bfloat16* OutputPTR = Output + offsets;\n"
              b"store = __float2bfloat16_rn((*(smem_CFrag + i))[j]);\n"
              b"__device__ __forceinline__ void VectorizedWriteToGlobalMemory remainder")
    actual = builder.patch_kernel(kernel)
    assert b"unchanged MMA loop" in actual and b"float* OutputPTR = Output + offsets" in actual
    assert actual.endswith(b"VectorizedWriteToGlobalMemory remainder")
    with pytest.raises(ValueError, match="Ambiguous"):
        builder.patch_kernel(kernel.replace(b"__float2bfloat16_rn", b"different_rounding"))


def test_precision_controller_is_bounded_and_does_not_request_profiling_privileges() -> None:
    runner = importlib.import_module("run_fp32_split")
    command = runner.command(1)
    assert "SYS_ADMIN" not in command and "--network" in command
    assert "/src/benchmark_fp32_split.py" in command
    assert "/result/precision.json" in command and "900" in command
    for rank, profile in ((-1, False), (0, True)):
        with pytest.raises(ValueError):
            runner.command(rank, profile=profile)
