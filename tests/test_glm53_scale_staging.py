"""Shared staging requires a complete, exact range proof including rare exceptions."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def policy(monkeypatch: pytest.MonkeyPatch):
    """Load the CPU range policy without importing the CUDA runtime."""
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))
    return importlib.import_module("moe_scale_policy")


def test_all_aligned_buckets_preserve_palette_and_source_bytes(policy) -> None:
    for base in range(0, 256, 16):
        raw = np.arange(16, dtype=np.uint8) + base
        saved = raw.copy()
        palette = base | (base + 1) << 8 | (base + 2) << 16 | 1 << 24
        descriptor, actual_base = policy.sparse_stage_descriptor(raw, palette)
        assert actual_base == base
        assert descriptor & ((1 << 26) - 1) == palette
        assert descriptor & policy.STAGE_VERIFIED and descriptor & policy.STAGE_NIBBLES
        assert descriptor >> policy.STAGE_BASE_SHIFT & 255 == base
        assert np.array_equal(raw, saved)


def test_rare_exception_controls_policy_even_across_chunk_boundary(policy) -> None:
    raw = np.resize(np.array([119, 120, 121, 120], dtype=np.uint8), 1024**2 + 32)
    palette = 119 | 120 << 8 | 121 << 16 | 1 << 24
    assert policy.sparse_stage_descriptor(raw, palette)[1] == 112
    raw[-1] = 128
    descriptor, base = policy.sparse_stage_descriptor(raw, palette)
    assert base is None and not descriptor & policy.STAGE_NIBBLES
    assert descriptor & policy.STAGE_VERIFIED
    # A short numerical span still requires fallback when it crosses a bucket.
    raw = np.resize(np.array([127, 128], dtype=np.uint8), 16)
    assert policy.sparse_stage_descriptor(raw, palette)[1] is None


def test_unused_palette_entries_do_not_override_complete_source_range(policy) -> None:
    raw = np.full(16, 255, dtype=np.uint8)
    descriptor, base = policy.sparse_stage_descriptor(raw, 0 | 1 << 8 | 255 << 16)
    assert base == 240 and descriptor & policy.STAGE_NIBBLES


def test_invalid_or_already_extended_policy_fails_closed(policy) -> None:
    for raw in (np.zeros(16, dtype=np.int16), np.zeros((1, 16), dtype=np.uint8),
                np.zeros(0, dtype=np.uint8), np.zeros(17, dtype=np.uint8),
                np.zeros(32, dtype=np.uint8)[::2]):
        with pytest.raises(ValueError, match="U8"):
            policy.sparse_stage_descriptor(raw, 0)
    raw = np.zeros(16, dtype=np.uint8)
    for descriptor in (-1, 1 << 26, policy.STAGE_VERIFIED):
        with pytest.raises(ValueError, match="descriptor"):
            policy.sparse_stage_descriptor(raw, descriptor)


def test_review_requires_nibble_and_fallback_exactness_coverage(policy) -> None:
    import copy

    analysis = importlib.import_module("analyze_sparse_scales")
    cases = [{"case": i, "bits_exact": True, "nondefault_stream": True,
              "graph_replay_growth_bytes": 0, "stage_metadata_verified": True,
              "aligned_nibble_base": (i % 16) * 16 if i < 20 else None} for i in range(24)]
    fallback = [{"rank": rank, "tokens": m, "synthetic_scale_edit": True,
        "complete_range_forces_u8_fallback": True, "concentrated_routing": True,
        "current_codec_control_candidate_eager_stream_graph_bits_exact": True,
        "managed_locks_reset": True,
        "dynamic_eager_stream_graph_matches_stock_for_own_alignment": True,
        "arithmetic_reference": "Previous exact U8 shared-stage consumer"}
        for rank in (0, 1) for m in (1, 128)]
    report = {"candidate_nibble_stage": True, "candidate_shared_stage": True,
              "decoder": {"status": "passed", "cases": cases}, "fallback_cases": fallback}
    assert analysis.decoder_audit(report)["exact_fallback_gemm_cases"] == 4
    bad = copy.deepcopy(report)
    bad["decoder"]["cases"][0]["stage_metadata_verified"] = False
    with pytest.raises(ValueError, match="range proofs"):
        analysis.decoder_audit(bad)
    bad = copy.deepcopy(report)
    bad["fallback_cases"].pop()
    with pytest.raises(ValueError, match="both-rank"):
        analysis.decoder_audit(bad)
    bad = copy.deepcopy(report)
    bad["decoder"]["cases"][0]["aligned_nibble_base"] = 1
    with pytest.raises(ValueError, match="aligned bucket"):
        analysis.decoder_audit(bad)
    static = copy.deepcopy(report)
    static["candidate_static_nibble_stage"] = True
    with pytest.raises(ValueError, match="before launch"):
        analysis.decoder_audit(static)
    for case in static["fallback_cases"]:
        case["fallback_selected_before_launch"] = True
    assert analysis.decoder_audit(static)["exact_fallback_gemm_cases"] == 4


def test_static_dispatch_requires_verified_range_and_explicit_exact_fallback(policy) -> None:
    from dataclasses import replace

    import torch

    runtime = importlib.import_module("moe_runtime")
    native = object.__new__(runtime.NativeMoe)
    native.nibble_stage, native.static_nibble_stage = True, True
    native.shared_stage, native.encoding, native.u8_fallback = True, "sparse_palette", None
    buffer = torch.empty(0, dtype=torch.uint8)
    packed = runtime.PackedScales((1, 1, 16), buffer, buffer, buffer,
        "sparse_palette", True, True, 112)
    assert native.scale_consumer(packed) is native
    unsupported = replace(packed, aligned_nibble_base=None)
    with pytest.raises(ValueError, match="explicitly bound"):
        native.scale_consumer(unsupported)
    with pytest.raises(ValueError, match="verified complete"):
        native.scale_consumer(replace(packed, stage_metadata_verified=False))
    fallback = object.__new__(runtime.NativeMoe)
    fallback.nibble_stage, fallback.shared_stage, fallback.encoding = False, True, "sparse_palette"
    native.set_u8_fallback(fallback)
    assert native.scale_consumer(unsupported) is fallback
    assert native.scale_consumer(packed) is native
    with pytest.raises(ValueError, match="separate exact"):
        native.set_u8_fallback(native)


def test_builder_rejects_incomplete_static_stage_policy_before_reading_sources(policy) -> None:
    builder = importlib.import_module("build_moe")
    with pytest.raises(ValueError, match="requires nibble staging"):
        builder.stage(Path("absent"), Path("unused"), ("sparse",), static_nibble_stage=True)


def test_affine_native_policy_rejects_nonconsecutive_palettes_before_cuda() -> None:
    import torch

    runtime = importlib.import_module("moe_runtime")
    native = runtime.NativeMoe.__new__(runtime.NativeMoe)
    native.compressed, native.encoding, native.affine_only = True, "sparse_palette", True
    empty = torch.empty(0)
    packed = runtime.PackedScales((1, 2, 64), empty, empty, empty, "sparse_palette", False)
    with pytest.raises(ValueError, match="verified consecutive"):
        native.bind(torch.empty((1, 64)), empty, packed, None, (), empty, 1, 8, False, (64, 128, 3))
