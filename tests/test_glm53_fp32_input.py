"""The exploratory input route preserves scope, scratch aliasing and capture ownership."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import torch


@pytest.fixture(autouse=True)
def source_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Load isolated experimental modules without starting CUDA."""
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))


def test_dispatch_cannot_expand_to_output_or_untested_rows() -> None:
    p = importlib.import_module("fp32_input_policy")
    geometries = [(12608, 4096, 12576), (4096, 4096, 4096), (12608, 4096, 12608),
                  (12576, 4096, 12576), (12608, 4095, 12576)]
    for n in range(257):
        for geometry in geometries:
            assert p.selected(n, *geometry) == (n in (1, 4, 8, 16) and geometry == p.INPUT_GEOMETRY)
    assert not p.policy()["parent_exact_word_screen_passed"]
    assert not p.policy()["model_quality_or_serving_speed_qualified"]


def test_fp32_view_reuses_bytes_and_stream_slots_do_not_alias() -> None:
    module = importlib.import_module("fp32_input_runtime")
    p = importlib.import_module("fp32_input_policy")
    words = p.scratch_bytes(16) // 2 + 128
    pool = torch.full((2, words), 13., dtype=torch.bfloat16)
    first, second = (module.workspace(pool, slot, 16) for slot in (0, 1))
    assert first.dtype == second.dtype == torch.float32
    assert first.numel() * 4 == 1613824
    assert first.data_ptr() == pool[0].data_ptr()
    assert second.data_ptr() == pool[1].data_ptr()
    first.fill_(17.)
    assert torch.equal(pool[1], torch.full_like(pool[1], 13.))
    assert torch.equal(pool[0, -128:], torch.full_like(pool[0, -128:], 13.))
    assert module.workspace(pool, 0, 1).numel() * 4 == 806912
    assert p.policy()["additional_resident_scratch_bytes"] == 0


def test_workspace_rejects_wrong_type_capacity_alignment_and_untested_rows() -> None:
    module = importlib.import_module("fp32_input_runtime")
    p = importlib.import_module("fp32_input_policy")
    words = p.scratch_bytes(16) // 2
    pool = torch.empty((2, words), dtype=torch.bfloat16)
    for bad, slot, n in [(pool.float(), 0, 16), (pool[:, :-1], 0, 16), (pool, 2, 16),
                         (pool, 0, 2), (pool[:, ::2], 0, 1), (pool.flatten(), 0, 1)]:
        with pytest.raises(ValueError):
            module.workspace(bad, slot, n)
    arena = torch.empty(2 * words + 1, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="alignment"):
        module.workspace(arena[1:].view(2, words), 0, 16)


def test_priming_inside_capture_fails_before_loading_native_library(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("fp32_input_runtime")
    monkeypatch.setattr(module, "_PRIMED", False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="eager warmup"):
        module.prime()


def test_opaque_schema_declares_pool_mutation_and_fake_keeps_leading_dimensions() -> None:
    module = importlib.import_module("fp32_input_runtime")
    from torch._subclasses.fake_tensor import FakeTensorMode

    schema = torch.ops.mxwave_glm53.kda_input_fp32.default._schema
    pool_argument = next(x for x in schema.arguments if x.name == "pool")
    assert pool_argument.alias_info is not None and pool_argument.alias_info.is_write
    with FakeTensorMode():
        for shape in ((4096,), (0, 4096), (2, 4, 4096), (16, 4096)):
            x = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
            pool = torch.empty((2, 12608 * 4096), dtype=torch.bfloat16, device="cuda")
            buffers = [torch.empty((32,), dtype=torch.uint8, device="cuda") for _ in range(7)]
            result = module.input_linear(x, buffers, pool, 12608, 4096, 1, 1, 0, 12576)
            assert result.shape == (*shape[:-1], 12576) and result.dtype == torch.bfloat16


def report_fixture() -> dict:
    """Represent complete wrapper evidence so missing fallbacks or false graph claims fail."""
    probe = importlib.import_module("fp32_input_probe")
    p = probe.protocol()
    cases = []
    graph = {"graph_matches_eager": True, "graph_retained_bytes": 0,
             "capture_extra_bytes": 1024, "graph_output_sha256": "a"}
    for layer in p["layers"]:
        for kind in p["kinds"]:
            entry = {"name": f"kda-{kind}-l{layer}-r0", "decoded_weight_bits_exact": True,
                "packed_buffers_unchanged": True, "outer_pool_guards_unchanged": True,
                "FP32_workspace_suffix_guards_unchanged": True, "fallback": [], "compiled": [],
                "known": [{"tokens": n, "activation": a, "matches_parent": True,
                    "output_sha256": "a", "parent_sha256": "a", **graph}
                    for n in p["tokens"] for a in (p["activation_cases"] if kind == "input" else ["787"])]}
            if kind == "input":
                entry["fallback"] = [{"tokens": n, "matches_qualified": True,
                    "actual": {"sha256": "a"}, "qualified": {"sha256": "a"}, **graph}
                    for n in p["fallback_tokens"]]
                entry["compiled"] = [{"tokens": n, "matches_parent": True,
                    "output_sha256": "a", "parent_sha256": "a", **graph}
                    for n in (p["compiled_tokens"] if layer == 0 else [])]
                entry["compiled_chain"] = {**graph, "physical_rows": [8, 17, 8],
                    "expected_output_sha256": ["a", "b", "c"], "output_sha256": ["a", "b", "c"],
                    "graph_output_sha256": ["a", "b", "c"]} if layer == 0 else None
                entry["stream_stress"] = {"replays_per_graph": 32, "owned_slots": [0, 1],
                    "captured_streams": 2, "outputs_match_parent": True,
                    "parent_sha256": ["a", "b"], "output_sha256": ["a", "b"],
                    "third_stream_rejected": True, "workspace_suffix_guards_unchanged": True,
                    "replay_retained_bytes": 0}
            cases.append(entry)
    return {"status": "complete", "rank": 0, "protocol": p, "cases": cases,
            "CUDA_peak_bytes": 512 * 1024**2}


def test_missing_fallback_coverage_or_compiled_pool_clone_rejected() -> None:
    probe = importlib.import_module("fp32_input_probe")
    report = report_fixture()
    result = probe.assess(report)
    assert result["known_parent_comparisons"] == 60 and result["fallback_comparisons"] == 33
    assert result["compiled_shapes"] == 2 and result["two_stream_stress_fixtures"] == 3
    assert not result["parent_exact_word_screen_passed"]
    assert not result["model_quality_or_serving_speed_qualified"]
    report["cases"][0]["fallback"].pop()
    with pytest.raises(ValueError, match="fallback coverage"):
        probe.assess(report)
    report = report_fixture()
    report["cases"][0]["compiled"][0]["capture_extra_bytes"] = 206569472
    with pytest.raises(ValueError, match="duplicates scratch"):
        probe.assess(report)


def test_same_stream_slot_or_forged_parent_equality_rejected() -> None:
    probe = importlib.import_module("fp32_input_probe")
    for key, value in (("owned_slots", [0, 0]), ("third_stream_rejected", False),
                       ("output_sha256", ["a", "a"]), ("replay_retained_bytes", 4)):
        report = report_fixture()
        report["cases"][0]["stream_stress"][key] = value
        with pytest.raises(ValueError, match="scratch"):
            probe.assess(report)


def test_component_controller_is_bounded_without_model_or_profiler() -> None:
    runner = importlib.import_module("run_fp32_input")
    command = runner.command(1)
    assert command[command.index("--network") + 1] == "none" and "SYS_ADMIN" not in command
    assert "/src/benchmark_fp32_input.py" in command and "/result/integration.json" in command
    assert "MXWAVE_GLM53_KDA_INPUT_FP32_SPLIT2=1" in command
    assert "12g" in command and "900" in command
    with pytest.raises(ValueError):
        runner.command(0, profile=True)
