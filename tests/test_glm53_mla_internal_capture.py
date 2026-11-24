"""Actual-operation hooks must preserve calls, byte masks, scope and the combined copy budget."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))

from mla_internal_capture import MLAInternalCapture


def test_actual_probe_plan_refuses_fewer_replays_or_unbalanced_policies() -> None:
    from measure_mla_internal import validate_plan

    plan = {"version": 5, "model_arm": "control", "model_loads": 1,
        "iterations_per_probe": 5, "reference_repeats": 0, "probe_repeats_per_target": 2,
        "workspace_policies": ["tracked_only", "restore_private", "restore_private", "tracked_only"],
        "capture_sizes": [1, 4], "targets": [{"group_start": 4, "graph_step": 0}],
        "qualification_attempted": False, "stop_on_kl_failure": False,
        "previous_failed_gates_preserved": True, "admission_grace_seconds": 1.0,
        "snapshot_backing": "ssd_mmap",
        "reference_evidence": {"archive_sha256":
            "fc9e387a3f99dd64fdd427e5d4f4b2b4251c03b54d9d2be0f0318a681ef35dbb"}}
    validate_plan(plan)
    for key, changed in (
        ("iterations_per_probe", 4), ("model_arm", "candidate"),
        ("previous_failed_gates_preserved", False),
        ("workspace_policies", ["tracked_only", "restore_private"]),
        ("targets", [{"group_start": 0, "graph_step": 2}]),
    ):
        with pytest.raises(ValueError, match="Frozen"):
            validate_plan({**plan, key: changed})


def test_selected_cache_masks_invalid_slots_and_tails_as_bytes() -> None:
    trace = MLAInternalCapture(lambda: True)
    cache = torch.arange(6 * 512).remainder(251).to(torch.uint8).view(6, 1, 512)
    original = cache.clone()
    indices = torch.tensor([[2, 4, -1, 999], [1, 0, 5, 3]], dtype=torch.int32)
    trace.selected_cache("cache", cache, indices, torch.tensor([2, 3], dtype=torch.int32))
    values = {row["path"]: row["value"] for row in trace.events[0]["tensors"]}
    assert values["/valid"].tolist() == [[True, True, False, False], [True, True, True, False]]
    assert torch.equal(values["/bytes"][0, :2], original[[2, 4], 0])
    assert torch.equal(values["/bytes"][1, :3], original[[1, 0, 5], 0])
    assert not bool(values["/bytes"][~values["/valid"]].any())
    cache.zero_()
    assert bool(values["/bytes"][values["/valid"]].any())
    assert trace.metadata()["combined_copy_staging_bytes"] <= 32 * 1024**2


def test_empty_nope_metadata_and_budget_rejection(monkeypatch: pytest.MonkeyPatch) -> None:
    import mla_internal_capture

    trace = MLAInternalCapture(lambda: True)
    trace.record("rope", "operand", {"q_pe": torch.empty(4, 32, 0)})
    assert trace.events[0]["tensors"] == []
    assert trace.events[0]["parameters"]["operand_layouts"]["/q_pe"]["shape"] == [4, 32, 0]
    monkeypatch.setattr(mla_internal_capture, "MAX_STAGING_BYTES", 16)
    with pytest.raises(MemoryError, match="budget"):
        trace.selected_cache("large", torch.zeros(4, 512, dtype=torch.uint8),
                             torch.tensor([0], dtype=torch.int32))
    assert trace.staging_bytes == 0
    trace.nbytes, trace.staging_bytes = 23 * 1024**2, 1024**2
    with pytest.raises(MemoryError, match="32 MiB"):
        trace.record("must_not_allocate", "operand", torch.ones(1, dtype=torch.uint8))


def test_planned_prefix_keeps_real_slot_zero_reads_beyond_conversion_validity() -> None:
    trace = MLAInternalCapture(lambda: True)
    cache = torch.full((2, 1, 512), 19, dtype=torch.uint8)
    cache[1] = 37
    # The native conversion clamps its invalid suffix to slot0. If the planned count
    # exceeds conversion-valid count1, those additional reads must remain observable.
    indices = torch.tensor([[1, 0, 0]], dtype=torch.int32)
    trace.selected_cache("planned", cache, indices, torch.tensor([2], dtype=torch.int32))
    tensors = {row["path"]: row["value"] for row in trace.events[0]["tensors"]}
    assert tensors["/valid"].tolist() == [[True, True, False]]
    assert bool((tensors["/bytes"][0, 0] == 37).all())
    assert bool((tensors["/bytes"][0, 1] == 19).all())
    assert not bool(tensors["/bytes"][0, 2].any())


def test_private_policy_restores_before_native_call_only_when_selected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from decode_state_values_worker import DecodeStateValuesWorker
    from mla_internal_worker import MLAInternalWorker

    events = []

    class Images:
        live = 29

        def restore_before(self) -> None:
            events.append("restore")
            self.live = 17

        def before_exact(self) -> bool:
            events.append("check")
            return self.live == 17

    images = Images()

    def native(self: Any, state: dict[str, Any], *args: Any, **kwargs: Any) -> str:
        events.append(("native", images.live, args, kwargs))
        return "original-result"

    monkeypatch.setattr(DecodeStateValuesWorker, "_mxwave_decode_call", native)
    worker = object.__new__(MLAInternalWorker)
    state = {"private_workspace_images": images, "workspace_policy": "tracked_only",
             "private_before_each_call": []}
    assert worker._mxwave_decode_call(state, "graph", flag=True) == "original-result"
    assert events == ["check", ("native", 29, ("graph",), {"flag": True})]
    assert state["private_before_each_call"] == [False]
    events.clear()
    state["workspace_policy"] = "restore_private"
    assert worker._mxwave_decode_call(state, "graph", flag=True) == "original-result"
    assert events == ["restore", "check", ("native", 17, ("graph",), {"flag": True})]
    assert state["private_before_each_call"] == [False, True]


def test_native_call_chain_is_unchanged_and_hooks_restore() -> None:
    active = {"capture": False}
    calls: list[str] = []

    class Wrapper:
        def __init__(self) -> None:
            self._plan_info = [42, 99]
            self._kv_indices_buf = torch.zeros(8, dtype=torch.int32)
            self._qo_indptr_buf = torch.arange(5, dtype=torch.int32)
            self._kv_indptr_buf = torch.arange(5, dtype=torch.int32) * 2
            self._kv_len_arr_buf = torch.full((4,), 2, dtype=torch.int32)

        def run(self, q: torch.Tensor, qpe: torch.Tensor, cache: torch.Tensor,
                kpe: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            calls.append("attention")
            return q * .5

    wrapper = Wrapper()
    state = SimpleNamespace(wrapper=wrapper)

    def convert(requests: torch.Tensor, block_table: torch.Tensor, topk: torch.Tensor,
                **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append("convert")
        return topk.clone(), torch.full((4,), 2, dtype=torch.int32)

    backend = SimpleNamespace(_get_sm90_state=lambda: state,
                              triton_convert_req_index_to_global_index=convert)

    class Impl:
        dcp_world_size = 1

        def do_kv_cache_update(self, latent: torch.Tensor, rope: torch.Tensor,
                              cache: torch.Tensor, slots: torch.Tensor,
                              dtype: str, scale: torch.Tensor) -> None:
            calls.append("cache")
            cache[slots] = 17

    class Target:
        qk_rope_head_dim, kv_lora_rank, num_heads = 0, 512, 32

        def __init__(self) -> None:
            self.impl = Impl()
            self.W_UK_T = torch.randn(32, 2, 512)
            self.W_UV = torch.randn(32, 512, 2)
            self.cache = torch.zeros(8, 1, 512, dtype=torch.uint8)

        def forward(self, q: torch.Tensor, latent: torch.Tensor,
                    rope: torch.Tensor) -> torch.Tensor:
            self.impl.do_kv_cache_update(latent, rope, self.cache,
                                        torch.arange(4), "fp8_e4m3", torch.tensor(1.))
            query = torch.bmm(q.transpose(0, 1), self.W_UK_T).transpose(0, 1)
            slots, _ = backend.triton_convert_req_index_to_global_index(
                torch.arange(4, dtype=torch.int32), torch.zeros(4, 1, dtype=torch.int32),
                torch.arange(8, dtype=torch.int32).view(4, 2), return_valid_counts=True)
            wrapper._kv_indices_buf.copy_(slots.reshape(-1))
            attended = wrapper.run(query, rope, self.cache, self.cache[..., :0], ckv_scale=.5)
            output = torch.empty(32, 4, 2)
            torch.bmm(attended.transpose(0, 1), self.W_UV, out=output)
            return output.transpose(0, 1)

    target = Target()
    model = SimpleNamespace(named_modules=lambda: [
        ("language_model.model.layers.3.self_attn.attn", target)])
    q, latent, rope = torch.randn(4, 32, 2), torch.randn(4, 512), torch.empty(4, 32, 0)
    wanted = target.forward(q, latent, rope)
    original_bmm = torch.bmm
    trace = MLAInternalCapture(lambda: active["capture"])
    try:
        trace.install(model, backend)
        assert torch.equal(target.forward(q, latent, rope), wanted)
        assert not trace.events
        active["capture"] = True
        assert torch.equal(target.forward(q, latent, rope), wanted)
    finally:
        trace.restore()
    names = [event["name"] for event in trace.events]
    assert names == ["mla/input", "kv_write/input", "kv_write/before", "kv_write/after",
        "query_bmm/input", "query_bmm/output", "sparse_convert/input", "sparse_convert/output",
        "attention/input", "attention/selected_cache", "attention/output",
        "value_bmm/input", "value_bmm/output", "mla/output"]
    assert calls == ["cache", "convert", "attention"] * 3
    assert torch.bmm is original_bmm and backend.triton_convert_req_index_to_global_index is convert
    assert trace.metadata()["python_wrappers_removed"]
    assert trace.captured_plan == [42, 99]
