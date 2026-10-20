"""Diagnostic copies must preserve operations and be bounded before graph replay."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from integrations.glm53_mxfp4 import drafter_component_probe as probe


def test_eight_row_scope_preserves_inputs_before_inplace_mutation() -> None:
    trace = probe.ComponentTrace(capturing=lambda: True)
    weight = torch.ones(512, 512)

    def pre(x, unused_weight):
        return x.add_(1)

    operation = trace.wrapper("layer0.hc_pre", first_args=1)(pre)
    layer = trace.wrapper("model.layers.0", layer=0)(
        lambda positions, hidden: operation(hidden, weight))
    x = torch.zeros(8, 16)
    assert layer(torch.arange(8), x) is x
    records = {event["name"]: event for event in trace.events}
    saved = records["layer0.hc_pre#0/input"]["tensors"]
    assert len(saved) == 1 and saved[0]["shape"] == [8, 16]
    assert torch.count_nonzero(saved[0]["value"]) == 0
    assert trace.active_layer is None
    assert trace.nbytes < weight.numel() * weight.element_size()


@pytest.mark.parametrize("rows,capturing", [(4, True), (16, True), (8, False)])
def test_scope_excludes_other_shapes_and_eager_calls(rows: int, capturing: bool) -> None:
    trace = probe.ComponentTrace(capturing=lambda: capturing)
    layer = trace.wrapper("model.layers.0", layer=0)(lambda p, h: h + 1)
    result = layer(torch.arange(rows), torch.zeros(rows, 16))
    assert torch.all(result == 1)
    assert not trace.events


def test_budget_checked_before_any_new_tensor_copy() -> None:
    trace = probe.ComponentTrace()
    trace.max_bytes = 1
    with pytest.raises(MemoryError, match="1-byte"):
        trace.record("large", "operand", torch.zeros(8, 16))
    assert trace.nbytes == 0 and trace.events == []


def test_topk_keeps_eight_live_rows_only() -> None:
    trace = probe.ComponentTrace()
    buffer = torch.arange(2048 * 128).reshape(2048, 128)
    trace._topk_storages.add(buffer.untyped_storage().data_ptr())
    trace.record("indices", "result", buffer)
    saved = trace.events[0]["tensors"][0]["value"]
    assert saved.shape == (8, 128) and torch.equal(saved, buffer[:8])


def test_capture_failure_removes_all_instance_wrappers(monkeypatch: pytest.MonkeyPatch) -> None:
    owner = SimpleNamespace(forward=lambda: "original")
    original = owner.forward

    class FailedTrace(probe.ComponentTrace):
        def install(self, model, ops):
            self.replace(owner, "forward", lambda f: lambda: "wrapped")

    class Manager:
        def _build_forward_fn_factory(self):
            return None

        def capture(self, model):
            assert owner.forward() == "wrapped"
            raise RuntimeError("capture failed")

    monkeypatch.setattr(probe, "ComponentTrace", FailedTrace)
    probe.patch_manager(Manager, None)
    with pytest.raises(RuntimeError, match="capture failed"):
        Manager().capture(None)
    assert owner.forward is original


def descriptor(requests: int, uniform: int = 8) -> SimpleNamespace:
    return SimpleNamespace(cg_mode="FULL", num_tokens=8, num_reqs=requests,
                           uniform_token_count=uniform, num_active_loras=0)


def test_dynamic_draft_graphs_with_same_token_count_do_not_mix() -> None:
    trace = probe.ComponentTrace()
    observed = []

    def build():
        def factory(desc, warmup):
            return lambda mode: observed.append(trace.selected_forward)
        return factory

    factory = trace.forward_factory(build)()
    for desc in (descriptor(2, 4), descriptor(1), descriptor(2, 4)):
        for warmup in (True, False):
            factory(desc, warmup)(None)
            assert not trace.selected_forward
    assert observed == [False, False, False, True, False, False]


def test_selected_scope_restored_after_failed_forward() -> None:
    trace = probe.ComponentTrace()

    def build():
        def factory(desc, warmup):
            def forward(mode):
                assert trace.selected_forward
                raise RuntimeError("native forward failed")
            return forward
        return factory

    with pytest.raises(RuntimeError, match="native forward failed"):
        trace.forward_factory(build)()(descriptor(1), False)(None)
    assert not trace.selected_forward
