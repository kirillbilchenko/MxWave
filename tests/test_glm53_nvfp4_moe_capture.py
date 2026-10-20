"""Validate operand ownership and replay output handling without a GPU runtime."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).parents[1] / "integrations/glm53_nvfp4"
previous_repeat = sys.modules.get("repeat_capture")
for name in ("repeat_capture", "moe_capture"):
    spec = importlib.util.spec_from_file_location(f"nv_{name}_test", ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if name == "repeat_capture":
        sys.modules[name] = module
if previous_repeat is None:
    sys.modules.pop("repeat_capture")
else:
    sys.modules["repeat_capture"] = previous_repeat


def probe():
    capture = module.MoeCapture.__new__(module.MoeCapture)
    capture.active = True
    capture.enabled = False
    capture.current_layer = 3
    capture.cases, capture.events, capture.tensors = [], [], []
    capture.counts = {}
    capture.replay_bytes = capture.used = 0
    capture.maximum_bytes = 1024**2
    capture.label = "cpu-test"
    return capture


def test_replay_clones_mutable_inputs_and_preserves_original_expected_output(tmp_path):
    capture = probe()

    def reduce(x, trunc_size=None):
        x.add_(1)
        return x

    owner = SimpleNamespace(reduce=reduce)
    capture.observe(owner, "reduce", "final-reduce", layer=3)
    x = torch.ones(3)
    output = owner.reduce(x)
    assert torch.equal(output, torch.full((3,), 2.0))
    x.zero_()
    capture.active = False
    report = capture.replay(tmp_path, 3)
    assert all(row[0]["bits_exact"] for row in report["cases"][0]["repeats"])
    assert torch.equal(capture.cases[0]["args"][0], torch.ones(3))


def test_inplace_sum_returning_none_uses_output_buffer(tmp_path):
    capture = probe()

    def sum_into(x, out):
        torch.sum(x, dim=0, out=out)

    owner = SimpleNamespace(sum_into=sum_into)
    capture.observe(owner, "sum_into", "sum", positional=(0, 2, 3), output_position=1)
    output = torch.empty(3)
    assert owner.sum_into(torch.ones(2, 3), output) is None
    output.fill_(42)
    capture.active = False
    report = capture.replay(tmp_path, 3)
    assert all(row[0]["bits_exact"] for row in report["cases"][0]["repeats"])


def test_inactive_probe_does_not_capture_or_change_call():
    capture = probe()
    capture.active = False
    owner = SimpleNamespace(call=lambda x: x)
    capture.observe(owner, "call", "routing", layer=3)
    x = torch.ones(2)
    assert owner.call(x) is x
    assert not capture.cases and not capture.events
