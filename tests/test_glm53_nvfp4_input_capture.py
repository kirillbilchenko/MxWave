"""Validate immutable prepared IDs and graph outputs without changing dispatch."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).parents[1] / "integrations/glm53_nvfp4"
previous = sys.modules.get("repeat_capture")
for name in ("repeat_capture", "input_capture"):
    spec = importlib.util.spec_from_file_location(f"nv_{name}_test", ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if name == "repeat_capture":
        sys.modules[name] = module
if previous is None:
    sys.modules.pop("repeat_capture")
else:
    sys.modules["repeat_capture"] = previous


def test_input_clones_survive_buffer_reuse_and_graph_is_called_once(tmp_path):
    ids = torch.tensor([11, 12, 13])
    calls = []
    batch = SimpleNamespace(input_ids=ids, num_tokens=3, num_tokens_after_padding=3,
                            num_reqs=1, has_prefill=False)
    state = SimpleNamespace(get_mm_embeddings=lambda: ids.float().reshape(3, 1),
                            prepare_inputs=dict)

    def graph():
        calls.append(True)
        return ids.float().reshape(3, 1)

    runner = SimpleNamespace(prepare_inputs=lambda: batch, model_state=state,
                             cudagraph_manager=SimpleNamespace(run_fullgraph=graph),
                             model=SimpleNamespace(compute_logits=lambda x: x * 2))
    capture = module.InputCapture(runner)
    capture.arm("inputs")
    assert runner.prepare_inputs() is batch
    result = runner.cudagraph_manager.run_fullgraph()
    runner.model.compute_logits(result)
    ids.zero_()
    report = capture.finish(tmp_path / "capture")
    event = next(e for e in report["events"] if e["label"] == "prepared/input_ids")
    expected = torch.tensor([11, 12, 13]).view(torch.uint8).numpy().tobytes()
    assert (tmp_path / "capture" / event["file"]).read_bytes() == expected
    assert calls == [True] and report["batches"][0]["step"] == 0
    assert not capture.tensors and not report["CPU_sync_during_forward"]
