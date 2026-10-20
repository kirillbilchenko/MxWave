"""Contracts for passive capture without changing model outputs or tensor contents."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.numpy import load_file

SOURCE = Path(__file__).parents[1] / "integrations/glm53_mxfp4"
spec = importlib.util.spec_from_file_location(
    "drafter_context_probe", SOURCE / "drafter_context_probe.py"
)
assert spec is not None and spec.loader is not None
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_cpu_capture_is_independent_and_preserves_bf16_bits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(SOURCE))
    tensor = torch.tensor([0, -32768, 1, 16256], dtype=torch.int16).view(torch.bfloat16)
    snapshot = probe.cpu_copy(tensor)
    tensor.fill_(4)
    path = tmp_path / "frame.safetensors"
    probe.retain_frame(path, {"values": snapshot}, {})
    assert load_file(path)["values"].tolist() == [0, -32768, 1, 16256]
    assert json.loads(path.with_suffix(".json").read_bytes())["tensor_dtypes"] == {
        "values": "torch.bfloat16"
    }
    assert path.stat().st_mode & 0o777 == 0o644


def test_passive_capture_returns_original_outputs_and_joins_real_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(SOURCE))
    (tmp_path / "probe-marker.json").write_text('{"active":true,"case":"code_32"}')
    logits = torch.arange(64, dtype=torch.float32).reshape(1, 64)
    output = SimpleNamespace(sampled_token_ids=torch.tensor([[63]]))
    decision = (output, torch.tensor([1]), torch.tensor([0]))
    proposal = torch.arange(7).reshape(1, 7)
    graph = {
        "hidden": torch.zeros(7, 4), "ids": torch.arange(112).reshape(7, 16),
        "values": torch.zeros(7, 16), "scores": torch.zeros(1, 7, 16, 16),
    }
    manager = SimpleNamespace(run_fullgraph=lambda descriptor: None)
    draft = SimpleNamespace(
        query_cudagraph_manager=manager,
        _mxwave_head_taps={"graphs": {(1, 8): graph}},
        draft_tokens=proposal, _selector_scores=torch.zeros(1, 7, 16),
        sample_pos=torch.arange(2, 9),
        input_buffers=SimpleNamespace(input_ids=torch.zeros(8), positions=torch.arange(1, 9)),
        hidden_states=torch.zeros(1, 4), context_positions=torch.tensor([0]),
        _context_slot_mappings=[torch.tensor([42])],
    )
    runner = SimpleNamespace(
        speculator=draft, model=SimpleNamespace(compute_logits=lambda hidden: logits)
    )

    def sample(hidden: torch.Tensor, batch: object, grammar: object) -> object:
        assert runner.model.compute_logits(hidden) is logits
        return decision

    def propose(
        input_batch: object, aux_hidden_states: list[torch.Tensor],
        last_sampled: torch.Tensor, temperature: torch.Tensor, seeds: torch.Tensor,
    ) -> torch.Tensor:
        manager.run_fullgraph(SimpleNamespace(num_reqs=1, num_tokens=8))
        return proposal

    runner.sample, draft.propose = sample, propose
    capture = probe.ContextCapture(
        runner, tmp_path,
        {"max_frames_per_case": 16, "full_logits_frames": 4,
         "target_feature_layers": [5, 14, 24, 33, 42]}, 0,
    )
    capture.attach()
    batch = SimpleNamespace(
        num_reqs=1, num_tokens=1, num_tokens_after_padding=1, num_draft_tokens=0,
        input_ids=torch.tensor([17]), positions=torch.tensor([0]),
        idx_mapping=torch.tensor([0]), query_start_loc=torch.tensor([0, 1]),
        logits_indices=torch.tensor([0]),
    )
    assert runner.sample(torch.ones(1, 4), batch, None) is decision
    assert draft.propose(
        batch, [torch.ones(1, 4) for _ in range(5)],
        torch.tensor([63]), torch.tensor([0.0]), torch.tensor([42]),
    ) is proposal
    assert capture.pending is None and capture.counts == {"code_32": 1}
    data = load_file(next(tmp_path.glob("*.safetensors")))
    assert data["target_logits"].tolist() == logits.tolist()
    assert data["returned_proposal"].tolist() == proposal.tolist()
    assert data["aux_layer_42"].tolist() == [[1, 1, 1, 1]]
    assert torch.equal(logits, torch.arange(64, dtype=torch.float32).reshape(1, 64))


def test_capture_discards_pending_observation_when_native_sampling_raises(
    tmp_path: Path,
) -> None:
    (tmp_path / "probe-marker.json").write_text('{"active":true,"case":"code_32"}')

    def fail(*args: object) -> None:
        raise RuntimeError("native sampling failed")

    runner = SimpleNamespace(
        sample=fail, model=SimpleNamespace(compute_logits=fail),
        speculator=SimpleNamespace(
            propose=fail, query_cudagraph_manager=SimpleNamespace(run_fullgraph=fail)
        ),
    )
    capture = probe.ContextCapture(runner, tmp_path, {"max_frames_per_case": 16,
                                                   "full_logits_frames": 4}, 0)
    batch = SimpleNamespace(
        num_reqs=1, num_tokens=1, num_tokens_after_padding=1, num_draft_tokens=0,
        input_ids=torch.zeros(1), positions=torch.zeros(1), idx_mapping=torch.zeros(1),
        query_start_loc=torch.zeros(2), logits_indices=torch.zeros(1),
    )
    with pytest.raises(RuntimeError, match="native sampling failed"):
        capture.sample(torch.zeros(1, 4), batch, None)
    assert capture.pending is None
