"""Check that forced-history diagnostics preserve raw predictions and chunk alignment."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


def load_worker(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Load the RPC mixin without a vLLM installation."""
    folder = Path(__file__).parents[1] / "integrations/glm53_mxfp4"
    monkeypatch.syspath_prepend(str(folder))
    parent = types.ModuleType("warmed_model_worker")
    parent.WarmedModelWorker = object  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "warmed_model_worker", parent)
    spec = importlib.util.spec_from_file_location("rounding_capture_test", folder / "kda_rounding_worker.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_capture_precedes_forcing_and_skips_incomplete_prefill(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A forced answer must never be mistaken for the original logit distribution."""
    module = load_worker(monkeypatch)
    monkeypatch.setattr(module, "Path", lambda _: tmp_path)
    worker = module.KdaRoundingWorker()
    logits = torch.arange(154880, dtype=torch.float32).reshape(1, -1)
    model = types.SimpleNamespace(compute_logits=lambda _: logits.clone())
    runner = types.SimpleNamespace(
        model=model, sample=lambda hidden, batch, grammar: model.compute_logits(hidden),
    )
    worker.model_runner = runner
    worker._rounding_rank = 0
    worker._rounding_mode = "original"
    worker._rounding_capture = None
    worker._install_rounding_observer()
    worker.mxwave_rounding_arm("a1_short", 2, [10, 11], [1, 2, 3, 4])

    def batch(positions: list[int], ids: list[int]) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            num_reqs=1, num_draft_tokens=0, is_padding=torch.zeros(len(ids), dtype=torch.bool),
            positions=torch.tensor(positions), input_ids=torch.tensor(ids),
            num_tokens=len(ids), num_tokens_after_padding=len(ids), has_prefill=len(ids) > 1,
        )

    first = runner.sample(None, batch([0, 1], [1, 2]), None)
    assert torch.equal(first, logits)
    assert worker._rounding_capture["steps"] == []
    final_prefill = runner.sample(None, batch([2, 3], [3, 4]), None)
    assert int(final_prefill.argmax()) == 10
    assert torch.isneginf(final_prefill[0, :10]).all()
    decode = runner.sample(None, batch([4], [10]), None)
    assert int(decode.argmax()) == 11
    report = worker.mxwave_rounding_disarm()
    raw = logits.numpy().astype("<f4").tobytes()
    assert len(report["batches"]) == 3 and len(report["steps"]) == 2
    for row in report["steps"]:
        assert row["sha256"] == hashlib.sha256(raw).hexdigest()
        assert row["argmax"] == 154879
        assert (tmp_path / row["file"]).read_bytes() == raw


def test_invalid_forced_history_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Out-of-vocabulary or truncated forcing must fail before an API request."""
    module = load_worker(monkeypatch)
    worker = module.KdaRoundingWorker()
    worker._rounding_capture = None
    worker._rounding_rank = 0
    worker._rounding_mode = "original"
    with pytest.raises(AssertionError):
        worker.mxwave_rounding_arm("bad", 2, [154880], [1, 2])
