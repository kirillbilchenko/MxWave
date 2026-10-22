"""Preserve the exact serving policy in the final same-runtime comparison."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("mode", ["timing", "gsm8k"])
def test_same_runtime_changes_only_declared_flags(
    monkeypatch: pytest.MonkeyPatch, mode: str,
) -> None:
    """Keep real production speculation, prefix, capacity and graph flags intact."""
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "deployment/glm53"))
    preparation = importlib.import_module("prepare_same_runtime")
    original = ["serve", "/model", "--port", "8000", "--master-port", "29669",
                "--max-model-len", "500000", "--max-num-seqs", "32",
                "--kv-cache-memory-bytes", str(6 * 1024**3), "--enable-prefix-caching",
                "--speculative-config", '{"model":"/draft","num_speculative_tokens":7}',
                "--compilation-config", '{"cudagraph_capture_sizes":[1,2,4,6,8,12,16,32]}']
    result = preparation.command_for(original, mode, 8051, 29741)
    assert original[original.index("--port") + 1] == "8000"
    assert result[-2:] == ["--worker-extension-cls", "same_runtime_plugin.SameRuntimeWorker"]
    result = result[:-2]
    for flag in ("--port", "--master-port"):
        result[result.index(flag) + 1] = original[original.index(flag) + 1]
    expected = original.copy()
    if mode == "gsm8k":
        index = expected.index("--speculative-config")
        del expected[index:index + 2]
    assert result == expected
    with pytest.raises(ValueError, match="existing production worker"):
        preparation.command_for(original + ["--worker-extension-cls", "original.Worker"],
                                mode, 8051, 29741)


@pytest.mark.parametrize("failure", ["missing_flag", "wrong_native", "unregistered_packer"])
def test_scale_contract_rejects_incompatible_producer_before_loading(
    monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    """Reject the actual launcher failure modes before any model or GPU allocation."""
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "deployment/glm53"))
    plugin = importlib.import_module("same_runtime_plugin")
    monkeypatch.setenv("MXWAVE_PALETTE_MODEL", "1")
    packer = object()
    model = SimpleNamespace(pack_scales=packer, native=lambda _: SimpleNamespace(
        encoding="aligned_nibbles" if failure == "wrong_native" else "sparse_palette"))
    monkeypatch.setitem(sys.modules, "model_plugin", model)
    monkeypatch.setitem(sys.modules, "palette_model_worker", SimpleNamespace(
        _palette_pack=object() if failure == "unregistered_packer" else packer))
    if failure == "missing_flag":
        monkeypatch.delenv("MXWAVE_PALETTE_MODEL")
    with pytest.raises(RuntimeError, match="requires MXWAVE_PALETTE_MODEL|must both use sparse_palette"):
        plugin.scale_codec_contract()


def test_staged_quality_client_requires_transitive_scorer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Catch a missing scorer in an isolated subprocess even when repo imports already work."""
    repository = Path(__file__).parents[1]
    deployment = repository / "deployment/glm53"
    integrations = repository / "integrations/glm53_mxfp4"
    monkeypatch.syspath_prepend(str(deployment))
    preparation = importlib.import_module("prepare_same_runtime")
    for name in ("compare_client.py", "control_stock_trial.py"):
        (tmp_path / name).write_bytes((deployment / name).read_bytes())
    (tmp_path / "gsm8k_benchmark.py").write_bytes((integrations / "gsm8k_benchmark.py").read_bytes())
    protocol = {"questions_per_arm": 1319,
                "cases": [{"source_index": i, "prompt": "Fixture"} for i in range(1319)],
                "speculative_decoding": False, "retries": 0, "seed": 42,
                "temperature": 0, "max_tokens": 1024, "reasoning_effort": "low"}
    (tmp_path / "gsm8k-protocol.json").write_text(json.dumps(protocol))
    with pytest.raises(RuntimeError, match="quality client failed CPU preflight"):
        preparation.quality_client_preflight(tmp_path)
    assert "No module named 'numeric_scoring'" in (
        tmp_path / "quality-client-preflight.log").read_text()
    (tmp_path / "numeric_scoring.py").write_bytes((integrations / "numeric_scoring.py").read_bytes())
    preparation.quality_client_preflight(tmp_path)
    report = json.loads((tmp_path / "quality-client-preflight.json").read_text())
    assert report["status"] == "passed"
    assert set(report["module_hashes"]) == {"compare_client", "gsm8k_benchmark", "numeric_scoring"}


def test_quality_continuation_preserves_deadline_and_rejects_question_retries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """An unfinished question or incomplete timing block must prevent a continuation."""
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "deployment/glm53"))
    preparation = importlib.import_module("prepare_same_runtime")
    state = {"phase": "failed_restored", "service_paused": False, "originals": [],
             "completed_stages": [f"timing-{a}-{t}" for t in range(3) for a in "AB"],
             "deadline_epoch": 12345}
    (tmp_path / "controller.json").write_text(json.dumps(state))
    (tmp_path / "prepared.json").write_text(json.dumps({"files": {}, "decision": {"frozen": True}}))
    result = preparation.quality_continuation(tmp_path)
    assert result["absolute_deadline_epoch"] == 12345
    assert result["decision"] == {"frozen": True}
    (tmp_path / "gsm8k-A").mkdir()
    with pytest.raises(RuntimeError, match="no quality questions attempted"):
        preparation.quality_continuation(tmp_path)
    (tmp_path / "gsm8k-A").rmdir()
    state["completed_stages"].pop()
    (tmp_path / "controller.json").write_text(json.dumps(state))
    with pytest.raises(RuntimeError, match="restored timing completion"):
        preparation.quality_continuation(tmp_path)
