"""Preserve the exact serving policy in the final same-runtime comparison."""

from __future__ import annotations

import importlib
from pathlib import Path

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
