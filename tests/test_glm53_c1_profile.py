"""Keep native profiling separate from serving changes and retain failure evidence."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


@pytest.fixture
def modules(monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, ModuleType]:
    """Import the operational tools without networking or CUDA."""
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "deployment/glm53"))
    return (importlib.import_module("prepare_c1_profile"),
            importlib.import_module("c1_profile_client"))


@pytest.mark.parametrize("production", [True, False])
def test_profile_preserves_draft_and_capacity_configuration(
    modules: tuple[ModuleType, ModuleType], production: bool,
) -> None:
    """Both deployments retain their existing draft depths, cache and graph policies."""
    preparation, _ = modules
    original = ["serve", "/model", "--port", "8000", "--master-port", "29719",
                "--host", "127.0.0.1", "--max-model-len", "500000",
                "--enable-prefix-caching", "--kv-cache-memory-bytes", str(6 * 1024**3),
                "--speculative-config", '{"num_speculative_tokens":5}',
                "--compilation-config", '{"cudagraph_capture_sizes":[1,2,4,6,8]}']
    result = preparation.profile_command(original, production=production)
    for flag in ("--port", "--master-port", "--host"):
        result[result.index(flag) + 1] = original[original.index(flag) + 1]
    del result[-2:]
    assert result == original
    with pytest.raises(ValueError, match="existing profiler"):
        preparation.profile_command(original + ["--profiler-config", "{}"],
                                    production=production)


@pytest.mark.parametrize("failure", ["api", "warm_cache", None])
def test_profile_cleanup_and_retention(
    modules: tuple[ModuleType, ModuleType], tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, failure: str | None,
) -> None:
    """A failed profiled request still stops profiling and keeps the raw failure."""
    _, client = modules
    (tmp_path / "workloads.json").write_text(json.dumps({
        "salt": "frozen", "cache_probe_ids": [1],
        "cells": [{"name": "code-short-c1", "prompts": [[1, 2, 3]]}],
    }))
    calls: list[str] = []
    salts: list[str] = []

    class FakeClient:
        def request(self, path: str, payload: Any = None, **kwargs: Any) -> bytes:
            calls.append(path)
            return (b'{"data":[{"max_model_len":500000}]}' if path == "/v1/models"
                    else b'{}')

    def fake_cohort(*args: Any, **kwargs: Any) -> dict[str, Any]:
        salts.append(kwargs["salt"])
        profiling = kwargs["salt"].endswith("-profile")
        return {"api_errors": int(profiling and failure == "api"), "output_tokens": 256,
                "counters_delta": {"vllm:prefix_cache_hits_total":
                                   int(profiling and failure == "warm_cache")}}

    monkeypatch.setattr(client, "cache_probe", lambda *a, **k: {"passed": True})
    monkeypatch.setattr(client, "cohort", fake_cohort)
    if failure:
        with pytest.raises(RuntimeError):
            client.collect(tmp_path, "B", FakeClient())
        assert (tmp_path / "profile-B/code-short-c1-failed-request.json").is_file()
    else:
        client.collect(tmp_path, "B", FakeClient())
        result = json.loads((tmp_path / "profile-B/code-short-c1.json").read_text())
        assert len(result["controls_before"]) == 3
        assert len(result["controls_after"]) == 2
        assert result["profiled_request_is_timing"] is False
    assert calls[-2:] == ["/start_profile", "/stop_profile"]
    assert len(salts) == len(set(salts))
