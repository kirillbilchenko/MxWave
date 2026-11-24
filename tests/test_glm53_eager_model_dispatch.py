"""Protect concurrent response indexing and completeness before a costly model comparison."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))
import measure_eager_model_dispatch as client
import run_eager_model_dispatch as controller


def test_concurrent_quality_sorts_choices_and_retains_full_conditioning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts = [[index] for index in range(16)]

    def request(base: str, route: str, body: dict[str, Any], timeout: int) -> dict[str, Any]:
        assert len(body["prompt"]) == 4 and body["max_tokens"] == 4
        return {"choices": [{"index": offset, "token_ids": [offset] * 4,
                "logprobs": {"top_logprobs": [{f"token_id:{i}": float(-i - 1)
                                               for i in range(4)} for _ in range(4)]}}
                            for offset in reversed(range(4))]}

    monkeypatch.setattr(client, "request", request)
    result = client.collect_parallel("isolated", prompts, tmp_path, "first", 4)
    assert result["status"] == "complete" and len(result["contexts"]) == 16
    assert len(result["raw_api_responses"]) == 4
    for index, context in enumerate(result["contexts"]):
        assert context["index"] == index and context["token_ids"] == [index % 4] * 4
        assert context["prompt_sha256"] == hashlib.sha256(
            json.dumps(prompts[index], separators=(",", ":")).encode()).hexdigest()
        assert context["steps"][2]["conditioning_token_ids"] == [index % 4] * 2
        for step in context["steps"]:
            values = np.fromfile(tmp_path / step["file"], dtype="<f4")
            np.testing.assert_array_equal(values, [-1., -2., -3., -4.])
    assert json.loads((tmp_path / "first-decode-quality.json").read_bytes()) == result


def test_missing_concurrent_choice_cannot_be_scored_as_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(client, "request", lambda *args, **kwargs: {"choices": [{"index": 0}]})
    with pytest.raises(ValueError, match="Incomplete concurrent"):
        client.collect_parallel("isolated", [[i] for i in range(16)], tmp_path, "missing", 4)
    assert json.loads((tmp_path / "missing-decode-quality.json").read_bytes())["status"] == "running"


def test_mode_switch_refuses_an_active_request(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = iter((0., 1., 11.))
    monkeypatch.setattr(client.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(client.time, "sleep", lambda value: None)
    monkeypatch.setattr(client.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(
        b'vllm:num_requests_running{engine="0"} 1\nvllm:num_requests_waiting{engine="0"} 0\n'))
    called = []
    monkeypatch.setattr(client, "request", lambda *args, **kwargs: called.append(args))
    with pytest.raises(RuntimeError, match="requests still in flight"):
        client.switch("isolated", "eager-direct")
    assert not called


def test_model_launch_loads_only_existing_compressed_candidate_and_keeps_batch_one_graphs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modes = []

    def engine(rank: int, mode: str, *args: Any, **kwargs: Any) -> Any:
        modes.append(mode)
        command = ["serve", "/model", "--worker-extension-cls", "original.Worker",
                   "--compilation-config", "{}"]
        return copy.deepcopy((["docker", "run", "--name", "owned", "--entrypoint", "python",
                               *command], {"command": command, "environment": {}, "mounts": []}))

    monkeypatch.setattr(controller.c, "engine", engine)
    argv, metadata = controller.launch(0, {}, 154880, "manifest")
    assert modes == ["kda"]
    command = metadata["command"]
    assert command[command.index("--worker-extension-cls") + 1].endswith(".EagerDispatchWorker")
    assert json.loads(command[command.index("--compilation-config") + 1])["cudagraph_capture_sizes"] == [1]
    assert metadata["environment"]["MXWAVE_MOE_MODE"] == "aligned"
    assert metadata["environment"]["MXWAVE_STABLE_MOE_DISPATCH"] == "1"
    assert metadata["environment"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in argv


def test_reference_probe_launch_disables_compression_and_keeps_graph_four(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reference-only diagnostic must not accidentally load the historical candidate arm."""
    modes = []

    def engine(rank: int, mode: str, *args: Any, **kwargs: Any) -> Any:
        modes.append(mode)
        command = ["serve", "/model", "--worker-extension-cls", "original.Worker",
                   "--compilation-config", "{}"]
        return copy.deepcopy((["docker", "run", "--name", "owned", "--entrypoint", "python",
                               *command], {"command": command, "environment": {}, "mounts": []}))

    monkeypatch.setattr(controller.c, "engine", engine)
    monkeypatch.setattr(controller, "ARM_LABEL", "control")
    monkeypatch.setattr(controller, "WORKER_CLASS", "decode_state_worker.DecodeStateWorker")
    monkeypatch.setattr(controller, "CAPTURE_SIZES", [1, 4])
    argv, metadata = controller.launch(0, {}, 154880, "manifest")
    assert modes == ["stock"]
    assert metadata["environment"]["MXWAVE_MOE_MODE"] == "stock"
    assert metadata["environment"]["MXWAVE_PALETTE_MODEL"] == "0"
    assert metadata["environment"]["MXWAVE_GLM53_VOCAB_MODE"] == "stock"
    assert metadata["environment"]["MXWAVE_STABLE_MOE_DISPATCH"] == "1"
    command = metadata["command"]
    assert command[command.index("--worker-extension-cls") + 1].endswith(".DecodeStateWorker")
    assert json.loads(command[command.index("--compilation-config") + 1])[
        "cudagraph_capture_sizes"] == [1, 4]
    assert not any("--speculative" in part for part in argv)
