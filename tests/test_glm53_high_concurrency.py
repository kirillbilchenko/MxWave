"""Keep the C16 capacity, policy matching and strict probability contracts explicit."""

from __future__ import annotations

import copy
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))
from high_concurrency_protocol import CAPTURES, validate_snapshot


def worker(rank: int) -> dict[str, Any]:
    """Describe a complete pair member without importing a GPU runtime."""
    return {"rank": rank, "physical_capture_sizes": CAPTURES,
        "graphs_enabled": True, "max_num_seqs": 16, "max_batched_tokens": 2048,
        "KV_bytes_per_rank": 4294967296, "max_model_len": 8192,
        "speculative_config_absent": True, "descriptors": {
            str(n): {"mode": "FULL", "num_tokens": n} for n in CAPTURES}}


def test_c16_requires_actual_capacity_and_exact_replay() -> None:
    rows = [worker(0), worker(1)]
    validate_snapshot(rows)
    for field, value in (("max_num_seqs", 8), ("KV_bytes_per_rank", 8589934592),
                         ("speculative_config_absent", False),
                         ("physical_capture_sizes", [1, 2, 4, 8])):
        changed = copy.deepcopy(rows)
        changed[1][field] = value
        with pytest.raises(ValueError, match="scope changed"):
            validate_snapshot(changed)
    changed = copy.deepcopy(rows)
    changed[1]["descriptors"]["16"] = {"mode": "NONE", "num_tokens": 16}
    with pytest.raises(ValueError, match="exact graph"):
        validate_snapshot(changed)
    with pytest.raises(ValueError, match="both"):
        validate_snapshot([worker(0), worker(0)])


def test_stock_and_compressed_keep_identical_serving_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = importlib.import_module("run_high_concurrency")
    labels = []

    def original(rank: int, revision: str, label: str) -> tuple[list[str], dict[str, Any]]:
        labels.append(label)
        command = ["serve", "/model", "--worker-extension-cls", "old.Worker",
            "--max-num-seqs", "8", "--kv-cache-memory-bytes", "4294967296",
            "--compilation-config", "{}"]
        return (["docker", "run", "--entrypoint", "python3", "image", *command],
                {"command": command, "environment": {"MXWAVE_MOE_MODE": label}})

    monkeypatch.setattr(runner.graph, "launch", original)
    result = [runner.launch(0, "c816v1", label) for label in ("control", "candidate")]
    assert labels == ["control", "candidate"]
    for argv, meta in result:
        assert "--speculative-config" not in argv
        assert meta["environment"]["MXWAVE_PREFILL_ADMISSION"] == "1"
        for command in (argv, meta["command"]):
            assert command[command.index("--max-num-seqs") + 1] == "16"
            assert command[command.index("--kv-cache-memory-bytes") + 1] == "4294967296"
            assert json.loads(command[command.index("--compilation-config") + 1]) == {
                "mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": [1, 2, 4, 8, 16], "max_cudagraph_capture_size": 16}
    assert result[0][1]["command"] == result[1][1]["command"]
    assert result[0][1]["environment"]["MXWAVE_MOE_MODE"] == "control"
    assert result[1][1]["environment"]["MXWAVE_MOE_MODE"] == "candidate"


def test_numerical_failures_remain_failures_and_all_comparisons_are_prespecified() -> None:
    analysis = importlib.import_module("analyze_high_concurrency")
    pairs = analysis.quality_pairs()
    assert len(pairs) == len(set(pairs)) == 24
    assert sum(a.startswith("control") and b.startswith("candidate") for a, b in pairs) == 8
    value = {"decode_steps": 48, "expected_decode_steps": 48,
        "all_decode_conditioning_matches": True, "all_generated_ids_match": True,
        "mean_decode_kl": 1e-5, "mean_decode_nll_difference": -1e-4}
    assert analysis.numerical_passed(value)
    for field, wrong in (("decode_steps", 47), ("all_generated_ids_match", False),
                          ("all_decode_conditioning_matches", False),
                          ("mean_decode_kl", 1.0001e-5),
                          ("mean_decode_nll_difference", -1.0001e-4)):
        assert not analysis.numerical_passed({**value, field: wrong})
