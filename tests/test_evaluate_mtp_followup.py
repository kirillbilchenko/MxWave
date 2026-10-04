"""Check matched target histories, probability diagnostics and isolated task scoring."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file


def _script(monkeypatch: pytest.MonkeyPatch, name: str = "evaluate_mtp_followup"):
    monkeypatch.syspath_prepend(str(Path(__file__).parent.parent / "scripts"))
    return importlib.import_module(name)


def test_same_target_loader_uses_context_suffixes_and_rejects_changed_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _script(monkeypatch)
    values = list(range(12289))
    path = tmp_path / "protocol-tokens.safetensors"
    save_file({"token_ids": torch.tensor(values, dtype=torch.int64)}, str(path))
    document = {
        "format": script.FORMAT,
        "token_data": {
            "filename": path.name,
            "sha256": script._hash_file(path),
            "count": len(values),
        },
        "books": [
            {
                "token_offset": 0,
                "token_count": len(values),
                "token_ids_sha256": script._token_digest(values),
            }
        ],
        "contexts": [
            {
                "book_index": 0,
                "end_offset": 8192,
                "prefix_tokens": length,
                "target_token_id": 8192,
                "token_ids_sha256": script._token_digest(values[8192 - length : 8192]),
            }
            for length in script.LENGTHS
        ],
        "diagnostics": [],
        "tasks": [],
    }
    manifest = tmp_path / "protocol.json"
    script._write_json(manifest, document)
    result, _ = script._load(manifest)
    assert {c["target_token_id"] for c in result["contexts"]} == {8192}
    assert result["contexts"][0]["token_ids"] == values[7680:8192]
    assert result["contexts"][2]["token_ids"][-512:] == result["contexts"][0]["token_ids"]
    document["contexts"][0]["target_token_id"] = 999
    script._write_json(manifest, document)
    with pytest.raises(ValueError, match="target mismatch"):
        script._load(manifest)


def test_near_tie_diagnostic_preserves_common_history_and_detects_non_argmax(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _script(monkeypatch)
    off = {
        "diagnostics": [
            {
                "case_id": "x",
                "repetition": 0,
                "prompt_sha256": "p",
                "token_ids": [5, 10],
                "logprobs": [{}, {"10": {"logprob": -0.68}, "11": {"logprob": -0.70}}],
            }
        ]
    }
    on = {
        "diagnostics": [
            {
                "case_id": "x",
                "repetition": 0,
                "prompt_sha256": "p",
                "token_ids": [5, 11],
                "logprobs": [{}, {"10": {"logprob": -0.70}, "11": {"logprob": -0.68}}],
            }
        ]
    }
    result = script._diagnostic_compare(off, on)[0]
    assert result["first_difference"] == 1
    assert result["off"]["selected_is_argmax"] and result["on"]["selected_is_argmax"]
    assert result["off"]["chosen_minus_other_logit"] == pytest.approx(0.02)
    assert result["common_prefix_sha256"] == script._token_digest([5])
    on["diagnostics"][0]["logprobs"][1]["11"]["logprob"] = -0.72
    assert not script._diagnostic_compare(off, on)[0]["on"]["selected_is_argmax"]


def test_book_averaging_keeps_independent_books_and_rejects_duplicate_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _script(monkeypatch)
    contexts = [
        {"book_index": b, "source_target_position": target, "prefix_tokens": length}
        for b in range(2)
        for target in script.TARGETS
        for length in script.LENGTHS
    ]
    values = np.array([1, 2, 3, 3, 4, 5, 10, 20, 30, 30, 40, 50])
    np.testing.assert_equal(script._book_values(values, contexts), [[2, 20], [3, 30], [4, 40]])
    contexts[3]["source_target_position"] = contexts[0]["source_target_position"]
    with pytest.raises(ValueError, match="pairing"):
        script._book_values(values, contexts)


def test_single_token_timing_has_no_decode_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    script = _script(monkeypatch)
    assert script._timing([[0.2, 1]], [7])["tpot_seconds"] is None
    assert script._timing([[0.2, 1], [0.3, 4]], [1, 2, 3, 4])["tpot_seconds"] == pytest.approx(
        0.1 / 3
    )
    with pytest.raises(ValueError, match="streaming"):
        script._timing([[float("nan"), 1]], [7])


def test_isolated_grader_checks_values_mutation_and_rejects_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _script(monkeypatch, "mtp_task_screen")
    case = next(c for c in script.task_cases() if c["id"] == "code-0")
    assert script.score(case, "def stable_unique(values):\n    return list(dict.fromkeys(values))")[
        "passed"
    ]
    assert not script.score(case, "def stable_unique(values):\n    return sorted(set(values))")[
        "passed"
    ]
    assert not script.score(case, "def stable_unique(values):\n    values.clear()\n    return []")[
        "passed"
    ]
    assert not script.score(case, "import os\ndef stable_unique(values):\n    return []")["passed"]
    math = next(c for c in script.task_cases() if c["id"] == "math-0")
    assert script.score(math, "1023\n")["passed"]
    assert not script.score(math, "1023 dollars")["passed"]


def test_runner_rejects_changed_source_before_pausing_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _script(monkeypatch)
    runner = importlib.import_module("run_mtp_followup")
    root = tmp_path / "pilot"
    (root / "validation").mkdir(parents=True)
    (root / "validation/protocol.json").write_text('{"executed_source_sha256":"wrong"}')
    monkeypatch.setattr(runner, "_docker", lambda *args: pytest.fail("Must not touch serving"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_mtp_followup.py",
            "--spark-root",
            str(tmp_path),
            "--experiment-root",
            str(root),
            "--pause-serving-container",
            "service",
        ],
    )
    with pytest.raises(ValueError, match="sources changed"):
        runner.main()
