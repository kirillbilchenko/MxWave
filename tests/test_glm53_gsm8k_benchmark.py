"""Full-test coverage, transparent scoring and no-retry contracts for GSM8K."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


@pytest.fixture
def bench(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load the CPU-only benchmark helpers."""
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))
    return importlib.import_module("gsm8k_benchmark")


def response(text: str, finish: str = "stop") -> dict[str, Any]:
    """Build one naturally stopped or truncated chat completion."""
    return {"choices": [{"message": {"content": text}, "finish_reason": finish}]}


@pytest.mark.parametrize(("text", "expected", "primary", "strict"), [
    ("#### 42", "42", True, True), ("42", "42", True, True),
    ("The final answer is 42 dollars.", "42", True, False),
    ("First 12, then #### 42 dollars.", "42", True, False),
    ("#### 1,234.50", "1234.5", True, True), ("#### 0", "0", True, True),
    ("<think>42", "42", False, False), ("<think>12</think>#### 42", "42", True, True),
    ("#### 41", "42", False, False), ("No answer", "42", False, False),
])
def test_primary_numeric_and_strict_format_are_separate(
    bench: ModuleType, text: str, expected: str, primary: bool, strict: bool,
) -> None:
    """Never rewrite a format failure as a pass in the original strict metric."""
    value = bench.score(response(text), expected)
    assert value["answer_exact_match"] is primary
    assert value["strict_format_match"] is strict
    truncated = bench.score(response(text, "length"), expected)
    assert not truncated["answer_exact_match"] and not truncated["strict_format_match"]
    assert truncated["truncated"]


def test_full_dataset_is_frozen_in_source_order(bench: ModuleType) -> None:
    """Require every test case, the original payload hash and deterministic generation settings."""
    raw = b"\n".join(json.dumps({"question": f"Question {i}", "answer": f"Work #### {i}"}).encode()
                      for i in range(1319))
    manifest = {"files": {"gsm8k-test.jsonl": {"sha256": hashlib.sha256(raw).hexdigest()}}}
    protocol = bench.freeze(raw, manifest)
    assert len(protocol["cases"]) == 1319 and protocol["cases"][-1]["source_index"] == 1318
    assert protocol["concurrency"] == 4 and protocol["max_tokens"] == 1024
    assert protocol["retries"] == 0 and not protocol["qualification_attempted"]
    with pytest.raises(ValueError, match="payload changed"):
        bench.freeze(raw + b"\n", manifest)
