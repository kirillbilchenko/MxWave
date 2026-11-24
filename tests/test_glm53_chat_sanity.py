"""Final-answer gates reject truncated reasoning without hiding the raw completion."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def checker(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load the optional CPU-only API checker with its neighboring utility imports."""
    directory = Path(__file__).parents[1] / "integrations/glm53_mxfp4"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("glm_chat_sanity_test",
                                                 directory / "check_full_model.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Avoid caching the neighboring optional integration module across tests.
    previous = sys.modules.pop("benchmark_full_model", None)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop("benchmark_full_model", None)
        if previous is not None:
            sys.modules["benchmark_full_model"] = previous
    return module


@pytest.mark.parametrize(("message", "finish", "passed"), [
    ({"content": "4."}, "stop", True),
    ({"content": "4"}, "length", False),
    ({"content": "<think>The result is 4"}, "stop", False),
    ({"content": "Reason about the sum.</think>4"}, "stop", True),
    ({"content": "Reason about the sum.</think>4"}, "length", False),
    ({"content": None, "reasoning": "The result is 4"}, "length", False),
    ({"content": "4", "reasoning_content": "Two plus two is four."}, "stop", True),
    ({"content": "5", "reasoning_content": "The result might be 4."}, "stop", False),
])
def test_only_finished_final_answers_pass(
    checker: ModuleType, message: dict[str, str | None], finish: str, passed: bool,
) -> None:
    choice = {"message": message, "finish_reason": finish}
    result = checker.assess_chat(choice, "4")
    assert result["passed"] is passed
    assert result["raw_message"] == message
    assert result["finish_reason"] == finish
