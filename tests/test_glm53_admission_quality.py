"""Concurrent likelihood evidence must keep context identity and reject partial vocabularies."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))
import admission_quality as quality
from decode_quality import compare


@pytest.mark.parametrize("concurrency", [2, 4, 8, 16])
def test_parallel_contexts_keep_correct_files_conditioning_and_full_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, concurrency: int,
) -> None:
    def request(base: str, route: str, body: dict[str, Any], timeout: int) -> dict[str, Any]:
        token = body["prompt"][0] % 3
        return {"choices": [{"token_ids": [token] * 4, "logprobs": {"top_logprobs": [
            {f"token_id:{i}": -.1 * (1 + i + token) for i in range(3)} for _ in range(4)]}}]}

    monkeypatch.setattr(quality, "request", request)
    prompts = [[i, 2, 1] for i in range(16)]
    a = quality.collect_concurrent("unused", prompts, tmp_path, "a", 3, concurrency=concurrency)
    b = quality.collect_concurrent("unused", prompts, tmp_path, "b", 3, concurrency=concurrency)
    assert [r["index"] for r in a["contexts"]] == list(range(16))
    assert [r["token_ids"][0] for r in a["contexts"]] == [i % 3 for i in range(16)]
    result = compare(a, b, tmp_path)
    assert result["decode_steps"] == result["expected_decode_steps"] == 48
    assert result["all_decode_conditioning_matches"] and result["all_generated_ids_match"]
    assert result["mean_decode_kl"] == 0.


def test_partial_vocabulary_is_retained_as_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(quality, "request", lambda *args, **kwargs: {"choices": [
        {"token_ids": [0] * 4, "logprobs": {"top_logprobs": [{"token_id:0": -1.}] * 4}}]})
    with pytest.raises(ValueError, match="Missing full"):
        quality.collect_concurrent("unused", [[i] for i in range(16)], tmp_path, "bad", 3)
    assert json.loads((tmp_path / "bad-decode-quality.json").read_bytes())["status"] == "failed"
