"""The prefix gate must retain API-null cases and reject false cache evidence."""

from __future__ import annotations

from deployment.glm53.smoke_client import summarize


def fixture() -> dict:
    """Return the nullable-usage shape observed in stock V6."""
    return {
        "responses": [{"response": {
            "choices": [{"message": {"content": None, "reasoning": "same"}}],
            "usage": {"prompt_tokens": 10000, "prompt_tokens_details": None},
        }} for _ in range(3)],
        "metrics_before": 'vllm:prefix_cache_hits_total{model="test"} 0\n'
                          'vllm:prefix_cache_hits_created{model="test"} 100\n',
        "metrics_after": 'vllm:prefix_cache_hits_total{model="test"} 0\n'
                         'vllm:prefix_cache_hits_created{model="test"} 200\n',
    }


def test_null_details_and_counter_timestamps_cannot_pass_reuse_gate():
    result = summarize(fixture(), min_prompt_tokens=9280)
    assert result["cached_tokens"] == [0, 0, 0]
    assert result["prefix_hit_counter_delta"] == 0
    assert result["status"] == "failed"


def test_real_reuse_still_requires_output_identity_and_prompt_coverage():
    data = fixture()
    data["metrics_after"] = 'vllm:prefix_cache_hits_total{model="test"} 9216\n'
    assert summarize(data, min_prompt_tokens=9280)["status"] == "passed"
    data["responses"][1]["response"]["choices"][0]["message"]["reasoning"] = "different"
    result = summarize(data, min_prompt_tokens=9280)
    assert result["prefix_reuse_observed"] and not result["repeated_outputs_equal"]
    assert result["status"] == "failed"
    data["responses"][1]["response"]["choices"][0]["message"]["reasoning"] = "same"
    data["responses"][0]["response"]["usage"]["prompt_tokens"] = 4435
    result = summarize(data, min_prompt_tokens=9280)
    assert not result["prompt_covers_cache_blocks"] and result["status"] == "failed"
