"""Verify streaming latency accounting and matched MTP comparisons."""

from __future__ import annotations

import copy
import importlib
import json
import sys
from pathlib import Path

import pytest


def _script(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parent.parent / "scripts"))
    return importlib.import_module("evaluate_fast_pilots")


def test_latency_uses_token_counts_when_mtp_emits_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    script = _script(monkeypatch)
    result = script._stream_summary([[0.2, 1], [0.3, 4], [0.4, 7]], list(range(7)))
    assert result["ttft_seconds"] == 0.2
    assert result["tpot_seconds"] == pytest.approx(0.2 / 6)
    assert result["output_tokens_per_second"] == pytest.approx(7 / 0.4)
    with pytest.raises(ValueError, match="Invalid"):
        script._stream_summary([[0.2, 1], [0.3, 1], [0.4, 7]], list(range(7)))
    with pytest.raises(ValueError, match="Invalid"):
        script._stream_summary([[0.2, 1], [0.1, 7]], list(range(7)))
    with pytest.raises(ValueError, match="Invalid"):
        script._stream_summary([[0.2, 1], [0.3, 4]], list(range(7)))


def test_mtp_comparison_pairs_repetitions_reports_parity_and_acceptance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _script(monkeypatch)
    off = {
        "serving": [
            {
                "case_id": "code-0",
                "category": "code",
                "repetition": 0,
                "prompt_sha256": "same",
                "token_ids": [1, 2, 3],
                "output_tokens": 3,
                "ttft_seconds": 0.2,
                "tpot_seconds": 0.1,
                "last_token_seconds": 0.4,
            }
        ]
    }
    on = copy.deepcopy(off)
    on["serving"][0].update(
        {
            "tpot_seconds": 0.05,
            "last_token_seconds": 0.3,
            "speculative_decoding": {
                "num_draft_tokens": 8,
                "num_accepted_draft_tokens": 6,
                "num_spec_steps": 4,
            },
        }
    )
    groups = script._serving_comparison(off, on)["groups"]
    assert [item["category"] for item in groups] == ["all", "code"]
    assert groups[0]["paired_median_decode_speedup"] == 2
    assert groups[0]["token_parity_count"] == 1
    assert groups[0]["draft_acceptance_rate"] == 0.75
    assert groups[0]["mean_acceptance_length"] == 2.5
    on["serving"][0]["token_ids"] = [1, 9, 3]
    assert script._serving_comparison(off, on)["groups"][0]["token_parity_count"] == 0
    on["serving"][0]["prompt_sha256"] = "changed"
    with pytest.raises(ValueError, match="identity"):
        script._serving_comparison(off, on)
    with pytest.raises(ValueError, match="pairing"):
        script._serving_comparison(off, {"serving": []})


def test_quality_runtime_does_not_enable_mtp_or_prefix_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _script(monkeypatch)
    off, on = script._engine_kwargs("/model", "h64", 0), script._engine_kwargs("/model", "h64", 2)
    assert not off["enable_prefix_caching"]
    assert off["enable_chunked_prefill"]
    assert off["kv_cache_dtype"] == "bfloat16"
    assert off["max_model_len"] > 8192
    assert "speculative_config" not in off
    assert "per_request_spec_decode_metrics" not in off
    assert on["speculative_config"] == {"method": "mtp", "num_speculative_tokens": 2}
    assert on["per_request_spec_decode_metrics"] == "summary"


def test_runner_creates_output_directory_and_refuses_changed_resume_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _script(monkeypatch)
    runner = importlib.import_module("run_fast_pilots")
    root = tmp_path / "pilot"
    (root / "validation").mkdir(parents=True)
    protocol = root / "validation/protocol.json"
    protocol.write_text("{}")
    original = tmp_path / "experiments/mxwave-corrected-controls-2026-10-03/measurements"
    original.mkdir(parents=True)
    for label in ("bf16", "h64"):
        (original / f"{label}.json").write_text(json.dumps({"checkpoint_sha256": f"sha-{label}"}))
    monkeypatch.setattr(
        runner, "_checkpoint_hash", lambda model: "sha-h64" if model.name == "h64" else "sha-bf16"
    )
    monkeypatch.setattr(runner, "_docker", lambda *args: '[{"State":{"Running":false}}]')
    monkeypatch.setattr(runner.signal, "signal", lambda *args: None)
    jobs = []

    def fake_job(root, name, command, model=None, *, resume=False):
        assert (root / "measurements").is_dir()
        jobs.append(name)

    monkeypatch.setattr(runner, "_run_job", fake_job)
    argv = [
        "run_fast_pilots.py",
        "--spark-root",
        str(tmp_path),
        "--experiment-root",
        str(root),
        "--pause-serving-container",
        "test-service",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    runner.main()
    assert len(jobs) == 4
    identity = (root / "run-identity.json").read_bytes()
    protocol.write_text('{"changed":true}')
    monkeypatch.setattr(sys, "argv", [*argv, "--resume"])
    with pytest.raises(ValueError, match="protocol_sha256"):
        runner.main()
    assert (root / "run-identity.json").read_bytes() == identity
