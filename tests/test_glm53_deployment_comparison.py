"""Guard full GSM8K accounting, cold-cache controls and unchanged production cloning."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Import the remote collectors without GPU or network dependencies."""
    root = Path(__file__).parents[1]
    monkeypatch.syspath_prepend(str(root / "deployment/glm53"))
    monkeypatch.syspath_prepend(str(root / "integrations/glm53_mxfp4"))
    return importlib.import_module("compare_client")


def test_quality_errors_count_once_and_cannot_be_retried(
    client: ModuleType, tmp_path: Path,
) -> None:
    """A transport failure stays in the denominator, and rerunning cannot hide it."""
    calls: list[dict[str, Any]] = []

    class FailingClient:
        def request(self, path: str, payload: dict[str, Any], **kwargs: Any) -> bytes:
            calls.append(payload)
            raise TimeoutError("fixture")

    case = {"source_index": 0, "prompt": "Question", "expected": "42"}
    protocol = {"temperature": 0, "seed": 42, "max_tokens": 1024,
                "reasoning_effort": "low", "cache_salt_prefix": "frozen"}
    row = client.gsm_case(FailingClient(), case, protocol, tmp_path)
    assert row["api_error"] and not row["answer_exact_match"]
    assert calls[0]["model"] == "glm-5.3-flash" and calls[0]["cache_salt"] == "frozen-0"
    with pytest.raises(FileExistsError):
        client.gsm_case(FailingClient(), case, protocol, tmp_path)
    assert len(calls) == 1
    saved = json.loads((tmp_path / "0000.json").read_bytes())
    assert saved["status"] == "complete" and saved["api_error"]


@pytest.mark.parametrize(("hits", "passed"), [
    ([0, 9216, 0], True), ([0, 0, 0], False), ([9216, 9216, 0], False),
    ([0, 9216, 9216], False),
])
def test_cache_control_requires_reuse_and_fresh_salt_miss(
    client: ModuleType, monkeypatch: pytest.MonkeyPatch, hits: list[int], passed: bool,
) -> None:
    """An enabled but ineffective cache, or a warm timed prefix, fails the control."""
    values = iter(hits)
    salts = []

    def fake_cohort(*args: Any, **kwargs: Any) -> dict[str, Any]:
        salts.append(kwargs["salt"])
        return {"api_errors": 0, "counters_delta": {
            'vllm:prefix_cache_hits_total{model_name="test"}': next(values)}}

    monkeypatch.setattr(client, "cohort", fake_cohort)
    result = client.cache_probe(None, [1, 2, 3], "nonce")
    assert result["passed"] is passed
    assert salts[0] == salts[1] and salts[2] != salts[0]


def test_quality_clone_changes_only_speculation_and_ports(client: ModuleType) -> None:
    """Retain all production capacity settings and overlays for target-quality comparison."""
    module = importlib.import_module("prepare_quality_clone")
    source = ["serve", "/model", "--port", "8000", "--master-port", "29669",
              "--max-model-len", "500000", "--enable-prefix-caching",
              "--speculative-config", '{"method":"dflash","num_speculative_tokens":7}',
              "--max-num-seqs", "32"]
    result = module.quality_command(source)
    assert result == ["serve", "/model", "--port", "8047", "--master-port", "29729",
                      "--max-model-len", "500000", "--enable-prefix-caching",
                      "--max-num-seqs", "32"]
    assert "--speculative-config" in source


def test_watchdog_stops_orphaned_client_before_production_restore(
    client: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dead controller cannot leave its client sending requests to restored production."""
    module = importlib.import_module("compare_control")
    state = {"phase": "timing", "service_paused": True, "client_pid": 200,
             "client_start_ticks": "child", "deadline_epoch": float("inf")}
    (tmp_path / "controller.json").write_text(json.dumps(state))
    events = []
    monkeypatch.setattr(module, "ticks", lambda pid: "child" if pid == 200 else None)
    monkeypatch.setattr(module.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(module.os, "killpg", lambda pid, sig: events.append(("stop", pid)))
    monkeypatch.setattr(module, "restore", lambda root: events.append(("restore", root)))
    module.watch(tmp_path, 100, "parent")
    assert events == [("stop", 200), ("restore", tmp_path)]
    assert "disappeared" in json.loads((tmp_path / "controller.json").read_bytes())["error"]


@pytest.mark.parametrize("accepted", [0, 2])
def test_corrected_compatibility_rejects_zero_acceptance_and_retains_quality_failure(
    client: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, accepted: int,
) -> None:
    """Useful speculation must return, while a prefix-output failure cannot become a pass."""
    plan = {"salt": "frozen", "cache_probe_ids": [1, 2], "banks": {
        domain: {"short": [{"prompt_ids": [1, 2]}]} for domain in ("code", "chat")}}
    (tmp_path / "workloads.json").write_text(json.dumps(plan))

    class FakeClient:
        def request(self, path: str, *args: Any) -> bytes:
            if path == "/v1/models":
                return json.dumps({"data": [{"max_model_len": 500000}]}).encode()
            return json.dumps({"results": [{
                "rank": rank, "max_model_len": 500000, "kv_cache_memory_bytes": 6 * 1024**3,
                "head_weight_dtype": "torch.bfloat16", "vocab_mode": "embedding",
            } for rank in (0, 1)]}).encode()

    outputs = iter([[10], [11], [12], [13], [13]])

    def fake_cohort(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"api_errors": 0, "counters_delta": {
            'vllm:spec_decode_num_accepted_tokens_total{engine="0"}': accepted},
            "requests": [{"token_ids": next(outputs)}]}

    monkeypatch.setattr(client, "cohort", fake_cohort)
    monkeypatch.setattr(client, "cache_probe", lambda *args: {"passed": True})
    if accepted == 0:
        with pytest.raises(RuntimeError, match="zero draft acceptance"):
            client.compatibility(tmp_path, FakeClient())
        assert not (tmp_path / "compatibility-B/result.json").exists()
    else:
        client.compatibility(tmp_path, FakeClient())
        result = json.loads((tmp_path / "compatibility-B/result.json").read_bytes())
        assert result["status"] == "execution_ready"
        assert not result["prefix_output_identity_passed"]
        assert not result["existing_target_kl_gate_passed"]
        assert not result["quality_qualified"] and not result["speed_qualified"]
