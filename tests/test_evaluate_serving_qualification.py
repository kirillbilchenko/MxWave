from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


def _load_script() -> ModuleType:
    path = Path("scripts/evaluate_serving_qualification.py")
    spec = importlib.util.spec_from_file_location("evaluate_serving_qualification", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_manifest_has_fixed_bounded_qualification_matrix(tmp_path: Path) -> None:
    module = _load_script()
    output = tmp_path / "cases.json"

    module.prepare_manifest(output, [8000, 32000, 60000])

    manifest = json.loads(output.read_text())
    assert manifest["format"] == "mxwave-serving-qualification-cases-v1"
    assert manifest["case_count"] == 100
    assert manifest["scoreable_case_count"] == 87
    cases = manifest["cases"]
    assert len({case["id"] for case in cases}) == 100
    long_cases = [case for case in cases if case["id"].startswith("long-")]
    assert len(long_cases) == 27
    assert {
        case["metadata"]["target_context_tokens"] for case in long_cases
    } == {8000, 32000, 60000}
    assert all(len(case["prompt_sha256"]) == 64 for case in cases)


def test_scorers_are_strict_but_json_key_order_is_irrelevant() -> None:
    module = _load_script()
    exact = {"id": "exact", "scoring": {"kind": "exact", "expected": "VALUE"}}
    structured = {
        "id": "json",
        "scoring": {"kind": "json", "expected": {"enabled": True, "items": 3}},
    }
    unscored = {"id": "none", "scoring": {"kind": "none"}}

    assert module._score_content(exact, " VALUE\n") is True
    assert module._score_content(exact, "The value is VALUE") is False
    assert module._score_content(structured, '{"items":3,"enabled":true}') is True
    assert module._score_content(structured, "```json\n{}\n```") is False
    assert module._score_content(unscored, "anything") is None


@pytest.mark.parametrize("reasoning_effort", [None, "none", "high"])
def test_collect_optionally_sends_top_level_reasoning_effort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reasoning_effort: str | None,
) -> None:
    module = _load_script()
    manifest_path = tmp_path / "cases.json"
    cli_args = [
        "collect",
        "--cases",
        str(manifest_path),
        "--output",
        str(tmp_path / "collection.json"),
        "--label",
        "nex",
        "--model",
        "nex",
        "--env-file",
        str(tmp_path / ".env"),
        "--expected-mtp-tokens",
        "0",
        "--checkpoint-sha256",
        "a" * 64,
    ]
    if reasoning_effort is not None:
        cli_args.extend(["--reasoning-effort", reasoning_effort])
    assert module.build_parser().parse_args(cli_args).reasoning_effort == reasoning_effort

    case = module._case(
        case_id="reasoning-effort",
        category="smoke",
        content="Return VALUE.",
        max_tokens=8,
        scoring={"kind": "exact", "expected": "VALUE"},
    )
    manifest_path.write_text(
        json.dumps(
            {
                "format": "mxwave-serving-qualification-cases-v1",
                "case_count": 1,
                "cases": [case],
            }
        )
    )
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_API_KEY=test-key\n")
    requests: list[dict[str, object]] = []

    def fake_request_json(
        *,
        url: str,
        api_key: str,
        payload: dict[str, object] | None,
        timeout_seconds: float,
    ) -> dict[str, object]:
        assert api_key == "test-key"
        assert timeout_seconds == 30.0
        if url.endswith("/models"):
            assert payload is None
            return {"data": [{"id": "nex"}]}
        assert url.endswith("/chat/completions")
        assert payload is not None
        requests.append(payload)
        return {
            "choices": [
                {
                    "message": {"content": "VALUE"},
                    "finish_reason": "stop",
                    "logprobs": {
                        "content": [
                            {
                                "token": "VALUE",
                                "bytes": list(b"VALUE"),
                                "logprob": 0.0,
                            }
                        ]
                    },
                }
            ],
            "usage": {"prompt_tokens": 4},
        }

    monkeypatch.setattr(module, "_request_json", fake_request_json)
    output = tmp_path / "collection.json"
    module.collect(
        manifest_path=manifest_path,
        output=output,
        label="nex",
        model="nex",
        env_file=env_file,
        base_url="http://localhost/v1",
        expected_mtp_tokens=0,
        context_length=4096,
        timeout_seconds=30.0,
        checkpoint_sha256="a" * 64,
        reasoning_effort=reasoning_effort,
    )

    assert len(requests) == 1
    report = json.loads(output.read_text())
    if reasoning_effort is None:
        assert "reasoning_effort" not in requests[0]
        assert "reasoning_effort" not in report["runtime"]
        assert requests[0]["chat_template_kwargs"] == {"enable_thinking": False}
        assert report["runtime"]["enable_thinking"] is False
    else:
        assert requests[0]["reasoning_effort"] == reasoning_effort
        assert report["runtime"]["reasoning_effort"] == reasoning_effort
        assert "chat_template_kwargs" not in requests[0]
        assert report["runtime"]["enable_thinking"] is (reasoning_effort != "none")


def test_collect_rejects_unsupported_reasoning_effort(tmp_path: Path) -> None:
    module = _load_script()

    with pytest.raises(ValueError, match="Reasoning effort must be one of"):
        module.collect(
            manifest_path=tmp_path / "missing-cases.json",
            output=tmp_path / "collection.json",
            label="invalid",
            model="invalid",
            env_file=tmp_path / ".env",
            base_url="http://localhost/v1",
            expected_mtp_tokens=0,
            context_length=4096,
            timeout_seconds=30.0,
            checkpoint_sha256="a" * 64,
            reasoning_effort="unsupported",
        )


def test_compare_reports_exact_parity_and_long_context_gate(tmp_path: Path) -> None:
    module = _load_script()
    manifest_path = tmp_path / "cases.json"
    module.prepare_manifest(manifest_path, [8000])
    manifest = json.loads(manifest_path.read_text())
    manifest_sha256 = module._sha256_file(manifest_path)

    results = []
    for case in manifest["cases"]:
        scoring = case["scoring"]
        content = (
            json.dumps(scoring["expected"], sort_keys=True)
            if scoring["kind"] == "json"
            else str(scoring.get("expected", "unscored"))
        )
        quality_pass = None if scoring["kind"] == "none" else True
        metadata = case["metadata"]
        prompt_tokens = metadata.get("target_context_tokens", 100)
        results.append(
            {
                "case_id": case["id"],
                "category": case["category"],
                "content": content,
                "content_sha256": module._sha256_bytes(content.encode()),
                "emitted_token_bytes_sha256": module._sha256_bytes(content.encode()),
                "finish_reason": "stop",
                "quality_pass": quality_pass,
                "usage": {"prompt_tokens": prompt_tokens},
                "error": None,
            }
        )

    paths = [tmp_path / "mtp0.json", tmp_path / "mtp2.json"]
    for index, path in enumerate(paths):
        path.write_text(
            json.dumps(
                {
                    "format": "mxwave-serving-qualification-collection-v1",
                    "complete": True,
                    "checkpoint_sha256": "a" * 64,
                    "protocol_sha256": manifest_sha256,
                    "label": f"mtp{index * 2}",
                    "manifest_sha256": manifest_sha256,
                    "runtime": {"expected_mtp_tokens": index * 2},
                    "results": results,
                }
            )
        )

    output = tmp_path / "comparison.json"
    module.compare(
        manifest_path=manifest_path,
        first_path=paths[0],
        second_path=paths[1],
        output=output,
    )

    report = json.loads(output.read_text())
    assert report["gates"] == {
        "long_context_semantic_accuracy": True,
        "pass": True,
        "token_parity": True,
    }
    assert report["parity"]["emitted_token_bytes_equal"] == 82
    assert report["first"]["quality"]["long_context_by_target_tokens"]["8000"] == {
        "accuracy": 1.0,
        "observed_prompt_tokens_max": 8000,
        "observed_prompt_tokens_min": 8000,
        "passed": 9,
        "total": 9,
    }
