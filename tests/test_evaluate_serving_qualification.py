from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType


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
