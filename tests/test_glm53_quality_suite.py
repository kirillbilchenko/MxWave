"""Quality evidence must retain conditioning, scores and the declared reference."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def source_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Import CPU-only helpers without a remote connection or GPU."""
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))


def test_math_scoring_does_not_extract_numbers_from_unfinished_reasoning() -> None:
    suite = importlib.import_module("quality_suite")
    assert suite.numeric_answer("We have 12 apples and need 3.") is None
    assert suite.numeric_answer("Reasoning with 500.\n#### 1,250") == Decimal(1250)
    assert suite.numeric_answer("#### 7, followed by more reasoning") is None
    assert suite.numeric_answer("-3.50") == Decimal("-3.5")


def test_worker_metadata_contains_only_plain_rpc_primitives() -> None:
    worker = importlib.import_module("quality_worker")
    report = worker.runtime_metadata(False)
    assert report["torch"] == str(worker.torch.__version__)
    assert report["graphs_captured"] is False
    assert all(value is None or type(value) in (str, bool, int, float) for value in report.values())


def test_teacher_forced_loss_is_weighted_by_tokens_and_checks_artifact_bytes(tmp_path) -> None:
    analyzer = importlib.import_module("analyze_quality_model")
    windows, rows = [[1, 2], [3, 4, 5, 6]], []
    for i, (ids, logp) in enumerate(zip(windows, ([-1.0], [-2.0, -2.0, -2.0]), strict=True)):
        file = tmp_path / f"window-{i}.f32"
        file.write_bytes(np.array(logp, dtype="<f4").tobytes())
        rows.append(
            {
                "index": i,
                "file": file.name,
                "tokens": len(logp),
                "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                "prompt_sha256": hashlib.sha256(
                    json.dumps(ids, separators=(",", ":")).encode()
                ).hexdigest(),
            }
        )
    protocol = {"wiki_windows": windows, "wiki_scored_tokens": 4}
    path = tmp_path / "control-heldout.json"
    path.write_text(json.dumps({"status": "complete", "windows": rows}))
    assert analyzer.heldout_values(tmp_path, "control", protocol)["mean_nll"] == 1.75
    (tmp_path / "window-1.f32").write_bytes(b"invalid")
    with pytest.raises(ValueError, match="bytes changed"):
        analyzer.heldout_values(tmp_path, "control", protocol)


def test_matching_logprobs_on_different_tokens_cannot_be_scored(tmp_path) -> None:
    analyzer = importlib.import_module("analyze_quality_model")
    (tmp_path / "control-heldout.json").write_text(
        json.dumps({"status": "complete", "windows": [{"index": 0, "prompt_sha256": "different"}]})
    )
    with pytest.raises(ValueError, match="conditioning changed"):
        analyzer.heldout_values(
            tmp_path, "control", {"wiki_windows": [[1, 2]], "wiki_scored_tokens": 1}
        )


def test_code_semantics_do_not_override_an_explicit_output_format_failure(tmp_path) -> None:
    analyzer = importlib.import_module("analyze_quality_model")
    case = {"name": "merge_intervals", "kind": "code", "prompt": "Return only code."}
    root = tmp_path / "candidate"
    root.mkdir()
    (root / "real-work.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "cases": [
                    {
                        "name": case["name"],
                        "prompt_sha256": hashlib.sha256(case["prompt"].encode()).hexdigest(),
                        "passed": True,
                        "sandbox_pending": False,
                        "format_passed": False,
                    }
                ],
            }
        )
    )
    (row,) = analyzer.task_scores(tmp_path, "candidate", {"tasks": [case]})
    assert row["semantic_passed"] is True and row["passed"] is False


def test_actual_quality_launch_selects_stock_reference_and_qualified_u8(monkeypatch) -> None:
    runner = importlib.import_module("run_quality_model")
    modes = []

    def fake_engine(rank, mode, *args, **kwargs):
        modes.append(mode)
        command = [
            "serve",
            "/model",
            "--worker-extension-cls",
            "old.Worker",
            "--compilation-config",
            "{}",
        ]
        return copy.deepcopy(
            (
                ["docker", "run", "--name", "old", "--entrypoint", "python", *command],
                {"command": command, "environment": {}, "mounts": []},
            )
        )

    monkeypatch.setattr(runner.c, "engine", fake_engine)
    control = runner.launch(0, "qualitytest", "control", {}, 154880, "manifest")
    candidate = runner.launch(0, "qualitytest", "candidate", {}, 154880, "manifest")
    assert modes == ["stock", "kda"]
    assert control[1]["command"] == candidate[1]["command"]
    assert control[1]["environment"]["MXWAVE_MOE_MODE"] == "stock"
    assert candidate[1]["environment"]["MXWAVE_MOE_MODE"] == "aligned"
    assert control[1]["environment"]["MXWAVE_GLM53_VOCAB_MODE"] == "stock"
    assert candidate[1]["environment"]["MXWAVE_GLM53_VOCAB_MODE"] == "packed"
    assert "quality-best-v1-native" in candidate[1]["environment"]["MXWAVE_MOE_ALIGNED_LIBRARY"]
    for argv, meta in (control, candidate):
        assert argv[argv.index("--tool-call-parser") + 1] == "glm47"
        assert "--enable-auto-tool-choice" in argv
        assert (
            meta["command"][meta["command"].index("--worker-extension-cls") + 1]
            == "quality_worker.QualityWorker"
        )
    assert control[1]["name"].endswith("-raw-r0")
    assert candidate[1]["name"].endswith("-aligned-r0")


def test_api_failures_in_both_arms_cannot_qualify_as_matching_quality(
    tmp_path, monkeypatch
) -> None:
    analyzer = importlib.import_module("analyze_quality_model")
    protocol_path = tmp_path / "quality-protocol.json"
    protocol_path.write_text(json.dumps({"limits": []}))
    for label in ("control", "candidate"):
        for mode in ("graph", "eager"):
            for suffix in ("", "-repeat"):
                (tmp_path / f"{label}-{mode}{suffix}-decode-quality.json").write_text("{}")
        sources = [
            {
                "tp_rank": rank,
                "status": "passed",
                "scale_groups": 84,
                "source_scale_hashes": [{"shape": [1], "sha256": str(i)} for i in range(84)],
                "allocation": {"cuda_allocated_bytes": 100},
            }
            for rank in (0, 1)
        ]
        measurement = {
            "status": "collected_pending_code_sandbox",
            "quality_protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
            "source_diagnostic": {"all_source_checks": True},
            "source_before": sources,
            "source_after": sources,
        }
        (tmp_path / (label + "-quality-measurement.json")).write_text(json.dumps(measurement))
    good = {
        "all_decode_conditioning_matches": True,
        "mean_decode_kl": 0.0,
        "mean_decode_nll_difference": 0.0,
    }
    monkeypatch.setattr(analyzer, "compare", lambda *a: good)
    monkeypatch.setattr(analyzer, "bitwise", lambda *a: True)
    monkeypatch.setattr(analyzer, "heldout_values", lambda *a: {"mean_nll": 1.0})
    monkeypatch.setattr(analyzer, "trace_review", lambda *a: None)
    row = {"name": "tool", "kind": "tool", "passed": False, "inference_completed": False}
    monkeypatch.setattr(analyzer, "task_scores", lambda *a: [row])
    failed = analyzer.review(tmp_path)
    assert failed["practical"]["no_reference_pass_lost"] is True
    assert failed["practical"]["all_inference_requests_completed"] is False
    assert failed["quality_qualified"] is False
    row["inference_completed"] = True
    assert analyzer.review(tmp_path)["quality_qualified"] is True
