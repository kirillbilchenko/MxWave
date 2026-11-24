"""A failed candidate must not invalidate a completed, hash-bound stock control."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def runner(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Import the controller without vLLM or GPU dependencies."""
    root = Path(__file__).parents[1] / "integrations/glm53_mxfp4"
    monkeypatch.syspath_prepend(str(root))
    spec = importlib.util.spec_from_file_location("glm53_combined_control_test", root / "run_kda_model.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def saved(tmp_path: Path) -> tuple[dict, dict, dict]:
    """Retain sixteen four-step controls plus source/chat qualification and identities."""
    current = {"originals": ["rank0", "rank1"],
               "preflight": {"source_metadata": {"source": "fixed"}, "libraries": {"native": "fixed"}},
               "launches": {"kda": [{"environment": {"GLM_ROCE_ALLREDUCE": "1"}}] * 2}}
    previous = {**copy.deepcopy(current), "phase": "failed_restored", "service_paused": False,
        "restoration": {"status": "complete"},
        "arms": [{"mode": "stock", "status": "complete"}, {"mode": "kda"}]}
    previous["launches"] = {"stock": copy.deepcopy(current["launches"]["kda"])}
    protocol = {"quality_prefix_token_ids": [7], "quality": [[1, 2]] * 16}
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    diagnostic = tmp_path / "combined-diagnostic"
    diagnostic.mkdir()
    (diagnostic / "model-diagnostic.json").write_text(json.dumps({
        "status": "complete", "phases": [{"sanity_passed": True}] * 3,
        "loaded_weight_checks": [{"status": "passed"}] * 2}))
    raw = b"\0" * (154880 * 4)
    (tmp_path / "step.f32").write_bytes(raw)
    prompt = hashlib.sha256(json.dumps([7, 1, 2], separators=(",", ":")).encode()).hexdigest()
    contexts = [{"index": i, "prompt_sha256": prompt, "token_ids": [3] * 4,
        "steps": [{"position": j, "kind": "decode" if j else "prefill",
            "conditioning_token_ids": [3] * j, "file": "step.f32",
            "sha256": hashlib.sha256(raw).hexdigest()} for j in range(4)]} for i in range(16)]
    for label in ("stock", "stock-repeat"):
        (tmp_path / (label + "-decode-quality.json")).write_text(json.dumps({
            "status": "complete", "vocabulary": 154880, "generated_tokens": 4,
            "contexts": contexts}))
    return previous, current, protocol


def test_completed_stock_survives_candidate_failure(
    runner: ModuleType, saved: tuple[dict, dict, dict], tmp_path: Path,
) -> None:
    previous, current, protocol = saved
    hashes = runner.validate_decode_control(previous, current, tmp_path, protocol)
    assert set(hashes) == {"stock", "stock-repeat"}
    assert all(len(value) == 64 for value in hashes.values())


@pytest.mark.parametrize("fault", ["source", "unfinished", "prompt", "bytes", "transport"])
def test_reuse_refuses_changed_or_incomplete_evidence(
    runner: ModuleType, saved: tuple[dict, dict, dict], tmp_path: Path, fault: str,
) -> None:
    previous, current, protocol = saved
    if fault == "source":
        previous["preflight"]["source_metadata"]["source"] = "changed"
    elif fault == "unfinished":
        previous["arms"][0]["status"] = "running"
    elif fault == "prompt":
        path = tmp_path / "stock-decode-quality.json"
        value = json.loads(path.read_bytes())
        value["contexts"][0]["prompt_sha256"] = "changed"
        path.write_text(json.dumps(value))
    elif fault == "transport":
        current["launches"]["kda"][1]["environment"]["GLM_ROCE_PROXY_CPUS"] = "auto"
    else:
        (tmp_path / "step.f32").write_bytes(b"changed")
    with pytest.raises(ValueError):
        runner.validate_decode_control(previous, current, tmp_path, protocol)


def test_dispatch_qualification_binds_code_libraries_and_all_shapes(runner: ModuleType) -> None:
    bound = {"code": {name: "fixed" for name in ("model_plugin.py", "moe_runtime.py")},
             "libraries": {f"libmxwave_moe_{v}.so": v for v in ("raw", "aligned")}}
    cases = [{"tokens": m, "concentrated": c, "dispatch_modes": ["custom", "dispatcher"],
              "stock_eager_stream_graph_bits_exact": True}
             for m in (1, 4, 8, 64, 128) for c in (False, True)]
    reports = [{"mode": v, "status": "complete", "code_sha256": dict(bound["code"]),
                "library_sha256": {f"MXWAVE_MOE_{x.upper()}_LIBRARY": x
                                   for x in ("raw", "aligned")}, "cases": copy.deepcopy(cases)}
               for v in ("raw", "aligned")]
    fixture = {"phase": "complete_idle", "dispatch_comparison": True}
    runner.validate_moe_dispatch_fixture(fixture, reports, bound)
    for fault in ("code", "library", "outputs", "shapes", "backend"):
        changed = copy.deepcopy(reports)
        if fault == "code":
            changed[0]["code_sha256"]["model_plugin.py"] = "different"
        elif fault == "library":
            changed[0]["library_sha256"]["MXWAVE_MOE_RAW_LIBRARY"] = "different"
        elif fault == "outputs":
            changed[0]["cases"][0]["stock_eager_stream_graph_bits_exact"] = False
        elif fault == "shapes":
            changed[0]["cases"].pop()
        else:
            changed[0]["cases"][0]["dispatch_modes"] = ["custom"]
        with pytest.raises(ValueError):
            runner.validate_moe_dispatch_fixture(fixture, changed, bound)
