"""Model qualification must observe the intended route and preserve failed gates."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def source_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Import the CPU model protocol without initializing CUDA."""
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))


def test_launcher_changes_only_input_policy_and_keeps_the_same_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("run_fp32_input_model")

    def original(*args: Any) -> tuple[list[str], dict[str, Any]]:
        command = ["serve", "--worker-extension-cls", "warmed_model_worker.WarmedModelWorker",
                   "--max-num-seqs", "16", "--kv-cache-memory-bytes", "4294967296"]
        return ["docker", "run", "--entrypoint", "python3", "image", *command], {
            "command": command, "environment": {"unchanged": "yes"}}

    monkeypatch.setattr(module.warmed, "launch", original)
    for arm in ("control", "candidate"):
        argv, meta = module.launch(0, "test", arm, {}, 4, "manifest")
        assert meta["environment"]["unchanged"] == "yes"
        assert meta["environment"]["MXWAVE_GLM53_KDA_INPUT_FP32_SPLIT2"] == (
            "1" if arm == "candidate" else "0")
        assert ("MXWAVE_GLM53_FP32_LIBRARY" in meta["environment"]) == (arm == "candidate")
        for command in (argv, meta["command"]):
            assert command[command.index("--worker-extension-cls") + 1] == (
                "fp32_input_worker.FP32InputWorker")
            assert command[command.index("--max-num-seqs") + 1] == "16"
            assert command[command.index("--kv-cache-memory-bytes") + 1] == "4294967296"
        assert "--speculative-config" not in argv
    with pytest.raises(ValueError):
        module.launch(0, "test", "undeclared", {}, 4, "manifest")


def profiles() -> list[dict[str, Any]]:
    """Represent both-rank graph replays with distinct input, output and reduction kernels."""
    result = []
    for arm in ("control", "candidate"):
        for rank in (0, 1):
            for n in (8, 16):
                label = f"{arm}-c{n}-profile"
                # Geometry observed in the retained fp32inputmodelv1 worker traces.
                variants = [
                    {"name": "BF16TripleBitmap_MM_Kernel", "count": 34, "grid": [1, 64, 1]},
                    {"name": "BF16TripleBitmap_MM_FP32_Kernel", "count": 34,
                     "grid": [1, 394, 1]},
                    {"name": "MxWaveReduceFP32", "count": 34,
                     "grid": [12608 * n // 256, 1, 1]},
                ]
                result.append({"rank": rank, "labels": [label], "steps": [{"label": label,
                    "sequence": 0, "batch": {"has_prefill": False, "num_tokens": n,
                                              "num_tokens_after_padding": n},
                    "cuda_graph_launches": 1,
                    "kernel_variants": variants if arm == "candidate" else []}]})
    return result


@pytest.mark.parametrize("fault", ["missing_reducer", "original_input", "wrong_geometry",
                                   "split1_geometry", "split4_geometry", "wrong_output",
                                   "no_graph", "reconstruction", "missing_rank"])
def test_profile_cannot_pass_with_wrong_route_or_incomplete_coverage(fault: str) -> None:
    module = importlib.import_module("analyze_fp32_input_model")
    values = profiles()
    assert len(module.verify_decode_mechanism(values)) == 8
    step = values[-1]["steps"][0]
    if fault == "missing_reducer":
        step["kernel_variants"].pop()
    elif fault == "original_input":
        step["kernel_variants"][1]["name"] = "BF16TripleBitmap_MM_Kernel"
    elif fault == "wrong_geometry":
        step["kernel_variants"][1]["grid"][1] = 64
    elif fault in ("split1_geometry", "split4_geometry"):
        step["kernel_variants"][1]["grid"][1] = 197 if fault == "split1_geometry" else 788
    elif fault == "wrong_output":
        step["kernel_variants"][0]["grid"][1] = 197
    elif fault == "no_graph":
        step["cuda_graph_launches"] = 0
    elif fault == "reconstruction":
        step["kernel_variants"].append({"name": "BF16TripleBitmap_Decompress_Kernel",
                                       "count": 1, "grid": [1, 1, 1]})
    else:
        values.pop()
    with pytest.raises(ValueError):
        module.verify_decode_mechanism(values)


def model_fixture(module: Any, enabled: bool) -> dict[str, Any]:
    """Observe the same frozen policy on both workers at each resource measurement."""
    return {stage: [{"rank": rank, "fp32_input_policy": {**module.policy(), "enabled": enabled}}
                    for rank in (0, 1)]
            for stage in ("snapshot_before", "snapshot_before_profiles", "snapshot_after")}


@pytest.mark.parametrize("fault", ["duplicate_rank", "candidate_disabled", "policy_expanded"])
def test_model_policy_requires_every_rank_and_stage(fault: str) -> None:
    module = importlib.import_module("analyze_fp32_input_model")
    value = model_fixture(module, True)
    module.validate_policy(value, "candidate")
    last = value["snapshot_after"][1]
    if fault == "duplicate_rank":
        last["rank"] = 0
    elif fault == "candidate_disabled":
        last["fp32_input_policy"]["enabled"] = False
    else:
        last["fp32_input_policy"]["physical_rows"].append(32)
    with pytest.raises(ValueError):
        module.validate_policy(value, "candidate")


def test_input_policy_attestation_does_not_override_failed_model_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("analyze_fp32_input_model")
    plan = {"fp32_input_policy": module.policy(),
            "fp32_input_integration_archive_sha256": module.INTEGRATION_ARCHIVE_SHA256}
    (tmp_path / "warmed-model-plan.json").write_text(json.dumps(plan))
    for arm in ("control", "candidate"):
        (tmp_path / f"{arm}-warmed-model.json").write_text(
            json.dumps(model_fixture(module, arm == "candidate")))
    failed = {"compression_numerical_gate_passed": False, "observed_speed_gate_passed": False,
              "automatic_promotion_allowed": False}
    monkeypatch.setattr(module.warmed, "review", lambda root: failed.copy())
    result = module.review(tmp_path)
    assert result["fp32_input_policy_checked_on_both_ranks"]
    assert not result["compression_numerical_gate_passed"]
    assert not result["observed_speed_gate_passed"]
    assert not result["automatic_promotion_allowed"]
