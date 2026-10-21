"""Keep diagnostic launch independent of an unfinished or failed timed comparison."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from integrations.glm53_mxfp4.prepare_pretrained_verification import QUEUE
from integrations.glm53_mxfp4.pretrained_verification_probe import PretrainedVerificationCapture


class Waiting(Exception):
    """Stop a test once the queue demonstrates that it does not launch."""


def setup(tmp_path: Path, phase: str) -> tuple[dict[str, Any], Path, Path]:
    """Create an isolated parent controller and source-pinned diagnostic queue."""
    parent = tmp_path / "drafter-pretrained-modal-v1"
    parent.mkdir()
    (parent / "plan.json").write_text("{}\n")
    (parent / "controller.json").write_text(json.dumps({
        "phase": phase, "service_paused": phase != "complete_restored",
        "restoration": {"health_200": True, "originals_running": True},
    }))
    root = tmp_path / "diagnostic"
    root.mkdir()
    plan = {"parent_plan_sha256": hashlib.sha256((parent / "plan.json").read_bytes()).hexdigest(),
            "calibration_cases": [{"id": "case"}],
            "expected_outputs": {"a1": list(range(64)), "b1": list(range(64))}}
    (root / "plan.json").write_text(json.dumps(plan))
    (root / "source-manifest.json").write_text(json.dumps({
        "plan.json": hashlib.sha256((root / "plan.json").read_bytes()).hexdigest(),
    }))
    ns: dict[str, Any] = {"__file__": str(root / "wait_for_comparison.py"),
                          "__name__": "queue_test"}
    exec(compile(QUEUE, "wait_for_comparison.py", "exec"), ns)  # noqa: S102
    return ns, root, parent


@pytest.mark.parametrize("phase", ["measuring_c1", "complete_restored"])
def test_unfinished_comparison_or_finalization_cannot_launch(tmp_path: Path, phase: str) -> None:
    ns, root, _ = setup(tmp_path, phase)
    launched = []

    def wait(_: float) -> None:
        raise Waiting

    ns["time"] = SimpleNamespace(time=lambda: 1.0, sleep=wait)
    ns["subprocess"] = SimpleNamespace(run=lambda *a, **k: launched.append(a))
    with pytest.raises(Waiting):
        ns["main"]()
    assert not launched
    assert json.loads((root / "queue.json").read_text())["phase"] == (
        "waiting_for_timing_and_restoration")


def test_failed_parent_cannot_launch(tmp_path: Path) -> None:
    ns, _, _ = setup(tmp_path, "failed_restored")
    with pytest.raises(AssertionError, match="Parent failed"):
        ns["main"]()


@pytest.mark.parametrize("changed_prefix", [False, True])
def test_finalized_parent_requires_all_six_observed_prefixes(
    tmp_path: Path, changed_prefix: bool,
) -> None:
    ns, root, parent = setup(tmp_path, "complete_restored")
    (parent / "run-prepared-status.json").write_text(json.dumps({
        "step": "finalize", "returncode": 0, "control_returncode": 0,
    }))
    (parent / "final-service-review.json").write_text(json.dumps({
        "status": "verified", "health_200": True, "originals_running": True,
    }))
    for arm in ("a1", "b1"):
        rows = [{"case": "case", "result": {"token_ids": list(range(256))}} for _ in range(6)]
        if changed_prefix:
            rows[-1]["result"]["token_ids"][47] = 999
        (parent / f"{arm}-measurement.json").write_text(json.dumps({
            "status": "complete", "repeat_tokens_and_counters_exact": True, "cases": rows,
        }))
    launched = []

    def run(argv: list[str], **_: Any) -> SimpleNamespace:
        launched.append(Path(argv[-1]).name)
        (root / "controller.json").write_text(json.dumps({
            "phase": "complete_restored", "service_paused": False,
        }))
        (root / "run-prepared-status.json").write_text(json.dumps({
            "step": "finalize", "returncode": 0, "control_returncode": 0,
        }))
        return SimpleNamespace(returncode=0)

    ns["subprocess"] = SimpleNamespace(run=run, STDOUT=-2)
    if changed_prefix:
        with pytest.raises(AssertionError):
            ns["main"]()
        assert not launched
    else:
        ns["main"]()
        assert launched == ["prepare.py", "run-prepared.py"]


def test_target_graph_observation_does_not_replace_native_output() -> None:
    capture = object.__new__(PretrainedVerificationCapture)
    original = object()
    descriptor = SimpleNamespace(cg_mode="FULL", num_tokens=8, num_reqs=1)
    capture.original_target_replay = lambda d: original
    assert capture.target_replay(descriptor) is original
    assert capture.descriptor == {"cg_mode": "FULL", "num_tokens": 8, "num_reqs": 1}
