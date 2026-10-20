"""Exercise the emitted client's actual idle parser before another GPU load."""

from __future__ import annotations

import ast
import re
from types import SimpleNamespace

import pytest

from integrations.glm53_mxfp4.prepare_drafter_verification import MEASURE


def client(metrics: str) -> dict:
    """Load just the emitted parser and idle check without remote dependencies."""
    tree = ast.parse(MEASURE)
    tree.body = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in ("request_counts", "idle")
    ]
    namespace = {
        "re": re,
        "m": SimpleNamespace(request=lambda endpoint: metrics.encode()),
    }
    exec(compile(tree, "<emitted-measure-features>", "exec"), namespace)  # noqa: S102
    return namespace


@pytest.mark.parametrize("labels", ["", '{model_name="glm-5.3-flash",engine="0"}'])
def test_idle_accepts_labeled_and_unlabeled_request_metrics(labels: str) -> None:
    metrics = (
        "# HELP vllm:num_requests_running Number of running requests\n"
        f"vllm:num_requests_running{labels} 0.0\n"
        f"vllm:num_requests_waiting{labels} 0.0\n"
        "vllm:num_requests_running_total 123\n"
    )
    generated = client(metrics)
    assert generated["request_counts"](metrics) == [0, 0]
    generated["idle"]()


@pytest.mark.parametrize("kind", ["running", "waiting"])
def test_idle_rejects_active_requests(kind: str) -> None:
    with pytest.raises(AssertionError, match="active requests"):
        client(f'vllm:num_requests_{kind}{{engine="0"}} 1.0\n')["idle"]()


def test_idle_distinguishes_missing_metrics_from_busy_service() -> None:
    with pytest.raises(AssertionError, match="metrics missing"):
        client("# HELP vllm:num_requests_running Number of requests\n")["idle"]()
