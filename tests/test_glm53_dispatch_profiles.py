"""Host attribution must not count nested runtime/driver waits twice."""

from __future__ import annotations

import gzip
import importlib
import json
from pathlib import Path

import pytest


def test_residual_excludes_nested_cuda_calls_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))
    module = importlib.import_module("analyze_dispatch_profiles")
    parent = {"pid": 1, "tid": 2, "ts": 0., "dur": 100.}
    children = [parent, {"pid": 1, "tid": 2, "ts": 10., "dur": 40.},
                {"pid": 1, "tid": 2, "ts": 20., "dur": 20.},
                {"pid": 1, "tid": 2, "ts": 60., "dur": 10.},
                {"pid": 1, "tid": 3, "ts": 0., "dur": 100.},
                {"pid": 1, "tid": 2, "ts": 90., "dur": 20.}]
    assert module.residual_us(parent, children) == 50.


def test_interval_union_retains_gaps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))
    module = importlib.import_module("analyze_dispatch_profiles")
    assert module.interval_union_us([(20., 40.), (0., 10.), (5., 8.), (8., 12.)]) == 32.


def test_only_complete_forward_stream_is_attributed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))
    module = importlib.import_module("analyze_dispatch_profiles")
    name = "execute_4_context_0_generation_4"
    events = [
        {"cat": "user_annotation", "ph": "X", "name": name, "ts": 0, "dur": 1000,
         "pid": 1, "tid": 2, "args": {"External id": 100}},
        {"cat": "cpu_op", "ph": "X", "name": "mxwave_glm53::kda_linear", "ts": 10,
         "dur": 100, "pid": 1, "tid": 2, "args": {"External id": 1}},
        {"cat": "cpu_op", "ph": "X", "name": "aten::fill_", "ts": 20,
         "dur": 10, "pid": 1, "tid": 2, "args": {"External id": 3}},
    ]
    for stream, count in ((17, 84), (18, 1)):
        events.append({"cat": "gpu_user_annotation", "ph": "X", "name": name,
            "ts": 1500, "dur": 200, "pid": 0, "tid": stream, "args": {"External id": 100}})
        events.extend({"cat": "kernel", "ph": "X", "name": "mxwave_grouped_aligned",
            "ts": 1520 + i, "dur": 1, "pid": 0, "tid": stream,
            "args": {"stream": stream}} for i in range(count))
    events.append({"cat": "kernel", "ph": "X", "name": "fill_helper", "ts": 1501,
        "dur": 2, "pid": 0, "tid": 17, "args": {"stream": 17, "External id": 3}})
    path = tmp_path / "trace.gz"
    with gzip.open(path, "wt") as handle:
        json.dump({"traceEvents": events}, handle)
    result = module.inspect(path)
    assert result["complete_forward_samples"] == 1
    row = result["shape_summary"][0]
    assert row["shape"] == [4, 0, 4]
    assert row["custom_scopes"]["mxwave_glm53::kda_linear"]["calls"] == 1
    assert row["kda_gpu_helpers"]["kernel/aten::fill_"]["calls"] == 1
    assert row["kda_gpu_helpers"]["kernel/aten::fill_"]["sum_us"] == 2
