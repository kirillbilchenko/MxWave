"""Attribute nested native calls without silently assigning unknown graph work."""

from __future__ import annotations

import gzip
import importlib
import json
from pathlib import Path


def test_native_kernel_attribution_respects_outer_kda_and_graph_assumption(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "integrations/glm53_mxfp4"))
    analyzer = importlib.import_module("analyze_combined_profiles")
    events = []
    for name, start, duration, external in (
        ("mxwave_glm53::kda_linear", 10, 20, 1),
        ("mxwave_zipserv::linear", 12, 10, 2),
        ("mxwave_zipserv::linear", 35, 10, 3),
    ):
        events.append({"cat": "cpu_op", "ph": "X", "name": name, "ts": start,
                       "dur": duration, "pid": 100, "tid": 1, "args": {"External id": external}})
    for external, graph, name in ((2, 0, "BF16TripleBitmap_MM_Kernel"),
                                  (3, 0, "BF16TripleBitmap_MM_Kernel"),
                                  (None, 4, "BF16TripleBitmap_MM_Kernel"),
                                  (None, 4, "_row_gather_kernel"),
                                  (None, 4, "mxwave_grouped_aligned")):
        events.append({"cat": "kernel", "ph": "X", "name": name, "ts": 60, "dur": 5,
                       "pid": 0, "tid": 7, "args": {"External id": external, "graph id": graph}})
    path = tmp_path / "trace.gz"
    with gzip.open(path, "wt") as handle:
        json.dump({"traceEvents": events}, handle)
    result = analyzer.attribute(path)
    counts = {row["component"]: row["calls"] for row in result["groups"]}
    assert counts == {"kda": 1, "head": 1, "unattributed": 1, "embedding": 1, "expert_scales": 1}
    result = analyzer.attribute(path, model_only_graphs=True)
    counts = {row["component"]: row["calls"] for row in result["groups"]}
    assert counts == {"kda": 2, "head": 1, "embedding": 1, "expert_scales": 1}
