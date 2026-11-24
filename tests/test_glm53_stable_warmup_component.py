"""Reject incomplete coverage, changed caches and metadata corruption in warmup evidence."""

from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest


@pytest.fixture
def benchmark(monkeypatch):
    """Load the CPU evidence evaluator without importing a GPU framework."""
    root = Path(__file__).parents[1] / "integrations/glm53_mxfp4"
    monkeypatch.syspath_prepend(str(root))
    spec = importlib.util.spec_from_file_location("warmup_component_test",
                                                root / "benchmark_stable_warmup.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def complete(benchmark):
    """Build a finite evidence fixture with cold compilation and reused warm kernels."""
    cache = [{"path": f"variant{i}/_canonicalize.json", "sha256": str(i), "mtime_ns": 42,
              "bytes": 12} for i in range(34)]
    runs = []
    for limit in (2048, 1536, 518, 1280, 263, 2048, 2048):
        cases = [{**c, "guards_and_metadata_exact": True, "sorted_integer_ids_exact": True,
                  "scratch_pointers_aligned_to_256_bytes": True, "first_call_seconds": 0.001,
                  "repeat_seconds": [0.001, 0.001, 0.001]}
                 for c in benchmark.specializations(limit)]
        runs.append({"CPU_before": {"VmRSS": 1000}, "CPU_after": {"VmRSS": 2000},
                     "cache_after": deepcopy(cache), "warmup": {
                         "max_tokens": limit, "status": "complete", "specializations": cases,
                         "source_sha256": {"kernel": "unchanged"}, "seconds": 1.0,
                         "cuda_allocated_before": 0, "cuda_allocated_after": 0,
                         "existing_kernel_implementation_changed": False,
                         "model_weights_or_live_routing_touched": False}})
    return {"protocol": benchmark.protocol(), "cache_before": [], "runs": runs,
            "trim": {"returned": 1, "CPU_before": {"VmRSS": 3000},
                     "CPU_after": {"VmRSS": 2000}}}


def test_valid_component_evidence_does_not_claim_model_qualification(benchmark, complete):
    result = benchmark.assess(complete)
    assert result["component_passed"]
    assert result["maximum_warm_call_seconds"] == 0.001
    assert not result["model_quality_or_serving_speed_qualified"]


@pytest.mark.parametrize("failure", ["guard", "allocation", "new_cache", "rewritten_cache",
                                     "missing_case", "slow_reuse", "cold_cache_populated"])
def test_corrupted_or_unwarmed_evidence_fails(benchmark, complete, failure):
    run = complete["runs"][1]
    if failure == "guard":
        run["warmup"]["specializations"][0]["guards_and_metadata_exact"] = False
    elif failure == "allocation":
        run["warmup"]["cuda_allocated_after"] = 4096
    elif failure == "new_cache":
        run["cache_after"].append({"path": "unexpected/_canonicalize.json"})
    elif failure == "rewritten_cache":
        run["cache_after"][0]["mtime_ns"] += 1
    elif failure == "missing_case":
        run["warmup"]["specializations"].pop()
    elif failure == "slow_reuse":
        run["warmup"]["specializations"][0]["first_call_seconds"] = 1.5
    else:
        complete["cache_before"] = [{"path": "old/_canonicalize.json"}]
    assert not benchmark.assess(complete)["component_passed"]


def test_missing_probe_and_changed_gate_are_rejected(benchmark, complete):
    changed = deepcopy(complete)
    changed["protocol"]["maximum_warm_call_seconds"] = 2.0
    with pytest.raises(ValueError, match="protocol"):
        benchmark.assess(changed)
    complete["runs"].pop()
    with pytest.raises(ValueError, match="Missing or reordered"):
        benchmark.assess(complete)
