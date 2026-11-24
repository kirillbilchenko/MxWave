"""Keep startup resource conditioning separate from full-model qualification."""

from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest


@pytest.fixture
def worker(monkeypatch):
    """Load the report validator without importing Torch or a native serving worker."""
    root = Path(__file__).parents[1] / "integrations/glm53_mxfp4"
    monkeypatch.syspath_prepend(str(root))
    spec = importlib.util.spec_from_file_location("warmed_model_worker_test",
                                                root / "startup_preparation.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def complete(worker):
    """Represent a nonzero live model allocation preserved across all scratch operations."""
    cases = [{**case, "guards_and_metadata_exact": True, "sorted_integer_ids_exact": True,
              "scratch_pointers_aligned_to_256_bytes": True, "first_call_seconds": 0.0001,
              "repeat_seconds": [0.0001] * 3} for case in worker.specializations()]
    run = {"status": "complete", "specializations": cases, "max_tokens": 2048,
           "specialization_count": 34, "source_sha256": {"kernel": "unchanged"},
           "existing_kernel_implementation_changed": False,
           "model_weights_or_live_routing_touched": False,
           "cuda_allocated_before": 1000000, "cuda_allocated_after": 1000000}
    cache = [{"path": f"variant{i}/_canonicalize.json", "sha256": str(i)} for i in range(34)]
    return {"warmup": deepcopy(run), "warmup_after_trim": deepcopy(run),
            "CUDA_allocated_before": 1000000, "CUDA_allocated_after": 1000000,
            "cache_after_warmup": deepcopy(cache), "cache_after_trim_probe": deepcopy(cache),
            "trim": {"returned": 1}, "model_quality_or_serving_speed_qualified": False}


def test_full_coverage_preserves_live_model_allocation(worker, complete):
    assert all(worker.validate_startup_report(complete).values())
    assert not complete["model_quality_or_serving_speed_qualified"]


@pytest.mark.parametrize("failure", ["model_allocation", "scratch_allocation", "missing_variant",
                                     "guard", "source", "cache", "slow_warm_call"])
def test_reject_changed_model_state_or_incomplete_preparation(worker, complete, failure):
    repeated = complete["warmup_after_trim"]
    if failure == "model_allocation":
        complete["CUDA_allocated_after"] -= 4096
    elif failure == "scratch_allocation":
        repeated["cuda_allocated_after"] += 4096
    elif failure == "missing_variant":
        repeated["specializations"].pop()
    elif failure == "guard":
        repeated["specializations"][0]["guards_and_metadata_exact"] = False
    elif failure == "source":
        repeated["source_sha256"]["kernel"] = "changed"
    elif failure == "cache":
        complete["cache_after_trim_probe"][0]["sha256"] = "new-compilation"
    else:
        repeated["specializations"][0]["first_call_seconds"] = 2.0
    with pytest.raises(ValueError, match="Startup preparation failed"):
        worker.validate_startup_report(complete)
