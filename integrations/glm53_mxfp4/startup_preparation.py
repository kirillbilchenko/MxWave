"""CPU-only evidence checks for full-model dispatch startup preparation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmark_stable_warmup import protocol
from stable_dispatch_warmup import specializations


def validate_startup_report(report: dict[str, Any]) -> dict[str, bool]:
    """Require component-qualified coverage and unchanged cache/allocation after startup."""
    first, repeated = report["warmup"], report["warmup_after_trim"]
    keys = ("block", "capacity", "first_tokens", "last_tokens", "padded_tokens")
    warm_bound = protocol()["maximum_warm_call_seconds"]
    checks = {
        "both_complete": first["status"] == repeated["status"] == "complete",
        "all_34_specializations": all(
            [{k: c[k] for k in keys} for c in run["specializations"]] == specializations()
            and run["max_tokens"] == 2048 and run["specialization_count"] == 34
            for run in (first, repeated)
        ),
        "exact_integer_metadata_and_guards": all(
            c["guards_and_metadata_exact"] and c["sorted_integer_ids_exact"]
            and c["scratch_pointers_aligned_to_256_bytes"]
            for run in (first, repeated) for c in run["specializations"]
        ),
        "kernel_and_sources_unchanged": first["source_sha256"] == repeated["source_sha256"]
        and all(not run["existing_kernel_implementation_changed"]
                and not run["model_weights_or_live_routing_touched"] for run in (first, repeated)),
        "no_persistent_scratch_allocation": all(
            run["cuda_allocated_before"] == run["cuda_allocated_after"]
            for run in (first, repeated)
        ),
        "model_allocation_unchanged": report["CUDA_allocated_before"]
        == report["CUDA_allocated_after"],
        "all_34_variants_cached": sum(Path(r["path"]).suffix == ".json"
                                      for r in report["cache_after_warmup"]) == 34,
        "cache_unchanged_after_trim": report["cache_after_warmup"]
        == report["cache_after_trim_probe"],
        "trim_return_valid": report["trim"]["returned"] in (0, 1),
        "warm_calls_under_component_bound": all(
            t <= warm_bound
            for c in repeated["specializations"]
            for t in (c["first_call_seconds"], *c["repeat_seconds"])
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"Startup preparation failed: {[k for k, v in checks.items() if not v]}")
    return checks
