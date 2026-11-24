"""Qualify deterministic-dispatch startup warming in a disposable GPU process."""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from stable_dispatch_warmup import specializations

PROBES = (1536, 518, 1280, 263, 2048)


def protocol() -> dict[str, Any]:
    """Declare the complete component scope before observing either GPU result."""
    return {
        "max_tokens": 2048, "expected_specializations": 34,
        "probes": list(PROBES), "post_trim_probe": 2048,
        "fresh_process_per_rank": [0, 1], "fresh_cache_required": True,
        "maximum_warm_call_seconds": 0.1, "component_timeout_seconds": 600,
        "container_CPU_limit": 2, "container_memory_bytes": 4 * 1024**3,
        "all_integer_outputs_and_guards_must_match": True,
        "no_additional_canonicalize_cache_entries": True,
        "no_persistent_fixture_CUDA_allocation": True,
        "model_quality_or_serving_speed_qualified": False,
        "cases": specializations(),
    }


def cpu_memory() -> dict[str, int]:
    """Read this disposable process's resident counters without combining CUDA totals."""
    wanted = {"VmRSS", "VmHWM", "RssAnon", "RssFile", "RssShmem"}
    return {
        key: int(value.split()[0]) * 1024
        for line in Path("/proc/self/status").read_text().splitlines()
        for key, _, value in [line.partition(":")]
        if key in wanted
    }


def cache_inventory(root: Path) -> list[dict[str, Any]]:
    """Bind compiler outputs and timestamps so subsequent compilation cannot be hidden."""
    return [
        {"path": str(p.relative_to(root)), "bytes": p.stat().st_size,
         "mtime_ns": p.stat().st_mtime_ns,
         "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in sorted(root.rglob("_canonicalize.*")) if p.is_file()
    ]


def assess(report: dict[str, Any]) -> dict[str, Any]:
    """Evaluate exact coverage, cache reuse, memory ownership and the frozen warm-call bound."""
    policy = protocol()
    if report["protocol"] != policy:
        raise ValueError("The prospective warmup protocol changed")
    runs = report["runs"]
    expected_limits = [2048, *PROBES, 2048]
    if [r["warmup"]["max_tokens"] for r in runs] != expected_limits:
        raise ValueError("Missing or reordered warmup probes")
    checks = {
        "fresh_cache": not report["cache_before"],
        "all_probes_complete": all(r["warmup"]["status"] == "complete" for r in runs),
        "complete_specialization_coverage": all(
            [{key: case[key] for key in ("block", "capacity", "first_tokens",
                                         "last_tokens", "padded_tokens")}
             for case in run["warmup"]["specializations"]]
            == specializations(run["warmup"]["max_tokens"])
            for run in runs
        ),
        "exact_integer_guards_and_metadata": all(
            c["guards_and_metadata_exact"] and c["sorted_integer_ids_exact"]
            and c["scratch_pointers_aligned_to_256_bytes"]
            for r in runs for c in r["warmup"]["specializations"]
        ),
        "no_persistent_fixture_CUDA_allocation": all(
            r["warmup"]["cuda_allocated_before"] == r["warmup"]["cuda_allocated_after"]
            for r in runs
        ),
        "exactly_34_compiled_variants": sum(
            Path(x["path"]).suffix == ".json" for x in runs[0]["cache_after"]
        ) == policy["expected_specializations"],
        "no_subsequent_cache_change": all(
            r["cache_after"] == runs[0]["cache_after"] for r in runs[1:]
        ),
        "kernel_sources_unchanged": all(
            r["warmup"]["source_sha256"] == runs[0]["warmup"]["source_sha256"]
            and not r["warmup"]["existing_kernel_implementation_changed"]
            and not r["warmup"]["model_weights_or_live_routing_touched"] for r in runs
        ),
        "trim_return_valid": report["trim"]["returned"] in (0, 1),
    }
    warm_calls = [value for r in runs[1:] for c in r["warmup"]["specializations"]
                  for value in [c["first_call_seconds"], *c["repeat_seconds"]]]
    maximum = max(warm_calls)
    checks["warm_call_bound_passed"] = maximum <= policy["maximum_warm_call_seconds"]
    return {"checks": checks, "component_passed": all(checks.values()),
            "maximum_warm_call_seconds": maximum,
            "cold_warmup_seconds": runs[0]["warmup"]["seconds"],
            "CPU_RSS_change_after_cold_warmup_bytes": (
                runs[0]["CPU_after"]["VmRSS"] - runs[0]["CPU_before"]["VmRSS"]),
            "CPU_RSS_reclaimed_by_trim_bytes": (
                report["trim"]["CPU_before"]["VmRSS"]
                - report["trim"]["CPU_after"]["VmRSS"]),
            "model_quality_or_serving_speed_qualified": False}


def main() -> None:
    """Measure cold compilation, representative reuse and one optional startup heap trim."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.describe:
        print(json.dumps(protocol()))
        return
    if args.report is None or args.report.exists():
        raise ValueError("Require a fresh report path")
    import torch
    from stable_dispatch_warmup import warmup

    root = Path(os.environ["TRITON_CACHE_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "status": "running", "started_epoch": time.time(), "pid": os.getpid(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol": protocol(), "runs": [], "cache_before": cache_inventory(root),
        "model_loaded": False, "model_CPU_saving_demonstrated": False,
    }

    def save() -> None:
        temporary = args.report.with_suffix(".incomplete")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.report)

    def measure(limit: int, stage: str) -> None:
        row = {"stage": stage, "CPU_before": cpu_memory(),
               "CUDA_reserved_before": torch.cuda.memory_reserved()}
        row["warmup"] = warmup(limit)
        row.update(CPU_after=cpu_memory(), CUDA_reserved_after=torch.cuda.memory_reserved(),
                   cache_after=cache_inventory(root))
        report["runs"].append(row)
        save()

    try:
        if report["cache_before"]:
            raise RuntimeError("Cold-cache component requires an empty canonicalize cache")
        torch.cuda.init()
        report["GPU"] = torch.cuda.get_device_name()
        report["torch_version"] = torch.__version__
        measure(2048, "cold")
        for limit in PROBES:
            measure(limit, "reuse")
        libc = ctypes.CDLL(None)
        libc.gnu_get_libc_version.argtypes = []
        libc.gnu_get_libc_version.restype = ctypes.c_char_p
        libc.malloc_trim.argtypes = [ctypes.c_size_t]
        libc.malloc_trim.restype = ctypes.c_int
        gc.collect()
        before = cpu_memory()
        started = time.perf_counter()
        returned = libc.malloc_trim(0)
        report["trim"] = {"glibc": libc.gnu_get_libc_version().decode(),
                          "returned": returned, "seconds": time.perf_counter() - started,
                          "CPU_before": before, "CPU_after": cpu_memory()}
        measure(2048, "after_trim")
        report.update(status="complete", finished_epoch=time.time())
        report["assessment"] = assess(report)
        save()
        print(json.dumps(report["assessment"]))
    except BaseException as exc:
        report.update(status="failed", error=repr(exc), finished_epoch=time.time())
        save()
        raise


if __name__ == "__main__":
    main()
