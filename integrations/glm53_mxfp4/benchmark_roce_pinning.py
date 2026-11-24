"""Compare RDMA proxy placement in an isolated TP2 process pair, without model weights."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

ORDER = ("off", "auto", "auto", "off", "auto", "off", "off", "auto")
CASES = (
    {"name": "single_graph", "graph": True, "repeats": 1, "spacing": "none", "trials": 48},
    {"name": "single_graph_spaced", "graph": True, "repeats": 1,
     "spacing": "gpu", "trials": 48},
    {"name": "single_host_spaced", "graph": False, "repeats": 1,
     "spacing": "host", "trials": 48},
    {"name": "burst_graph", "graph": True, "repeats": 91, "spacing": "none", "trials": 8},
    {"name": "model_spaced_graph", "graph": True, "repeats": 91,
     "spacing": "gpu", "trials": 8},
    {"name": "model_host_spaced", "graph": False, "repeats": 91,
     "spacing": "host", "trials": 8},
)


def percentile(values: list[float], fraction: float) -> float:
    """Use a fixed nearest-rank percentile for all arms, including small sample sets."""
    import math

    if not values or not 0 < fraction <= 1:
        raise ValueError("Require nonempty samples and a percentile in (0, 1]")
    return sorted(values)[max(0, math.ceil(fraction * len(values)) - 1)]


def review_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Require both complete orders and exact bytes before evaluating the declared lead gates."""
    if len(reports) != 2 or {r.get("rank") for r in reports} != {0, 1}:
        raise ValueError("Require both experiment ranks")
    summaries = []
    for report in reports:
        records = report["records"]
        expected = {(block, case["name"]) for block in range(len(ORDER)) for case in CASES}
        if (report.get("status") != "complete" or len(records) != len(expected)
                or {(r["block"], r["case"]) for r in records} != expected
                or report.get("runtime", {}).get("error_seq") != 0):
            raise ValueError("Incomplete or unhealthy two-order communication evidence")
        if any(not r.get("bits_exact") or r["mode"] != ORDER[r["block"]] for r in records):
            raise ValueError("Incorrect bytes or treatment order")
        for record in records:
            affinity = record["observed_proxy_affinity"]
            if (record["mode"] == "off" and affinity != report["original_proxy_affinity"]
                    or record["mode"] == "auto" and len(affinity) != 1):
                raise ValueError("Proxy placement was not observed")
        for case in CASES:
            for group in range(2):
                selected = [r for r in records if r["case"] == case["name"]
                            and r["block"] // 4 == group]
                arms = {mode: {metric: statistics.median(r[metric] for r in selected
                                                    if r["mode"] == mode)
                               for metric in ("gpu_median_us", "gpu_p95_us",
                                              "wall_median_us", "wall_p95_us")}
                        for mode in ("off", "auto")}
                summaries.append({"rank": report["rank"], "case": case["name"],
                    "order_group": group, "arms": arms,
                    "reductions": {metric: 1 - arms["auto"][metric] / arms["off"][metric]
                                   for metric in arms["off"]}})
    if (len({r["source_sha256"] for r in reports}) != 1
            or len({row["output_sha256"] for r in reports for row in r["records"]}) != 1):
        raise ValueError("Ranks used different code or output bytes")
    primary = [r for r in summaries if r["case"] == "single_graph_spaced"]
    model = [r for r in summaries if r["case"] == "model_spaced_graph"]
    component = all(r["reductions"]["gpu_p95_us"] >= .05
                    and r["reductions"]["gpu_median_us"] >= -.02 for r in primary)
    followup = component and all(r["reductions"]["wall_median_us"] >= .01 for r in model)
    return {"status": "complete", "paired_summary": summaries,
            "component_lead_gate_passed": component, "full_model_followup_gate_passed": followup,
            "interpretation": "Synthetic communication lead; not model throughput or quality"}


def task_snapshot(tid: int) -> dict[str, Any]:
    """Read scheduling counters for one thread belonging to this experiment process."""
    root = Path(f"/proc/self/task/{tid}")
    values = [int(v) for v in (root / "schedstat").read_text().split()]
    status = dict(line.split(":", 1) for line in (root / "status").read_text().splitlines()
                  if ":" in line)
    return {"tid": tid, "affinity": sorted(os.sched_getaffinity(tid)),
            "runtime_ns": values[0], "runqueue_wait_ns": values[1], "timeslices": values[2],
            "voluntary_switches": int(status["voluntary_ctxt_switches"]),
            "involuntary_switches": int(status["nonvoluntary_ctxt_switches"])}


def telemetry() -> dict[str, Any]:
    """Sample clocks and pressure outside measured regions; retain unsupported-query errors."""
    command = ["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu",
               "--format=csv,noheader,nounits"]
    result = subprocess.run(command, text=True, capture_output=True, timeout=10, check=False)
    return {"gpu_csv": result.stdout.strip(), "gpu_query_exit": result.returncode,
            "gpu_query_error": result.stderr.strip()[:400],
            "cpu_cgroup": Path("/sys/fs/cgroup/cpu.stat").read_text(),
            "pressure": {kind: Path(f"/proc/pressure/{kind}").read_text()
                         for kind in ("cpu", "memory")}}


def main() -> None:
    """Run both ABBA orders with exact results, observed affinities and bounded CUDA storage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--master", default="192.168.192.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    import glm_roce_proxy_pin as placement
    import torch
    import torch.distributed as dist
    from b12x.comm.roce import AllReduce

    if args.report.exists():
        raise FileExistsError("Preserve prior measurements")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(0.008)
    dist.init_process_group("gloo", init_method=f"tcp://{args.master}:{args.port}",
                            world_size=2, rank=args.rank,
                            timeout=datetime.timedelta(seconds=45))
    runtime = None
    report: dict[str, Any] = {"status": "running", "rank": args.rank,
        "started_epoch": time.time(), "order": ORDER, "cases": CASES, "records": [],
        "message_bytes": 4 * 4096 * 2, "model_weights_loaded": False,
        "predeclared_gate": {
            "minimum_primary_p95_reduction_each_order": 0.05,
            "maximum_primary_median_regression_each_order": 0.02,
            "primary_case": "single_graph_spaced",
            "minimum_model_spacing_wall_reduction_each_order": 0.01,
            "model_spacing_case": "model_spaced_graph",
            "required": "Both ranks, both ABBA/BAAB groups; exact outputs; observed affinities",
            "meaning": "Component follow-up lead only, not a model speedup gate"},
        "limitations": ["Synthetic spacing is not model GEMM or memory traffic.",
            "Gloo trial barriers and CUDA event records affect absolute latency.",
            "Sleeping host spacing includes OS timer jitter.",
            "Neighbor GLM remains resident; activity guard rejects any serving interval.",
            "Eight order blocks are clustered samples, not independent per-call replicates."]}

    def save() -> None:
        temporary = args.report.with_suffix(".incomplete")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.report)

    try:
        save()
        runtime = AllReduce.from_exchange_group(exchange_group=dist.group.WORLD,
            device=torch.device("cuda", 0), max_size=4 * 1024**2, max_gather_bytes=16 * 1024**2)
        runtime.prepare((torch.bfloat16,))
        dist.barrier()
        tids = list(placement._STATE["tids"])
        if len(tids) != 1:
            raise RuntimeError("Require one observed native RDMA proxy thread")
        proxy_tid = tids[0]
        initial_mask = sorted(placement._STATE["tids"][proxy_tid])
        if len(initial_mask) < 2:
            raise RuntimeError("Unpinned control must have a scheduler choice of CPUs")
        report.update(proxy_tid=proxy_tid, original_proxy_affinity=initial_mask,
            gpu_name=torch.cuda.get_device_name(), torch_version=torch.__version__,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        base = (torch.arange(4 * 4096).reshape(4, 4096) % 31 - 15).float() / 16
        expected = ((base + 0) + (base + 1)).to(torch.bfloat16)
        x = (base + args.rank).to(dtype=torch.bfloat16, device="cuda")
        stream = torch.cuda.Stream()

        def barrier() -> None:
            torch.cuda.synchronize()
            dist.barrier()

        # Calibrate the synthetic GPU delay. Keep the resulting cycle count fixed in both arms.
        with torch.cuda.stream(stream):
            torch.cuda._sleep(1_000_000)
            torch.cuda._sleep(1_000_000)
            stream.synchronize()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            torch.cuda._sleep(1_000_000)
            b.record()
            b.synchronize()
            cycles = max(1, int(1_000_000 / a.elapsed_time(b)))
        report["spacing_calibration"] = {"target_gpu_ms": 1., "sleep_cycles": cycles,
            "million_cycle_ms": a.elapsed_time(b), "host_sleep_seconds": .001}

        prepared = {}
        for case in CASES:
            out = torch.empty_like(x)
            begin = torch.cuda.Event(enable_timing=True, external=True)
            end = torch.cuda.Event(enable_timing=True, external=True)
            graph = torch.cuda.CUDAGraph() if case["graph"] else None

            def enqueue(case=case, out=out, begin=begin, end=end) -> None:
                if case["repeats"] == 1 and case["spacing"] == "gpu":
                    torch.cuda._sleep(cycles)
                if case["repeats"] == 1 and case["spacing"] == "host":
                    time.sleep(.001)
                begin.record()
                for _ in range(case["repeats"]):
                    if case["repeats"] > 1:
                        if case["spacing"] == "gpu":
                            torch.cuda._sleep(cycles)
                        elif case["spacing"] == "host":
                            time.sleep(.001)
                    runtime.all_reduce(x, out=out)
                end.record()

            barrier()
            with torch.cuda.stream(stream):
                enqueue()
            barrier()
            if graph is not None:
                with runtime.capture(stream=stream), torch.cuda.graph(graph, stream=stream):
                    enqueue()
            prepared[case["name"]] = (enqueue, graph, begin, end, out)

        for block, mode in enumerate(ORDER):
            barrier()
            os.environ["GLM_ROCE_PROXY_CPUS"] = "0" if mode == "off" else "auto"
            placement.apply_current()
            observed = sorted(os.sched_getaffinity(proxy_tid))
            selected = sorted(placement.choose("auto", set(initial_mask), placement.big_cores()))
            if observed != (initial_mask if mode == "off" else selected):
                raise RuntimeError("The requested proxy placement was not applied")
            cases = CASES if block % 2 == 0 else tuple(reversed(CASES))
            for case in cases:
                enqueue, graph, begin, end, out = prepared[case["name"]]
                before = {"proxy": task_snapshot(proxy_tid), "main": task_snapshot(os.getpid()),
                          "system": telemetry()}
                gpu, wall = [], []
                for trial in range(case["trials"] + 3):
                    barrier()
                    started = time.perf_counter_ns()
                    with torch.cuda.stream(stream):
                        if graph is None:
                            enqueue()
                        else:
                            graph.replay()
                    end.synchronize()
                    elapsed = (time.perf_counter_ns() - started) / 1000
                    if trial >= 3:
                        gpu.append(begin.elapsed_time(end) * 1000)
                        wall.append(elapsed)
                runtime.check_health()
                actual = out.cpu()
                if not torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)):
                    raise RuntimeError("All-reduce changed result bytes")
                record = {"block": block, "mode": mode, "case": case["name"],
                    "observed_proxy_affinity": observed, "gpu_us": gpu, "wall_us": wall,
                    "gpu_median_us": statistics.median(gpu), "gpu_p95_us": percentile(gpu, .95),
                    "wall_median_us": statistics.median(wall), "wall_p95_us": percentile(wall, .95),
                    "output_sha256": hashlib.sha256(actual.view(torch.uint8).numpy().tobytes()
                                                    ).hexdigest(), "bits_exact": True,
                    "before": before,
                    "after": {"proxy": task_snapshot(proxy_tid),
                              "main": task_snapshot(os.getpid()), "system": telemetry()}}
                report["records"].append(record)
                save()
                print(json.dumps({k: record[k] for k in ("block", "mode", "case",
                    "gpu_median_us", "gpu_p95_us")}), flush=True)
        report.update(status="complete", finished_epoch=time.time(), runtime=runtime.stats(),
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        save()
        # A poisoned peer is bounded by the controller; never manipulate production resources.
        if runtime is not None:
            runtime.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
