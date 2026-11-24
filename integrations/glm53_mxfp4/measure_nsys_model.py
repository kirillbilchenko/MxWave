"""One candidate load: matched B4 requests before, during and after an Nsight capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

from benchmark_full_model import allocations, batch, request, write
from check_full_model import SANITY, assess_chat, rpc


def validate_ownership(value: dict[str, Any]) -> None:
    """Require all four qualified adapters on both workers before measuring anything."""
    rows = value.get("results", [])
    if len(rows) != 2:
        raise ValueError("Require both TP2 ownership reports")
    for row in rows:
        kda, vocab = row["kda"], row["vocabulary"]
        scales = [scale for layer in row["layers"] for scale in layer["scales"]]
        if (kda["resident_layers"] != 68 or kda["dense_parameter_references_alive"]
                or kda["pool_bytes"] != 206569472 or kda["fused_max_tokens"] != 8
                or vocab["resident_layers"] != 2 or vocab["dense_parameter_references_alive"]
                or {r["kind"] for r in vocab["layers"]} != {"head", "embedding"}
                or row["layers_seen"] != 42 or sum(s["compressed"] for s in scales) != 83):
            raise ValueError("Combined adapter coverage, ownership or scratch changed")


def main() -> None:
    """Retain every request and partial report; no stock efficacy/quality claim is made."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:18784")
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    path = args.root / "nsys-measurement.json"
    if path.exists():
        raise FileExistsError("Preserve previous diagnostic evidence")
    protocol = json.loads((args.root / "protocol.json").read_bytes())
    prompts = protocol["prompts"]
    if len(prompts) != 4 or any(len(ids) != 256 for ids in prompts):
        raise ValueError("Require the frozen B4/256-token input")
    state: dict[str, Any] = {"status": "running", "phase": "ownership",
        "started_epoch": time.time(), "stages": [],
        "protocol_sha256": hashlib.sha256((args.root / "protocol.json").read_bytes()).hexdigest(),
        "purpose": "Candidate-only critical-path and active-instrumentation diagnostic",
        "limitations": ["Before/after remain under the inactive Nsight launcher and CUDA NVTX wrapper.",
            "This does not measure an uninjected server or compare stock versus compressed.",
            "One captured trial estimates instrumentation cost; it is not an efficacy benchmark."]}

    def save(phase: str) -> None:
        state.update(phase=phase, updated_epoch=time.time())
        write(path, state)

    profiling = False
    try:
        state["ownership_before"] = allocations(args.base)
        validate_ownership(state["ownership_before"])
        state["loaded_weight_checks"] = rpc(args.base, "mxwave_checkpoint_check")
        if any(row["status"] != "passed" for row in state["loaded_weight_checks"]):
            raise RuntimeError("Selected loaded weights differ from the qualified source")
        save("source_checked")
        state["mhc_warmup"] = rpc(args.base, "mxwave_mhc_warmup", [2048])
        state["chat_checks"] = []
        for prompt, expected in SANITY:
            result = request(args.base, "/v1/chat/completions", {
                "model": "mxwave-glm53-pilot", "messages": [{"role": "user", "content": prompt}],
                "chat_template_kwargs": {"reasoning_effort": "low"}, "temperature": 0,
                "seed": 42, "max_tokens": 256})
            state["chat_checks"].append(assess_chat(result["choices"][0], expected))
            save("chat_checked")
        if not all(row["passed"] for row in state["chat_checks"]):
            raise RuntimeError("Candidate chat sanity gate failed")
        state["warmups"] = [batch(args.base, prompts, protocol["tokens"]) for _ in range(2)]
        for label, count in (("before", 5), ("captured", 1), ("after", 5)):
            stage: dict[str, Any] = {"label": label, "trials": [], "started_epoch": time.time()}
            state["stages"].append(stage)
            stage["worker_before"] = rpc(args.base, "mxwave_nsys_observe", [label == "captured"])
            if any(row["kineto_profiler_enabled"] for row in stage["worker_before"]):
                raise RuntimeError("A PyTorch profiler is active; refuse competing CUPTI clients")
            save(label)
            if label == "captured":
                # CUDA-only vLLM profiler: issues cudaProfilerStart/Stop on both workers.
                profiling = True
                request(args.base, "/start_profile", {})
            try:
                for _ in range(count):
                    stage["trials"].append(batch(args.base, prompts, protocol["tokens"]))
                    save(label)
            finally:
                if profiling:
                    request(args.base, "/stop_profile", {}, timeout=240)
                    profiling = False
            stage["worker_after"] = rpc(args.base, "mxwave_nsys_observe", [False])
            stage.update(finished_epoch=time.time(), median_tokens_per_second=statistics.median(
                trial["aggregate_tokens_per_second"] for trial in stage["trials"]))
            save(label + "_complete")
            if label == "captured":
                # Nsight may generate its report while the target remains alive. Keep
                # that CPU/disk work outside the subsequent timing window.
                started, previous, stable = time.monotonic(), None, 0
                while time.monotonic() - started < 180:
                    reports = [row["capture_reports"] for row in
                               rpc(args.base, "mxwave_nsys_observe", [False])]
                    present = all(len(rows) == 1 and rows[0]["bytes"] > 0 for rows in reports)
                    stable = stable + 1 if present and previous == reports else 0
                    previous = reports
                    if stable >= 3:
                        state["capture_generation_wait_seconds"] = time.monotonic() - started
                        state["capture_reports_before_after_stage"] = reports
                        break
                    time.sleep(2)
                else:
                    raise TimeoutError("Both Nsight reports must finish before after-capture timing")
        reference = [r["token_ids"] for r in state["stages"][0]["trials"][0]["requests"]]
        state["generated_ids_match"] = all(
            [r["token_ids"] for r in trial["requests"]] == reference
            for stage in state["stages"] for trial in stage["trials"])
        state["ownership_after"] = allocations(args.base)
        validate_ownership(state["ownership_after"])
        state.update(status="complete", finished_epoch=time.time())
        save("complete")
    except BaseException as exc:
        state.update(status="failed", error=repr(exc))
        save("failed")
        raise
    finally:
        if profiling:
            request(args.base, "/stop_profile", {}, timeout=240)


if __name__ == "__main__":
    main()
