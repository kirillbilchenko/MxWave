"""Replay retained native scores and validate canonical pool order on the actual GPU."""

from __future__ import annotations

import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from .canonical_sparse_prefill import canonical_topk


def telemetry() -> dict[str, Any]:
    """Retain clock and thermal context for every interleaved timing trial."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=clocks.current.sm,temperature.gpu,power.draw,pstate",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15, check=False,
    )
    return {"epoch": time.time(), "returncode": result.returncode, "values": result.stdout.strip()}


def qualify(fixture: Path, output: Path) -> dict[str, Any]:
    """Gate installation on repeated native selection, exact masking, expansion and graph replay."""
    from vllm.models.glm5next.nvidia.ops.kpool_compress import expand_pools_and_append_tail

    if output.exists():
        raise FileExistsError(output)
    raw = load_file(str(fixture))
    native = torch.ops._C.top_k_per_row_prefill
    if getattr(native, "_mxwave_canonical_prefill", False):
        raise ValueError("Replay native and canonical separately before installing the policy")
    retained: dict[str, torch.Tensor] = {}
    reports = []
    for rows in (8, 16, 64, 128, 3973):
        indices = (torch.arange(3973) if rows == 3973
                   else torch.linspace(0, 3972, rows).long())
        if rows != 3973:
            indices[:4] = torch.tensor([0, 1, 2, 3])
        values = {k: v[indices].cuda() for k, v in raw.items()}
        scores, starts, ends = (values[k] for k in ("scores", "starts", "ends"))
        expected = values["native_pools"].sort(dim=-1).values
        actual = torch.empty_like(expected)

        arguments = (scores, starts, ends, actual, rows, scores.stride(0), scores.stride(1), 512)

        def call(canonical: bool, args: tuple[Any, ...] = arguments) -> None:
            # Match the caller's invalid-slot initialization before each selection.
            # Native selection is allowed to leave unused output slots untouched.
            args[3].fill_(-1)
            if canonical:
                canonical_topk(native, *args)
            else:
                native(*args)

        native_order_changes = 0
        for _ in range(32):
            call(False)
            if not torch.equal(actual, values["native_pools"]):
                native_order_changes += 1
            if not torch.equal(actual.sort(dim=-1).values, expected):
                raise ValueError("Native replay changes selected membership on retained scores")
            call(True)
            if not torch.equal(actual, expected):
                raise ValueError("Canonical native selection does not repeat exactly")
        expanded = expand_pools_and_append_tail(actual.to(torch.int64), values["seq_lens"], 4)
        padded = torch.nn.functional.pad(expanded, (0, values["native_tokens"].shape[1] - expanded.shape[1]), value=-1)
        if not torch.equal(padded.sort(dim=-1).values, values["native_tokens"].sort(dim=-1).values):
            raise ValueError("Native expansion changes selected tokens or invalid-slot multiplicity")
        if torch.any((expanded < -1) | (expanded >= values["seq_lens"][:, None])):
            raise ValueError("Canonical expansion introduces an invalid or noncausal token")
        # Sorting/copying has fixed shapes per captured fixture; retain its replay output.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                call(True)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call(True)
        for _ in range(32):
            graph.replay()
            if not torch.equal(actual, expected):
                raise ValueError("Canonical top-k graph replay changes selected output")
        trials = []
        if rows in (16, 128, 3973):
            for _ in range(20):
                call(False)
                call(True)

            def elapsed(canonical: bool) -> float:
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                for _ in range(100):
                    call(canonical)
                end.record()
                end.synchronize()
                return begin.elapsed_time(end) / 100

            for trial in range(6):
                before = telemetry()
                order = [False, True, False] if trial % 2 == 0 else [True, False, True]
                ms = [elapsed(arm) for arm in order]
                drift = abs(ms[2] / ms[0] - 1)
                trials.append({"order": ["canonical" if a else "native" for a in order],
                               "milliseconds": ms, "repeat_relative_drift": drift,
                               "repeat_control_passed": drift <= 0.02,
                               "before": before, "after": telemetry()})
        stable = [t for t in trials if t["repeat_control_passed"]]
        deltas = []
        for trial in stable:
            middle = trial["milliseconds"][1]
            outside = statistics.mean((trial["milliseconds"][0], trial["milliseconds"][2]))
            deltas.append(middle - outside if trial["order"][1] == "canonical" else outside - middle)
        call(True)
        for name, tensor in {"canonical_pools": actual, "expanded_tokens": padded,
                             "source_rows": indices, **values}.items():
            retained[f"rows{rows}_{name}"] = tensor.detach().cpu().contiguous()
        reports.append({"rows": rows, "canonical_repeats_exact": 32, "graph_replays_exact": 32,
                        "native_order_changes": native_order_changes,
                        "selected_membership_and_masking_exact": True,
                        "trials": trials, "stable_timing_trials": len(stable),
                        "median_added_ms_stable_trials": statistics.median(deltas) if deltas else None})
        del graph, actual, expected, expanded, padded, values
    payload = output.with_suffix(".safetensors")
    save_file(retained, str(payload))
    payload.chmod(0o644)
    result = {"status": "passed", "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
              "payload_sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
              "fixtures": reports, "serving_overhead_qualified": False,
              "component_graph_replay_only": True, "repeat_limit": 0.02}
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result
